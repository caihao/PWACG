import time
import os

from multiprocessing import Array, Process, Pipe, Lock, Barrier, Value, Manager
import numpy as onp
import jax.numpy as np
from jax.config import config

from pwa import Class_pwa as Cp


{% macro args() %}
class args(object):
    def __init__(self):
        {% for key in run_config -%}
        self.{{key}} = None
        {% endfor %}
        {% for key in data_config -%}
        self.{{key}} = None
        {% endfor %}
        self.bic_delete_mods = list()

{% endmacro %}

{% macro ProcessReturns() %}
class ProcessReturns:
    def __init__(self):
        self.manager = Manager()
        self._dict = self.manager.dict(
            data_numbering=None,
            process_id=None,
            parameter_values=None,
            parameter_errors=None,
            min_fcn=None,
            timer=None, 
            gpu_id=None)
    def set(self, key, value):
        self._dict[key] = value

    def get(self, key):
        return self._dict[key]

    def info(self):
        logger.info("No. {} data on GPU{} for {} s".format(
            self._dict["data_numbering"],
            self._dict["gpu_id"],
            self._dict["timer"]))
        for value, error in zip(self._dict["parameter_values"], self._dict["parameter_errors"]):
            logger.debug("value={}, error={}".format(value, error))

{% endmacro %}


{% macro ProcessInitializer(data_path,mc_path) %}
class ProcessInitializers:
    def __init__(self):
        {% for data in data_collection %}
        self.data_{{data}} = None
        self.mc_{{data}} = None
        self.truth_{{data}} = None
        {% endfor %}
        {% for lh in lh_coll %}
        self.wt_data_{{lh.tag}} = None
        {% endfor %}

        self.data_numbering = None
        self.gpu_id = None
        self.gpu_memory_limit_percentage = None

    def __repr__(self):
        return "No. " + str(self.data_numbering) + " data batch for process initializing on gpu" + str(self.gpu_id)

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        return self


class Process_Initializer_Generator():
    def __init__(self
    {%- for key in data_config -%}
    , {{key}}=None
    {%- endfor -%}
    ):
    {% for key in data_config %}
        self.{{key}} = {{key}}
    {% endfor %}

    def reader_amp(self,file_name):
        amp_list = list()
        with onp.load(file_name) as amp:
            for amp_name in amp.files:
                amp_list.append((amp[amp_name])[:,0:2])
        return onp.array(amp_list)

    def data_npz(self,num):
        {% for tensor in amp_collection %}
        self.data_{{tensor}} = onp.load("{{data_path}}/{{tensor}}.npy")
        self.all_data_{{tensor}} = onp.array_split(self.data_{{tensor}},num,axis=1)
        {% endfor %}
        {% for sbc in sbc_collection %}
        data_{{sbc}} = onp.load("{{data_path}}/{{sbc}}.npy")
        self.all_data_{{sbc}} = onp.array_split(data_{{sbc}},num,axis=0)
        {% endfor %}
        {% for lh in lh_coll %}
        wt_data_{{lh.tag}} = onp.load("data/weight/weight_{{lh.tag}}.npy")
        self.all_wt_data_{{lh.tag}} = onp.array_split(wt_data_{{lh.tag}},num,axis=0)
        {% endfor %}

    def mc_npz(self,num):
        {% for tensor in amp_collection %}
        self.mc_{{tensor}} = onp.load("{{mc_path}}/{{tensor}}.npy")
        self.all_mc_{{tensor}} = onp.array_split(self.mc_{{tensor}},num,axis=1)
        {% endfor %}
        {% for sbc in sbc_collection %}
        mc_{{sbc}} = onp.load("{{mc_path}}/{{sbc}}.npy")
        self.all_mc_{{sbc}} = onp.array_split(mc_{{sbc}},num,axis=0)
        {% endfor %}

    def truth_npz(self,num):
        {% for tensor in amp_collection %}
        self.truth_{{tensor}} = onp.load("data/mc_truth/{{tensor}}.npy")
        self.all_truth_{{tensor}} = self.truth_{{tensor}}[:,0:150000]
        {% endfor %}
        {% for sbc in sbc_collection %}
        truth_{{sbc}} = onp.load("data/mc_truth/{{sbc}}.npy")
        self.all_truth_{{sbc}} = truth_{{sbc}}[0:150000]
        {% endfor %}

    def regular(self):
        {% for tensor in amp_collection %}
        self.re_{{tensor}} = onp.load("{{mc_path}}/{{tensor}}.npy")
        {% endfor %}
        {% for tensor in amp_collection %}
        regular_{{tensor}} = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_{{tensor}})**2,axis=2)),axis=1)
        self.all_data_{{tensor}} = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_{{tensor}}),regular_{{tensor}})
        self.all_mc_{{tensor}} = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_{{tensor}}),regular_{{tensor}})
        self.all_truth_{{tensor}} = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_{{tensor}}),regular_{{tensor}})
        {% endfor %}
    
    def process_initializer_generator(self):
        self.data_npz(self.data_slices)
        self.mc_npz(self.mc_slices)
        self.truth_npz(self.mc_slices)
        self.regular()
        logger.info("============ w i t h  n e x t ===============")

        data_numbering = 0
        event_num = self.mini_run
        while data_numbering<event_num:
            _ = ProcessInitializers()
            # 这里 %self.data_slices 是为了在多进程的时候使程序只靠 mini_run 控制循环次数
            {% for sbc in sbc_collection %}
            _.data_{{sbc}} = self.all_data_{{sbc}}[data_numbering%self.data_slices]
            _.mc_{{sbc}} = self.all_mc_{{sbc}}[data_numbering%self.mc_slices]
            _.truth_{{sbc}} = self.all_truth_{{sbc}}
            {% endfor %}
            {% for tensor in amp_collection %}
            _.data_{{tensor}} = self.all_data_{{tensor}}[data_numbering%self.data_slices]
            _.mc_{{tensor}} = self.all_mc_{{tensor}}[data_numbering%self.mc_slices]
            _.truth_{{tensor}} = self.all_truth_{{tensor}}
            {% endfor %}
            {% for lh in lh_coll %}
            _.wt_data_{{lh.tag}} = self.all_wt_data_{{lh.tag}}[data_numbering%self.data_slices]
            {% endfor %}
            _.data_numbering = data_numbering
            yield _
            data_numbering += 1
        return

        {{caller()}}
{% endmacro %}