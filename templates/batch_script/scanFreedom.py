#!/usr/bin/env python3
# coding: utf-8
import copy
import glob
import json
import os
import sys
import re
import time
import logging
from importlib import reload
import numpy as onp
import itertools
import jinja2
import pandas as pd

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
from create_code import create_control, prepare_all_collection

class base_batch(create_control.Create_Code):
    def __init__(self, generator_path):
        with open(generator_path, encoding='utf-8') as f:
            self.generator_dict = json.loads(f.read())

    def set_cycles(self, num):
        self.generator_dict["annex_info"]["fit"]["Cycles"] = num
    
    def set_total_frac(self, num):
        self.generator_dict["annex_info"]["fit"]["total_frac"]["kk"] = num
        self.generator_dict["annex_info"]["fit"]["total_frac"]["pipi"] = num
    
    def set_random(self, bl):
        self.generator_dict["annex_info"]["fit"]["random"] = bl

    def generator_init(self,generator_dict):
        # initializer Prepare_All
        super().__init__(generator_dict)
    
    def batch_prepare(self,key):
        # read pwa info from json
        self.initial_prepare()
        self.read_pwa(key)
    

class scan(base_batch):
    def __init__(self,generator_path):
        super().__init__(generator_path)
        self.generator_init(self.generator_dict)
        self.initial_prepare()
        self.json_pwa["fit"] = {{json_pwa}}
        self.read_pwa("fit")
        self.scan_binding_point = self._binding_point

        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        Logger = fit_object.Logger("fit")
        self.batch_args = fit_object.args()
        {% for key, value in run_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}
    
    def prepare_mode_info(self,freedom,all_mod_info):
        _all_mod_info = copy.deepcopy(all_mod_info)
        exten_mod = list()
        {% for lh in scan_lh_coll %}
        with open("config/extension/pwa_info_{{lh}}.json","r") as f: 
            pwa_info = json.load(f)
            exten_mod.append(pwa_info["mod_info"])
        {% endfor %}

        _index = onp.random.randint(0,len(freedom))
        freedom_mod = freedom[_index]

        add_mod = list()
        for mod_info in exten_mod:
            for mod in mod_info:
                for fm in freedom_mod:
                    if re.match(fm+".*",mod["mod"]):
                        add_mod.append(mod)

        for mod_info in _all_mod_info:
            for admod in add_mod:
                for n,mod in enumerate(mod_info):
                    if "_".join([admod["mod"].split("_")[i] for i in[0,-1]]) == "_".join([mod["mod"].split("_")[i] for i in [0,-1]]) and admod["prop"]["prop_f"]["name"] == mod["prop"]["prop_f"]["name"]:
                        mod_info.insert(n,admod)
                        break
                if n+1 == len(mod_info):
                    mod_info.insert(n+1,admod)

        # for mod_info in _all_mod_info:
        #     for n,mod in enumerate(mod_info):
        #         print(mod["mod"])

        return _all_mod_info
    
    def Loop(self):
        render_dict = dict()
        freedom = {{freedom}}
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        onp.random.seed(num_seed)
        all_mod_info = copy.deepcopy(self.all_mod_info)

        result_info = dict()

        for n in range({{scan_cycles}}):
            self.info["fit"]["lambda_tfc"] = {{lambda_tfc}}
            self.info["fit"]["total_frac"]["kk"] = {{tfc}}
            self.info["fit"]["total_frac"]["pipi"] = {{tfc}}
            self.info["fit"]["Cycles"] = 1
            self.info["fit"]["random"] = {{random}}
            self.info["fit"]["use_weight"] = {{use_weight}}
            result_dir = os.path.join("{{result_dir}}","temp")
            if not os.path.exists(result_dir):
                os.system("mkdir -p {}".format(result_dir))
            self.jinja_fit_info["fit"]["ResultFile"] = result_dir
            self.jinja_fit_info["fit"]["CodeScript"] = "{{codescript}}"

            self.all_mod_info = self.prepare_mode_info(freedom,all_mod_info)
            self._binding_point = self.scan_binding_point
            self.render_dict["batch_json"] = "total"
            self.jinja_fit()

            import rendered_scripts.{{codescript|replace('.py','')}} as fit_object
            reload(fit_object)
            cl = fit_object.Control(self.batch_args)
            cl.run_multiprocess()
            cl.get_result_dict()
            if cl.fcn is not None:
                _success = True
            else:
                _success = False
            result_info["result_"+str(n)] = {"fcn":cl.fcn[0],"success":_success}
            cl.save_in_json(cl.fvalues,onp.zeros(len(cl.fvalues)),"{{result_dir}}",str(n),result_info)


if __name__ == "__main__":
    ob_scan = scan("config/generator_{{generator_id}}.json")
    ob_scan.Loop()
