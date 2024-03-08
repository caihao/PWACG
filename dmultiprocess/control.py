import os
import sys

from multiprocessing import Array, Process, Pipe, Lock, Barrier, Value, Manager
import numpy as onp
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import multiprocess as multi

{% macro Control(save_addr) %}
class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        {% for key, value in run_config.items() -%}
        self.{{key}} = args.{{key}}
        {% endfor %}
        self.data_config = dict()
        {% for key, value in data_config.items() -%}
        self.data_config["{{key}}"] = args.{{key}}
        {% endfor %}
        self.args_list = onp.array({{initial_parameters.all_parameters}})
        self.float_list = onp.array({{initial_parameters.float_index}})

        max_processes = self.max_processes
        self.process_pool = [Process()] * max_processes
        {% for lh in lh_coll -%}
        self.{{lh.tag}}_data_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        self.{{lh.tag}}_mc_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.{{lh.tag}}_data_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.{{lh.tag}}_mc_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.{{lh.tag}}_mc_grad_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.{{lh.tag}}_data_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.{{lh.tag}}_mc_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.{{lh.tag}}_mc_hvp_g = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.{{lh.tag}}_mc_hvp_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        {% endfor %}

    def grad_double(self,a,b,x):
        return (b-a)/2.0*onp.cos(x)
    
    def grad_a(self,x):
        return x/onp.sqrt(onp.power(x,2)+1)

    def grad_b(self,x):
        return -1.0*x/onp.sqrt(onp.power(x,2)+1)

    def likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        {% for lh in lh_coll %}
            self.{{lh.tag}}_data_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_data_likelihood_{{lh.tag}}(args_float)   
            self.{{lh.tag}}_mc_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_{{lh.tag}}(args_float)   
        {% endfor %}

    def thread_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        {% for lh in lh_coll %}
        # result_{{lh.tag}} = onp.sum(self.{{lh.tag}}_data_lh) + {{lh.data_size}}*onp.log(onp.sum(self.{{lh.tag}}_mc_lh))
        result_{{lh.tag}} = onp.sum(self.{{lh.tag}}_data_lh) + {{lh.data_size}}*onp.log(onp.sum(self.{{lh.tag}}_mc_lh)/{{lh.mc_size}})
        {% endfor %}
        result = 
        {%- for lh in lh_coll -%}
        + result_{{lh.tag}}
        {%- endfor %}
        return result

    def jit_grad_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        {% for lh in lh_coll %}
            self.{{lh.tag}}_data_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_data_likelihood_{{lh.tag}}(args_float)
            self.{{lh.tag}}_mc_grad_v[device_rank,_], self.{{lh.tag}}_mc_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_mc_likelihood_{{lh.tag}}(args_float)
        {% endfor %}

    def thread_grad_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_grad_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        {% for lh in lh_coll %}
        result_{{lh.tag}} = onp.sum(self.{{lh.tag}}_data_grad,axis=(0,1)) + {{lh.data_size}}*onp.sum(self.{{lh.tag}}_mc_grad,axis=(0,1))/onp.sum(self.{{lh.tag}}_mc_grad_v)
        {% endfor %}
        result = 
        {%- for lh in lh_coll -%}
        + result_{{lh.tag}}
        {%- endfor %}
        return result

    def mc_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        {% for lh in lh_coll %}
            self.{{lh.tag}}_mc_hvp_v[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_{{lh.tag}}(args_float)
        {% endfor %}

    def thread_mc_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.mc_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        {% for lh in lh_coll %}
        self.{{lh.tag}}_mc_hvp_value = onp.sum(self.{{lh.tag}}_mc_hvp_v)
        {% endfor %}

    def jit_hvp_in_sigle_device(self, args_float, any_vector, device_rank):
        for _ in range(self.threads_in_one_gpu):
        {% for lh in lh_coll %}
            self.{{lh.tag}}_data_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_data_fwdrev_{{lh.tag}}(args_float, any_vector)[1]
            self.{{lh.tag}}_mc_hvp_g[device_rank,_,:], self.{{lh.tag}}_mc_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_mc_fwdrev_{{lh.tag}}(args_float, any_vector)
        {% endfor %}

    def thread_hvp(self, args_float, any_vector):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_hvp_in_sigle_device, args = (args_float, any_vector, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        self.thread_mc_likelihood(args_float)
        {% for lh in lh_coll %}
        result_{{lh.tag}} = onp.sum(self.{{lh.tag}}_data_hvp, axis=(0,1)) + {{lh.data_size}}*(onp.sum(self.{{lh.tag}}_mc_hvp,axis=(0,1))/self.{{lh.tag}}_mc_hvp_value - onp.einsum("ikm,jln,n->m",self.{{lh.tag}}_mc_hvp_g,self.{{lh.tag}}_mc_hvp_g,any_vector)/(self.{{lh.tag}}_mc_hvp_value**2))
        {% endfor %}
        result = 
        {%- for lh in lh_coll -%}
        + result_{{lh.tag}}
        {%- endfor %}
        return result
    
    def require_compile(self, fcn, *args):
        logger.info("{} compile start".format(fcn.__name__))
        t1 = time.time()
        fcn(*args)
        t2 = time.time()
        logger.info("{} compile complete, time is {}".format(fcn.__name__,t2-t1))

    def my_callback(self, xk):
        logger.info(" fcn: {}".format(self.thread_likelihood(xk)))

    def compile_func(self):
        start = time.time()
        vector = onp.ones(self.args_float.shape)
        compile_likelihood = Thread(target = self.require_compile, args = (self.thread_likelihood, self.args_float))
        compile_grad = Thread(target = self.require_compile, args = (self.thread_grad_likelihood, self.args_float))
        compile_hvp = Thread(target = self.require_compile, args = (self.thread_hvp, self.args_float, vector))

        compile_hvp.start()
        compile_likelihood.start()
        compile_grad.start()

        compile_hvp.join()
        compile_likelihood.join()
        compile_grad.join()
        stop = time.time()
        logger.info("compile time: {}".format(stop-start))

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x),process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(process_initializer[0][0].gpu_memory_limit_percentage)
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # 用多少拿多少的选项
        config.update("jax_enable_x64", True)

        logger.info("fit_pull_sample begins!!!!")

        start_time = time.time()

        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id]) for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()
        {{caller()}}
        stop_time = time.time()
        logger.info("=================set return=====================")
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        process_returns.info()
        process_returns.set("timer", stop_time - start_time)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

        return

    def run_multiprocess(self):
        self.process_initializer = Process_Initializer_Generator(**self.data_config)
        process_initializer_generator = self.process_initializer.process_initializer_generator()
        self.process_returns = []
        self.start_time = time.time()
        while True:
            try:
                for _proc_numbering, _proc in enumerate(self.process_pool):
                    if not _proc.is_alive():
                        try:
                            _proc.join()
                        except AssertionError:
                            pass
                        finally:
                            # 创建多线程拆分模型, 注意! 一个进程只可以用来计算一个完整的拟合
                            # 线程模型是[i, j]形状, 对应[gpus, thread_in_gpus], 总线程为 gpus*thread_in_gpus
                            _proc_initializer_list = [[None for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
                            for i in range(self.thread_gpus):
                                for j in range(self.threads_in_one_gpu):
                                    with next(process_initializer_generator) as _proc_initializer_list[i][j]:
                                        logger.info("data: {}".format(_proc_initializer_list[i][j].data_numbering))
                                        _proc_initializer_list[i][j].gpu_memory_limit_percentage = self.max_processes_memory 
                                        _proc_initializer_list[i][j].gpu_id = i + _proc_numbering % self.processes_gpus # gpu的分配靠 thread_gpus 和 processes_gpus 和进程数调节
                                        _proc_initializer_list[i][j].total_gpu_id = self.total_gpu_id
                            
                            _proc_returns = ProcessReturns()
                            _proc_returns.set("data_numbering", [[_proc_initializer_list[i][j].data_numbering for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)])
                            self.process_pool[_proc_numbering] = Process(target=self.run, args=(_proc_initializer_list, _proc_returns))
                            self.process_pool[_proc_numbering].start()
                            self.process_returns.append(_proc_returns)
                            logger.info("=========== chambering data_numbering.{} ============".format(_proc_returns.get("data_numbering")))

            except StopIteration:
                for _ in self.process_pool:
                    try:
                        _.join()
                    except AssertionError:
                        pass
                break

        self.stop_time = time.time()
        logger.info("Total time: {}".format(self.stop_time - self.start_time))
    
    def get_result_dict(self):
        self.fvalues = (self.process_returns[0]._dict)["parameter_values"]
        self.fcn = (self.process_returns[0]._dict)["min_fcn"]

    def save_in_npz(self):
        logger.info("save in npz")
        total_values = []
        total_errors = []
        for _ in self.process_returns:
            total_values.append(_._dict["parameter_values"])
            total_errors.append(_._dict["parameter_errors"])
        np_values = onp.squeeze(onp.array(total_values), axis=None)
        np_errors = onp.squeeze(onp.array(total_errors), axis=None)
        logger.debug("{}".format(np_values.shape))
        logger.debug("{}".format(np_errors.shape))
        if total_values[0] is not None or total_errors[0] is not None:
            onp.savez("{{save_addr}}", fvalue=np_values, ferror=np_errors)

        logger.info("program over")
    
    def save_in_json(self, np_values, np_errors, save_addr = "", n = "",result_info={}):
        logger.info("save in json")
        json_addr_list = []
        {% for lh in lh_coll %}
        json_addr_list.append("pwa_info_{{lh.tag}}.json")
        {% endfor %}
        name_list = {{name_list}}
        mod_name_list = {{mod_name_list}}
        for json_addr in json_addr_list:
            filename = glob.glob(os.path.join("config/{{batch_json}}",json_addr+"*"))
            with open(filename[0], encoding='utf-8') as f:
                pwa_json = json.loads(f.read())
                mod_info = pwa_json["mod_info"]
                mod_index = [n for n, mod in enumerate(mod_info) if re.match(".*"+"_".join(mod["mod"].split("_")[0:-1])+".*"," ".join(mod_name_list))]
                mod_info = [mod_info[i] for i in mod_index]
                for i in range(len(name_list)):
                    for mod in mod_info:
                        for arg_key in mod["args"].keys():
                            if name_list[i] == arg_key:
                                mod["args"][arg_key]["value"] = float(np_values[i])
                                mod["args"][arg_key]["error"] = float(np_errors[i])
                _pwa_json = {"mod_info":mod_info} 
                json_str = json.dumps(_pwa_json, indent=4)
                with open(save_addr + "/{0}.{1}".format(json_addr, n), 'w') as json_file:
                    json_file.write(json_str)
        with open(save_addr + "/result_info.json", 'w') as f:
            json.dump(result_info,f)
{% endmacro %}