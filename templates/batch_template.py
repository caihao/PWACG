import copy
import glob
import json
import os
import re
import time
import logging
from importlib import reload
import numpy as onp
# import ROOT
import itertools

import jinja2
from create_code import create_control, prepare_all_collection
import pandas as pd


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



class submit(base_batch):
    def __init__(self, generator_path):
        super().__init__(generator_path)
        self.generator_init(self.generator_dict)
        self.initial_prepare()
        self.read_pwa("fit")
        self.cgpwa_dir = "/project/caihao5/sean/CGPWA/"

    def create(self, render_dict):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        template = env.get_template("templates/batch_script/sean-pwa.sbatch")
        template_out = template.render(**render_dict)
        with open("temp.sbatch", "w",encoding="utf-8") as f:
            f.writelines(template_out)

    def ordinaryJob(self,n,scanX):
        print("submit {} job".format(n))
        render_dict = dict()
        iter_n = str(n)
        render_dict["run_num"] = iter_n
        render_dict["generator_id"] = "{{generator_id}}"
        render_dict["run_file"] = self.cgpwa_dir + "run/fit_{{generator_id}}_{}.py".format(iter_n)
        render_dict["log_file"] = "output/fit/fit_result_{0}/fit_{0}.log".format(iter_n)
        self.create(render_dict)
        result_dir = "output/fit/fit_result_"+iter_n
        if not os.path.exists(result_dir):
            os.system("mkdir {}".format(result_dir))
            os.system("cp -r config/ {}".format(result_dir))

        self.info["fit"]["randomseed"] = n*30 + 100
        self.info["fit"]["lambda_tfc"] = 0.0
        self.info["fit"]["total_frac"]["kk"] = scanX
        self.info["fit"]["total_frac"]["pipi"] = scanX
        self.info["fit"]["Cycles"] = 30
        self.info["fit"]["random"] = True
        self.info["fit"]["use_weight"] = False
        self.jinja_fit_info["fit"]["CodeScript"] = "fit_object_{{generator_id}}_{}.py".format(iter_n)
        self.jinja_fit_info["fit"]["RunScript"] = "fit_{{generator_id}}_{}.py".format(iter_n)
        self.jinja_fit_info["fit"]["ResultFile"] = result_dir
        self.jinja_fit()
        os.system("sbatch temp.sbatch")
        os.system("cp rendered_scripts/fit_object_{{generator_id}}_{}.py {}".format(iter_n,result_dir))

    def set_mod_info(self):
        exten_mod = list()
        {% for lh in lh_coll %}
        with open("config/extension/pwa_info_{{lh.tag}}.json","r") as f:
            pwa_info = json.load(f)
            exten_mod.append(pwa_info["mod_info"])
        {% endfor %}
        # print(exten_mod)
        mod_name = [mod["mod"] for mod in exten_mod[0]]
        print(mod_name)
        name_list = ["_".join(name.split("_")[0:-1]) for name in mod_name]
        name_list = sorted(list(set(name_list)),key=name_list.index)
        print(name_list)
        combine_list = [list(itertools.combinations(name_list, r)) for r in range(1,len(name_list)+1)]
        # print(combine_list)
        def get_freedom(combine_c):
            freeinfo = [["f0",6],["f2",12]] # kk
            # freeinfo = [["f0",10],["f2",22],["K1680",6]] # combine
            free = 0
            for name in combine_c:
                for clas in freeinfo:
                    if re.match(".*"+clas[0]+".*",name):
                        free += clas[1]
            return free
        combine_freedom = [[c,get_freedom(c)] for clas in combine_list for c in clas]
        # print(combine_freedom)
        mod_dict = dict()
        temp = list()
        for comb in combine_freedom:
            if not comb[1] in temp:
                temp.append(comb[1])
                mod_dict[str(comb[1])] = list()
            mod_dict[str(comb[1])].append(comb[0])
        print(len(list(mod_dict.keys())))
        # print(mod_dict[list(mod_dict.keys())[-1]])
        return mod_dict

    def generatorJob(self,N,tfc,n,scanX,min_index=0):
        n = "{}_{}".format(N,n)
        print("submit freedom number {}'s job".format(n))
        gen_render_dict = dict()

        result_dir = "output/fit/scanFreedom/fit_result_"+str(n)
        if not os.path.exists(result_dir):
            os.system("mkdir -p {}".format(result_dir))
        gen_render_dict["result_dir"] = result_dir
        gen_render_dict["freedom"] = scanX
        gen_render_dict["use_weight"] = False
        gen_render_dict["random"] = False
        gen_render_dict["tfc"] = tfc
        gen_render_dict["lambda_tfc"] = 1e4
        gen_render_dict["scan_cycles"] = 40
        gen_render_dict["codescript"] = "scanFreedom_object_{}.py".format(n)

        temp_json_pwa = list()
        {% for lh in lh_coll %}
        temp_json_pwa.append("config/baseline/pwa_info_{{lh.tag}}.json")
        # temp_json_pwa.append("{}/pwa_info_{{lh.tag}}.json.{}".format("output/fit/fit_result_kk/fit_result_kk_2",min_index))
        {% endfor %}
        temp_json_pwa.append("config/baseline/pwa_info_ctrl.json")
        gen_render_dict["json_pwa"] = temp_json_pwa

        gen_render_dict["jinja_fit_info"] = self.jinja_fit_info
        gen_render_dict["run_config"] = {**self.parameters["base"]["run_config"], **self.parameters["fit"]["run_config"]}
        gen_render_dict["data_config"] = {**self.parameters["base"]["data_config"], **self.parameters["fit"]["data_config"]}
        gen_render_dict["scan_lh_coll"] = self.info["combine"]["tag"]
        gen_render_dict["generator_id"] = self.generator_id

        env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        template = env.get_template("templates/batch_script/scanFreedom.py")
        template_out = template.render(**gen_render_dict)
        run_file = self.cgpwa_dir + "run/temp_{}.py".format(n)
        with open(run_file, "w",encoding="utf-8") as f:
            f.writelines(template_out)

        render_dict = dict()
        render_dict["run_num"] = n
        render_dict["generator_id"] = "{{generator_id}}"
        render_dict["run_file"] = run_file
        # render_dict["run_file"] = "/mnt/sean/CGPWA/"+run_file
        render_dict["log_file"] = result_dir+"/run.log"
        self.create(render_dict)
        os.system("sbatch temp.sbatch")
        os.system("cp run/temp_{}.py {}".format(n,result_dir))

    def submit(self):
        print("begain to submit job to slurm")
        frac_tfc = [1.2]*10
        # frac_tfc = onp.arange(1.0,1.3,0.05)

        # mod_info = self.set_mod_info()
        # mod_info = list(mod_info.items())

        # with open("output/fit/fit_result_kk/fit_result_kk_2/result_info.json",encoding='utf-8') as f:
        #     mydict = json.loads(f.read())
        # arr_lh = onp.array([float(mydict["result_"+str(j)]["fcn"]) for j in range(60)])
        # min_index = list(onp.where(arr_lh == arr_lh.min()))[0][0]

        # for N, tfc in enumerate(frac_tfc):
        #     for n, scanX in mod_info:
        #         self.generatorJob(N,tfc,n,scanX)

        for n, scanX in enumerate(frac_tfc):
            self.ordinaryJob(n,scanX)
    
    def scan_read_pwa(self, key):
        self.all_mod_info = list()
        for addr_pwa_info in self.json_pwa[key]:
            filename = addr_pwa_info
            if filename:
                with open(filename, encoding='utf-8') as f:
                    print(filename)
                    dict_json = json.loads(f.read())
                    if "mod_info" in dict_json:
                        self.all_mod_info.append(dict_json["mod_info"])
                    if "external_binding" in dict_json:
                        self._binding_point = {**self._binding_point, **dict_json["external_binding"]}
            else:
                print(" Warning! No such file \"{}\", You should run fit create such file".format(addr_pwa_info))

    def pull(self,n,num):
        result_dir = f"output/fit/fit_result_{n}"
        scanX = 1.03
        self.info["fit"]["lambda_tfc"] = 0.0
        self.info["fit"]["total_frac"]["kk"] = scanX
        self.info["fit"]["total_frac"]["pipi"] = scanX
        self.info["fit"]["Cycles"] = 1
        self.info["fit"]["random"] = False
        self.info["fit"]["use_weight"] = False
        self.jinja_fit_info["fit"]["CodeScript"] = "fit_object_{{generator_id}}_{}.py".format(n)
        self.jinja_fit_info["fit"]["RunScript"] = "fit_{{generator_id}}_{}.py".format(n)
        self.jinja_fit_info["fit"]["ResultFile"] = result_dir
        self.jinja_fit_info["fit"]["CodeTemplate"] = "batch_script/pull_template.py"

        self.json_pwa["fit"] = list()
        self.json_pwa["fit"].append("config/pwa_info_kk.json")
        self.scan_read_pwa("fit")
        self.initial_prepare()
        self.render_dict.update(my_pull_mc_path=f"data/split_data/real_data_{num}")
        self.jinja_fit()
        os.system(f"python run/fit_kk_{n}.py")
        time.sleep(1)

        self.json_pwa["fit"] = list()
        self.json_pwa["fit"].append("{}/pwa_info_kk.json.0".format(result_dir))
        self.scan_read_pwa("fit")
        self.initial_prepare()
        self.render_dict.update(my_pull_mc_path=f"data/split_data/real_data_{num}")
        self.jinja_fit()
        os.system(f"python run/fit_kk_{n}.py")

        os.system("cp rendered_scripts/fit_object_{{generator_id}}_{}.py {}".format(n,result_dir))
        os.rename(f"{result_dir}/pwa_info_kk.json.0", f"{result_dir}/pwa_info_kk.json.{num}")
        os.rename(f"{result_dir}/correlation.npy", f"{result_dir}/correlation.npy.{num}")

    def pull_run(self,n,b,e):
        for i in range(b,e):
            self.pull(n,i)
    
    def submit_pull(self):
        epoch = 10
        batch_size = 10
        for i in range(epoch):
            result_dir = f"output/fit/fit_result_{i}"
            if not os.path.exists(result_dir):
                os.system("mkdir {}".format(result_dir))

            iter_i = str(i)
            print("submit {} job".format(iter_i))
            render_dict = dict()
            render_dict["run_num"] = iter_i
            render_dict["generator_id"] = "{{generator_id}}"
            render_dict["run_file"] = self.cgpwa_dir + "run/temp_pull_run_{}.py".format(iter_i)
            render_dict["log_file"] = "output/fit/fit_result_{0}/fit_{0}.log".format(iter_i)
            self.create(render_dict)

            render_dict["begin"] = batch_size*i
            render_dict["end"] = batch_size*(i+1)
            env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
            template = env.get_template("templates/batch_script/batch_pull.py")
            template_out = template.render(**render_dict)
            with open(f"run/temp_pull_run_{iter_i}.py", "w",encoding="utf-8") as f:
                f.writelines(template_out)

            os.system("sbatch temp.sbatch")





class scan(base_batch):
    def __init__(self,generator_path):
        super().__init__(generator_path)
        self.generator_init(self.generator_dict)
        self.initial_prepare()
        self.read_pwa("fit")
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        Logger = fit_object.Logger("fit")
        self.batch_args = fit_object.args()
        {% for key, value in run_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}

    def scan_read_pwa(self, key):
        self.all_mod_info = list()
        for addr_pwa_info in self.json_pwa[key]:
            filename = addr_pwa_info
            if filename:
                with open(filename, encoding='utf-8') as f:
                    print(filename)
                    dict_json = json.loads(f.read())
                    if "mod_info" in dict_json:
                        self.all_mod_info.append(dict_json["mod_info"])
                    if "external_binding" in dict_json:
                        self._binding_point = {**self._binding_point, **dict_json["external_binding"]}
            else:
                print(" Warning! No such file \"{}\", You should run fit create such file".format(addr_pwa_info))

    def scan_lh(self):
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        self.info["fit"]["Cycles"] = 1
        self.info["fit"]["lambda_tfc"] = 0.0
        self.info["fit"]["use_weight"] = False
        self.info["fit"]["random"] = False
        self.info["fit"]["total_frac"]["kk"] = 1.2
        self.info["fit"]["total_frac"]["pipi"] = 1.2
        self.jinja_fit_info["fit"]["CodeTemplate"] = "batch_script/scanLikelihood.py"
        Cycles = 1
        result_dir_list = glob.glob("output/fit/fit_result_kk")
        # result_dir_list = glob.glob("output/fit/scanFreedom/fit_result_*")
        # dir_index = [float(_dir.split("_")[-1]) for _dir in result_dir_list]
        # zip_dir = sorted(zip(dir_index,result_dir_list))
        # result_dir_list = [_dir[1] for _dir in zip_dir]
        # print(result_dir_list)

        for i in range(len(result_dir_list)):
            result_dir = result_dir_list[i]
            print(result_dir)
            result_info = dict()
            if os.path.isfile(os.path.join(result_dir,"have_scan_lh")):
                print("have scan this path")
                continue
            if os.path.exists(result_dir):
                for n in range(Cycles):
                    self.json_pwa["fit"] = list()
                    {% for lh in lh_coll %}
                    self.json_pwa["fit"].append("{}/pwa_info_{{lh.tag}}.json.{}".format(result_dir,n))
                    # self.json_pwa["fit"].append("{}/pwa_info_{{lh.tag}}.json.{}".format(result_dir,n))
                    {% endfor %}
                    self.json_pwa["fit"].append("config/total/pwa_info_ctrl.json")
                    self.initial_prepare()
                    self.scan_read_pwa("fit")
                    self.jinja_fit()
                    reload(fit_object)
                    cl = fit_object.Control(self.batch_args)
                    cl.run_multiprocess()
                    cl.get_result_dict()
                    print("result :",cl.fcn)
                    result_info["result_"+str(n)] = cl.fcn
                print(result_info)

                result_info_file = result_dir + "/result_info.json"
                if os.path.isfile(result_info_file):
                    with open(result_info_file, encoding='utf-8') as f:
                        result_info_json = json.loads(f.read())
                else:
                    result_info_json = dict()
                for key, value in result_info.items():
                    result_info_json[key]["likelihood"] = value
                with open(result_info_file, "w") as f:
                    json.dump(result_info_json, f)
                # os.system("touch {}".format(os.path.join(result_dir,"have_scan_lh")))

    def scan_frac(self):
        {% for codescript in jinja_draw_info.draw_wt.CodeScript %}
        from rendered_scripts import {{codescript|replace('.py','')}} as draw_object_{{loop.index0}}
        {% endfor %}
        args = draw_object_0.args()
        {% for key, value in run_config.items() -%}
        args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        args.{{key}} = {{value}}
        {% endfor %}
        self.generator_dict["jinja_draw_info"]["draw_wt"]["CodeTemplate"] = "batch_script/scanFraction.py"
        def run_draw():
            {% for codescript in jinja_draw_info.draw_wt.CodeScript %}
            reload(draw_object_{{loop.index0}})
            {% endfor %}
            wt_list = list()
            {% for codescript in jinja_draw_info.draw_wt.CodeScript %}
            draw = draw_object_{{loop.index0}}.Control(args)
            draw.run_multiprocess()
            draw.get_result_dict()
            wt_list.append(draw.fcn)
            {% endfor %}
            return wt_list
        # lambda_tfc = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
        # frac_tfc = [1.0,1.2,1.4,1.6,1.8,2.0]
        lambda_tfc = range(60)
        result_dir_list = glob.glob("output/fit/scanFreedom/fit_result_*")
        dir_index = [float(_dir.split("_")[-1]) for _dir in result_dir_list]
        zip_dir = sorted(zip(dir_index,result_dir_list))
        result_dir_list = [_dir[1] for _dir in zip_dir]
        print(result_dir_list)

        for i in range(len(result_dir_list)):
            result_dir = result_dir_list[i]
            print(result_dir)
            result_info = dict()
            if not os.path.exists(result_dir):
                print(result_dir," not exists")
                continue
            result_info = list()
            if os.path.isfile(os.path.join(result_dir,"have_scan_frac")):
                continue
            for n,_lambda in enumerate(lambda_tfc):
                self.json_pwa["draw"] = list()
                {% for lh in lh_coll %}
                self.json_pwa["draw"].append("{}/pwa_info_{{lh.tag}}.json.{}".format(result_dir,n))
                # self.json_pwa["draw"].append("{}/scan_{{lh.tag}}.json.{}".format(result_dir,n))
                {% endfor %}
                self.initial_prepare()
                self.scan_read_pwa("draw")
                self.jinja_draw()
                wt_list = run_draw()
                print(wt_list)
                result_info.append(wt_list)
            os.system("touch {}".format(os.path.join(result_dir,"have_scan_frac")))

            result_info_file = os.path.join(result_dir,"frac_info.json")
            if os.path.isfile(result_info_file):
                os.system("rm {}".format(result_info_file))

            with open(result_info_file, "w") as f:
                json.dump(result_info, f)

    def frac_jinja_fit(self,rand=False):
        print("jinja_fit:")
        self.mod_info = sum(self.all_mod_info, [])
        self.frac_prepare_all(rand)
        for module in self.jinja_fit_info.keys():
            self.render_dict.update(run_config = {**self.parameters["base"]["run_config"], **self.parameters[module]["run_config"]})
            self.render_dict.update(data_config = {**self.parameters["base"]["data_config"], **self.parameters[module]["data_config"]})
            address = self.jinja_fit_info[module]
            env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
            template = env.get_template("templates/" + address["CodeTemplate"])
            template_out = template.render(**self.render_dict)
            with open("rendered_scripts/" + address["CodeScript"], "w",encoding="utf-8") as f:
                f.writelines(template_out)
            run = env.get_template("templates/" + address["RunTemplate"])
            run_out = run.render(**self.render_dict)
            with open("run/" + address["RunScript"], "w",encoding="utf-8") as f:
                f.writelines(run_out)

    def frac_prepare_all(self,rand):
        self.append_calculate_func()
        self.get_mods_coll()
        self.get_data_coll()
        self.get_args_array()
        self.func_info_god()
        self.get_args_dict()
        self.get_binding_list()

        self.get_inintial_para()
        self.get_args_index_collection()

        # frac_index = self.args_index_coll["const"] + self.args_index_coll["theta"]
        frac_index = self.args_index_coll["mass"] + self.args_index_coll["width"] + self.args_index_coll["flatte"] + self.args_index_coll["const"] + self.args_index_coll["theta"]

        self.float_index = onp.sort(onp.array(self.float_index)[frac_index]).tolist()
        self.initial_parameters["float_index"] = self.float_index

        if rand:
            num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
            onp.random.seed(num_seed)
            disturb = 10
            _args_list = onp.array(self.initial_parameters["all_parameters"])
            _args_float = onp.array(_args_list[self.float_index])
            _args_float = _args_float*((onp.random.rand(_args_float.shape[0]) - 0.5) / disturb + 1.0)
            print(_args_float)
            _args_list[self.float_index] = _args_float
            self.initial_parameters["all_parameters"] = _args_list.tolist()

        self.render_dict.update(initial_parameters = self.initial_parameters)

        self.get_args_index_collection()
        self.write_all_amp_add()
        self.get_return()
        self.get_range_list()
        self.get_slit_args()
        self.get_lh_collection()

    def cal_args(self):
        # 使用前必须把中心值的配置文件放在config和output里，这样中心值的读出才没问题
        self.frac_jinja_fit()        
        intial_parameters = self.render_dict["initial_parameters"]
        args_list = onp.array(intial_parameters["all_parameters"])
        float_list = onp.array(intial_parameters["float_index"])
        source_args = args_list[float_list]
        def extract_number(file_name):
            # Extracting the number after the last dot
            number = file_name.split('.')[-1]
            return int(number)
        check_args_list = list()
        for i in range(10):
            # json_dir = f"pull_result_980_2340/pull_result_c/fit_result_{i}"
            json_dir = f"pull_result/fit_result_{i}"
            files_in_current_dir = os.listdir(json_dir)
            json_files = list()
            for f in files_in_current_dir:
                if os.path.isfile(os.path.join(json_dir, f)):
                    if f.startswith('pwa') and 'json' in f:
                        json_files.append(os.path.join(json_dir, f))
            json_files = sorted(json_files, key=extract_number)
            for file in json_files[:]:
                num = file.split(".")[-1]
                self.json_pwa["fit"] = list()
                self.json_pwa["fit"].append(file)
                # self.json_pwa["fit"].append("config/pwa_info_ctrl.json")
                self.info["fit"]["Cycles"] = 1
                self.info["fit"]["lambda_tfc"] = 0.0
                self.jinja_fit_info["fit"]["CodeTemplate"] = "fit_template.py"
                self.initial_prepare()
                self.read_pwa("fit")
                self.frac_jinja_fit()        
                intial_parameters = self.render_dict["initial_parameters"]
                args_list = onp.array(intial_parameters["all_parameters"])
                float_list = onp.array(intial_parameters["float_index"])
                args_float = args_list[float_list]
                correlation = onp.load(f"{json_dir}/correlation.npy.{num}")
                check_list = list()
                check_list.append(args_float)
                for _ in range(1000):
                    check = onp.random.multivariate_normal(args_float,correlation)
                    check_list.append(check)
                check_args = onp.array(check_list)
                check_args_list.append(check_args)
        onp.savez("output/draw/fraction_paras.npz",args_sample=onp.array(check_args_list),source_args=source_args)

    def cal_fraction_pull(self):
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        self.json_pwa["fit"] = list()
        {% for lh in lh_coll %}
        self.json_pwa["fit"].append("output/fit/fit_result_{{generator_id}}/pwa_info_{{lh.tag}}.json")
        {% endfor %}
        self.json_pwa["fit"].append("config/pwa_info_ctrl.json")
        self.info["fit"]["Cycles"] = 1
        self.info["fit"]["lambda_tfc"] = 0.0

        # calculate fraction sample
        self.jinja_fit_info["fit"]["CodeTemplate"] = "batch_script/calFractionError.py"
        self.initial_prepare()
        self.read_pwa("fit")
        self.frac_jinja_fit()
        reload(fit_object)
        cl = fit_object.Control(self.batch_args)
        cl.run_multiprocess()


    def Loop(self):
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        render_dict = dict()
        # lambda_tfc = self.ScanRange(0.1,10.0,20)
        # lambda_tfc = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5]
        # frac_tfc = [1.0,1.2,1.4,1.6,1.8,2.0]
        frac_tfc = [1.1,1.2,1.3,1.4,1.5]
        for n, scanX in enumerate(frac_tfc):
            result_dir = "output/fit/fit_result_"+str(n)
            if not os.path.exists(result_dir):
                os.system("mkdir -p {}".format(result_dir))
            self.info["fit"]["lambda_tfc"] = 1e4
            self.info["fit"]["total_frac"]["kk"] = scanX
            self.info["fit"]["total_frac"]["pipi"] = scanX
            self.info["fit"]["Cycles"] = 2
            self.info["fit"]["random"] = True
            self.info["fit"]["use_weight"] = False
            self.jinja_fit_info["fit"]["ResultFile"] = result_dir
            self.jinja_fit()
            reload(fit_object)
            cl = fit_object.Control(self.batch_args)
            cl.run_multiprocess()


    def stepBYstep(self):
        scanPwaDir = "output/fit/fit_result/fit_result_0"
        result_dir = "output/fit/scan/"
        runID = [0,1]
        # runID = onp.arange(0,6)
        for _id in runID:
            self.json_pwa["fit"] = list()
            {% for lh in lh_coll %}
            self.json_pwa["fit"].append(os.path.join(scanPwaDir,"pwa_info_{{lh.tag}}.json.{}".format(_id)))
            {% endfor %}

            lambda_tfc = onp.arange(1,2,0.05)
            # lambda_tfc = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
            result_dir_n = result_dir + "scan_{}".format(_id)
            if not os.path.exists(result_dir_n):
                os.system("mkdir -p {}".format(result_dir_n))
            for n, _lambda in enumerate(lambda_tfc):
                self.info["fit"]["lambda_tfc"] = 100
                self.info["fit"]["total_frac"]["kk"] = _lambda
                self.info["fit"]["total_frac"]["pipi"] = _lambda
                self.info["fit"]["Cycles"] = 1
                self.info["fit"]["random"] = False
                self.info["fit"]["use_weight"] = False
                self.jinja_fit_info["fit"]["CodeScript"] = "scan_object_{}.py".format(str(_id))
                self.jinja_fit_info["fit"]["RunScript"] = "scan_{}.py".format(str(_id))
                self.jinja_fit_info["fit"]["ResultFile"] = result_dir_n
                self.json_pwa["fit"].append("config/pwa_info_ctrl.json")
                self.initial_prepare()
                self.scan_read_pwa("fit")
                self.jinja_fit()

                scan_object = __import__("rendered_scripts.scan_object_{}".format(str(_id)),fromlist=[None])
                print(n)
                print(self.json_pwa)
                reload(scan_object)

                cl = scan_object.Control(self.batch_args)
                cl.run_multiprocess()

                {% for lh in lh_coll %}
                os.system("mv {} {}".format(os.path.join(result_dir_n, "pwa_info_{{lh.tag}}.json.0"),os.path.join(result_dir_n, "scan_{{lh.tag}}.json.{}".format(n))))
                {% endfor %}

                self.json_pwa["fit"] = list()
                {% for lh in lh_coll %}
                self.json_pwa["fit"].append(os.path.join(result_dir_n,"scan_{{lh.tag}}.json.{}".format(n)))
                {% endfor %}

    def draw_all(self):
        draw_object = __import__("rendered_scripts.draw_wt_object_{{generator_id}}",fromlist=[None])
        alldir = "output/scanSF/scan-select-5/fit_result_*"
        Cycles = 200 # files in each dir

        result_dir_list = glob.glob(alldir)
        dir_index = [float(_dir.split("_")[-1]) for _dir in result_dir_list]
        zip_dir = sorted(zip(dir_index,result_dir_list))
        result_dir_list = [_dir[1] for _dir in zip_dir]
        print("all dirs",result_dir_list)

        sof = list()
        runs = len(result_dir_list) # run which dir
        for i in range(runs):
            result_dir = result_dir_list[i]
            print("=="*30)
            print("run in",result_dir)
            print("=="*30)
            if os.path.exists(result_dir):
                for n in range(Cycles):
                    self.json_pwa["draw"] = list()
                    {% for lh in lh_coll %}
                    self.json_pwa["draw"].append("{}/pwa_info_{{lh.tag}}.json.{}".format(result_dir,n))
                    {% endfor %}
                    self.initial_prepare()
                    self.scan_read_pwa("draw")
                    self.jinja_draw()
                    reload(draw_object)
                    cl = draw_object.Control(self.batch_args)
                    cl.run_multiprocess()
                    cl.get_result_dict()
                    print("="*30)
                    print("sof",cl.fcn[0])
                    print("="*30)
                    sof.append(float(cl.fcn[0]))
        print("="*30)
        print("sof list",sof)
        print("="*30)
        onp.save("./output/scanSF/scan-select-5.npy",onp.array(sof))


    def Iterate(self):
        fit_object = __import__("rendered_scripts.fit_object_{{generator_id}}",fromlist=[None])
        # frac_tfc = [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2]
        frac_tfc = [1.15, 1.1, 1.05, 1.0, 0.95]

        self.info["fit"]["lambda_tfc"] = 1e2
        self.info["fit"]["Cycles"] = 1
        self.info["fit"]["random"] = False
        self.info["fit"]["boundary"] = True
        self.info["fit"]["use_weight"] = False
        result_dir = "output/fit/scan_sof"
        if not os.path.exists(result_dir):
            os.system("mkdir -p {}".format(result_dir))
        scan_v = dict()
        for n, scanX in enumerate(frac_tfc):
            {% for lh in lh_coll %}
            self.info["fit"]["total_frac"]["{{lh.tag}}"] = scanX
            {% endfor %}
            s_v = 1000.0
            s_n = 0
            for m in range(3):
                self.initial_prepare()
                self.scan_read_pwa("fit")
                self.jinja_fit()
                reload(fit_object)
                cl = fit_object.Control(self.batch_args)
                cl.run_multiprocess()
                cl.get_result_dict()
                if cl.fcn[0] < s_v:
                    s_v = cl.fcn[0]
                    s_n = m
                {% for lh in lh_coll %}
                os.system("cp output/fit/fit_result_{{generator_id}}/pwa_info_{{lh.tag}}.json.0 {0}/pwa_info_{{lh.tag}}.json.{1}".format(result_dir,m))
                {% endfor %}

            {% for lh in lh_coll %}
            os.system("cp {0}/pwa_info_{{lh.tag}}.json.{1} config/pwa_info_{{lh.tag}}.json".format(result_dir,s_n))
            {% endfor %}

            for mm in range(2):
                self.initial_prepare()
                self.scan_read_pwa("fit")
                self.jinja_fit()
                reload(fit_object)
                cl = fit_object.Control(self.batch_args)
                cl.run_multiprocess()
                cl.get_result_dict()
                s_v = cl.fcn[0]
                {% for lh in lh_coll %}
                os.system("cp output/fit/fit_result_{{generator_id}}/pwa_info_{{lh.tag}}.json.0 config/pwa_info_{{lh.tag}}.json")
                {% endfor %}

            scan_v[str(scanX)] = s_v
            {% for lh in lh_coll %}
            os.system("cp output/fit/fit_result_{{generator_id}}/pwa_info_{{lh.tag}}.json.0 config/pwa_info_{{lh.tag}}.json")
            os.system("cp output/fit/fit_result_{{generator_id}}/pwa_info_{{lh.tag}}.json.0 {0}/pwa_info_{{lh.tag}}.{1}.json".format(result_dir,scanX))
            {% endfor %}

            print("="*38)
            print(scan_v)
            print("="*38)


    def scan_select(self):
        select_object = __import__("rendered_scripts.select_object_{{generator_id}}",fromlist=[None])
        fit_object = __import__("rendered_scripts.fit_object_{{generator_id}}",fromlist=[None])

        for n in range(200):
            num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
            print("seed=",num_seed)
            onp.random.seed(num_seed)

            sl = select_object.Control(self.batch_args)
            sl.run_multiprocess()
            sl.get_result_dict()
            frac_tfc = onp.array([1.15])
            for sn, scanX in enumerate(frac_tfc):
                result_dir = "output/fit/fit_result_"+str(sn)
                if not os.path.exists(result_dir):
                    os.system("mkdir -p {}".format(result_dir))
                self.info["fit"]["lambda_tfc"] = 0.0
                self.info["fit"]["total_frac"]["kk"] = scanX
                self.info["fit"]["total_frac"]["pipi"] = scanX
                self.info["fit"]["Cycles"] = 1
                self.info["fit"]["random"] = False
                self.info["fit"]["use_weight"] = False
                self.jinja_fit_info["fit"]["ResultFile"] = result_dir
                self.jinja_fit()
                reload(fit_object)
                cl = fit_object.Control(self.batch_args)
                cl.run_multiprocess()
                cl.get_result_dict()
                os.system("cp {0}/pwa_info_kk.json.0 {0}/pwa_info_kk.json.{1}".format(result_dir,str(n)))
                # with open('scan_select.txt', 'a') as ssf:
                #     ssf.write(str(cl.fcn[0])+" "+str(scanX)+" "+str(sl.fcn[0])+"\n")




class sort_table():
    def __init__(self):
        self.significance_table = self.read_json("output/significance/significance_table.json")
        {% for tag in info.combine.tag %}
        self.fit_fraction_table_{{tag}} = self.read_json("output/draw/fit_fraction_table_{{tag}}.json")
        self.merge_table(self.fit_fraction_table_{{tag}})
        {% endfor %}

    def read_json(self, filename):
        with open(filename, encoding='utf-8') as f:
            json_dict = json.loads(f.read())
        return json_dict

    def merge_table(self, _dict):
        for sig_name in self.significance_table.keys():
            for mod in _dict:
                if re.match(sig_name+"*",mod["mod_name"]):
                    mod["significance"] = str(self.significance_table[sig_name])
        print("=*="*8)
        print("{}".format(json.dumps(_dict, ensure_ascii=False)))
        print("=*="*8)

class calculate_significance(base_batch):
    def __init__(self, generator_path):
        super().__init__(generator_path)
        self.generator_dict["json_pwa"]["fit"] = [
            {% for tag in info.combine.tag %}
            "output/significance/pwa_info_{{tag}}.json",
            {% endfor %}
            "config/pwa_info_ctrl.json"
        ]
        self.use_fit_object()

    def use_fit_object(self):
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        Logger = fit_object.Logger("fit")
        self.logger = logging.getLogger("fit")
        self.batch_args = fit_object.args()
        {% for key, value in run_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        self.batch_args.{{key}} = {{value}}
        {% endfor %}

    def fit_run(self):
        import rendered_scripts.{{jinja_fit_info.fit.CodeScript|replace('.py','')}} as fit_object
        reload(fit_object)
        cl = fit_object.Control(self.batch_args)
        cl.run_multiprocess()
        return cl

    def cal_origin_fcn(self):
        self.set_random(False)
        self.generator_init(self.generator_dict)
        self.batch_prepare("fit")
        self.jinja_fit()
        origin_obj = self.fit_run()
        origin_obj.get_result_dict()
        return origin_obj.fcn

    # def use_other_json(self):
    #     self.generator_dict["json_pwa"]["fit"] = [
    #         {% for tag in info.combine.tag %}
    #         "output/significance/1.4/pwa_info_{{tag}}.json",
    #         {% endfor %}
    #         "config/pwa_info_ctrl.json"
    #     ]
    #     self.generator_init(self.generator_dict)
    #     self.batch_prepare("fit")
    #     self.jinja_fit()
    #     self.sfc_name_dict = copy.deepcopy(self.sif_free_dict)
    #     self.sfc_mod_info = copy.deepcopy(self.all_mod_info)

    def cycle_calculate(self):
        self.set_cycles(1)
        origin_fcn = self.cal_origin_fcn()
        self.logger.info("origin fcn value : {}".format(origin_fcn))
        self.sfc_name_dict = copy.deepcopy(self.sif_free_dict)
        self.sfc_mod_info = copy.deepcopy(self.all_mod_info)
        self.set_random(False)
        self.generator_dict["annex_info"]["fit"]["lambda_tfc"] = 4
        for name in self.sfc_name_dict.keys():
            if re.match(".*f0_980.*",name):
                continue
            self.logger.info("remove mod : {}".format(name))
            self.all_mod_info = list()
            for layer0 in self.sfc_mod_info:
                temp = list()
                for mod in layer0:
                    if not re.match(name+"*", mod["mod"]):
                        temp.append(mod)
                self.all_mod_info.append(temp)

            self.generator_init(self.generator_dict)
            self.initial_prepare()
            self.jinja_fit()
            temp_obj = self.fit_run()
            temp_obj.get_result_dict()
            temp_fcn = temp_obj.fcn

            prob = ROOT.TMath.Prob(2.0*(temp_fcn[0] - origin_fcn[0]), self.sfc_name_dict[name][0])
            significance = ROOT.RooStats.PValueToSignificance(0.5*prob)
            self.logger.info("significance : {}".format(significance))
            self.sfc_name_dict[name] = significance

        self.logger.info("# table of all mods significance ")
        self.logger.info("{}".format(json.dumps(self.sfc_name_dict, ensure_ascii=False)))
        with open("output/significance/significance_table.json", "w") as json_file:
            json_file.write("{}".format(json.dumps(self.sfc_name_dict, ensure_ascii=False)))

    def save(self):
        localtime = time.localtime(time.time())
        new_dir = 'result_repo/{:0>2d}{:0>2d}-{:0>2d}{:0>2d}'.format(localtime[1],localtime[2],localtime[3],localtime[4])
        os.system('rm -rf {}'.format(new_dir))
        os.system('mkdir {}'.format(new_dir))
        os.system('cp -r output/ {}'.format(new_dir))

class calculate_branch(base_batch):
    def __init__(self,generator_path):
        super().__init__(generator_path)
        self.generator_dict["jinja_draw_info"]["draw_wt"]["CodeTemplate"] = "batch_script/branch_template.py"

    def jinja_draw(self,data_path,mc_path):
        print("jinja_draw:")
        for n, mod_info in enumerate(self.all_mod_info):
            self.mod_info = mod_info
            self.prepare_all()
            self.render_dict.update(data_path=data_path)
            self.render_dict.update(mc_path=mc_path)
            self.render_dict.update(run_config = {**self.parameters["base"]["run_config"], **self.parameters["draw_wt"]["run_config"]})
            self.render_dict.update(data_config = {**self.parameters["base"]["data_config"], **self.parameters["draw_wt"]["data_config"]})
            address = self.jinja_draw_info["draw_wt"]
            if "ResultFile" in address:
                self.render_dict.update(draw_result_file = address["ResultFile"][n])
            if "LassoResultFile" in address:
                self.render_dict.update(lasso_result_file = address["LassoResultFile"][n])
            env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
            template = env.get_template("templates/" + address["CodeTemplate"])
            template_out = template.render(**self.render_dict)
            with open("rendered_scripts/" + address["CodeScript"][n], "w",encoding="utf-8") as f:
                f.writelines(template_out)
            run = env.get_template("templates/" + address["RunTemplate"])
            run_out = run.render(**self.render_dict)
            with open("run/" + address["RunScript"], "w",encoding="utf-8") as f:
                f.writelines(run_out)

    def run_draw(self):
        {% for codescript in jinja_draw_info.draw_wt.CodeScript %}
        from rendered_scripts import {{codescript|replace('.py','')}} as draw_object_{{loop.index0}}
        reload(draw_object_{{loop.index0}})
        {% endfor %}
        args = draw_object_0.args()
        {% for key, value in run_config.items() -%}
        args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        args.{{key}} = {{value}}
        {% endfor %}
        wt_list = list()
        {% for codescript in jinja_draw_info.draw_wt.CodeScript %}
        draw = draw_object_{{loop.index0}}.Control(args)
        draw.run_multiprocess()
        wt_list.append(draw.fcn)
        {% endfor %}
        return wt_list

    def cal_wt_see(self):
        self.generator_init(self.generator_dict)
        self.batch_prepare("draw")
        self.jinja_draw("data/see","data/mc_int")
        wt_see = self.run_draw()
        return wt_see

    def cal_wt_truth(self):
        self.generator_init(self.generator_dict)
        self.batch_prepare("draw")
        self.jinja_draw("data/truth","data/mc_int")
        wt_truth = self.run_draw()
        return wt_truth

    def cal_branch(self):
        wt_see = self.cal_wt_see()
        wt_truth = self.cal_wt_truth()
        branch_kk = N_obs/(B_kk*(wt_see[0]/wt_truth[0])*N_tot)
        branch_pipi = N_obs/(B_pipi*(wt_see[1]/wt_truth[1])*N_tot)

