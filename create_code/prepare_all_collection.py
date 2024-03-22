#!/usr/bin/env python
# coding: utf-8
import copy
import json
import os
import re
import numpy as onp

import jinja2

# os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Prepare_All():
    def __init__(self, dict_generator):
        self.generator_id = dict_generator["id"]
        self.jinja_fit_info = dict_generator["jinja_fit_info"]
        self.jinja_draw_info = dict_generator["jinja_draw_info"]
        self.json_pwa = dict_generator["json_pwa"]
        self.info = dict_generator["annex_info"]

        with open("config/parameters.json", encoding='utf-8') as f:
            dict_json = json.loads(f.read())
            self.parameters = dict_json["parameters"]
            self.draw_config = dict_json["draw_config"]
            self.CacheTensor = dict_json["CacheTensor"]
        
    def initial_prepare(self):
        self.render_dict = dict()
        self.mod_info = list()
        self._binding_point= dict()
        self.binding_point= dict()
        self.result_parameters = dict()
        self.all_result_parameters = dict()
        self.render_dict.update(generator_id = self.generator_id)
        self.render_dict.update(info = self.info)
        self.render_dict.update(jinja_fit_info = self.jinja_fit_info)
        self.render_dict.update(jinja_draw_info = self.jinja_draw_info)
        self.render_dict.update(json_pwa = self.json_pwa)
        self.render_dict.update(draw_config = self.draw_config)
        self.render_dict.update(mod_info = self.mod_info)
        self.render_dict.update(CacheTensor = self.CacheTensor)

    def prepare_all(self):
        self.append_calculate_func()
        self.get_mods_coll()
        self.get_data_coll()
        self.get_args_array()
        self.func_info_god()
        self.get_args_dict()
        self.get_binding_list()

        self.get_inintial_para()
        self.write_all_amp_add()
        self.get_args_index_collection()
        self.get_return()
        self.get_range_list()
        self.get_slit_args()
        self.get_lh_collection()
    
    def append_calculate_func(self):
        # 向mod_info中添加calculate_func标签
        calculate_func_coll = list()
        for mod in self.mod_info:
            mod.update(calculate_func = "{}_{}_{}".format(mod["amp"],mod["prop"]["prop_phi"]["name"],mod["prop"]["prop_f"]["name"]))
            mod.update(prop_name = "{}_{}".format(mod["prop"]["prop_phi"]["name"],mod["prop"]["prop_f"]["name"]))
            calculate_func_coll.append(mod["calculate_func"])
        self.calculate_func_coll = sorted(set(calculate_func_coll),key=calculate_func_coll.index)
        self.render_dict.update(calculate_func_coll = self.calculate_func_coll)
        # print("mod_info\n",self.mod_info)
        # 得到mod_name的集合
        self.mod_name_list = list()
        for mod in self.mod_info:
            self.mod_name_list.append(mod["mod"])
        # print(sorted(set(self.mod_name_list),key=self.mod_name_list.index))

    def get_mods_coll(self):
        # 针对不同的 prop, amp 的组合将所有的mod按照amp_prop的方式分类, key 为 amp_prop
        self.mods_collection = dict()
        for func in self.calculate_func_coll:
            self.mods_collection[func] = dict()
        for mod in self.mod_info:
            self.mods_collection[mod["calculate_func"]][mod["mod"]] = mod
        self.render_dict.update(mods_collection = self.mods_collection)
        # print("mods_collection:\n",self.mods_collection)
    
    def func_info_god(self):
        # func_info 就是把 mod_info 变成list, mod_info 是将 amp_prop 作为标签, 简并了同一类的其他字典
        self.func_info = list()
        for key_func, func in self.mods_collection.items():
            key = list(func.keys())[0]
            self.func_info.append(func[key])
        # print(self.func_info)
        self.render_dict.update(func_info = self.func_info)
        # 在func_info里增加参数分类
        for func in self.func_info:
            temp = list()
            all_temp = list()
            theta_const_temp = list()
            func[self.info["merge"]] = list()
            for key, args in func["args"].items():
                temp.append(args["name"])
            for para in temp:
                all_temp.append(para)
                if re.match(".*theta", para):
                    theta_const_temp.append(para)
                    func["theta"] = para
                if re.match(".*const", para):
                    theta_const_temp.append(para)
                    func["const"] = para
                if re.match(".*"+self.info["merge"],para):
                    func[self.info["merge"]].append(para)
            func["all_paras"] = sorted(set(all_temp),key=all_temp.index)
            func["compl_paras"] = [ i for i in all_temp if i not in theta_const_temp ]
        # print(self.func_info)
        # 在 func_info 里添加独立分波个数和同一个mod的共振态的个数
        for func in self.func_info:
            damp = 0
            for key, args in func["args"].items():
                if re.match(".*const", args["name"]):
                    damp+=1
            func["damp"] = damp
        # 在 func_info 里添加无sbc参数表
        for func in self.func_info:
            for key, prop in func["prop"].items():
                prop["_paras"] = prop["paras"][:-1]                   
        # 得到calculate_func_collection
        prop_coll = list()
        temp = list()
        for func in self.func_info:
            str_temp = " ".join(temp)
            if not re.match(".*" + func["prop_name"], str_temp):
                prop_coll.append(func)
            temp.append(func["prop_name"])
        # print("cal\n",prop_coll)
        self.render_dict.update(prop_coll = prop_coll)
        # 在func里添加mod名字和位置的集合
        self.name_index_complete = dict()
        index = 0
        for name in self.name_list_complete:
            self.name_index_complete[name] = index
            index += 1
        mod_name_list = list()
        for func in self.func_info:
            func["mod_name_list"] = list()
            for mod in self.mod_info:
                if re.match(func["calculate_func"],mod["calculate_func"]):
                    func["mod_name_list"].append({mod["mod"]:{key:self.name_index_complete[key] for key in mod["args"] if not (re.match(".*phi",key) or re.match(".*c",key) or re.match(".*t",key))}})
                    mod_name_list += [mod['mod'].replace("_"+tag,"") for tag in self.info["combine"]["tag"] if re.match(".*"+tag,mod["mod"])]
            # print("func[mod_name_list]\n",func["mod_name_list"])
        self.render_dict.update(mod_name_list = mod_name_list)
        dimensions_list = list()
        for func in self.func_info:
            for mod in self.mod_info:
                if re.match(func["calculate_func"],mod["calculate_func"]):
                    temp = 0
                    for arg in mod["args"].values():
                        if not "fix" in arg:
                            temp += 1
                    dimensions_list.append(temp)
        self.lasso_frac_dict = dict()
        for dimension, name in zip(dimensions_list, mod_name_list):
            self.lasso_frac_dict[name] = [dimension]
        self.render_dict.update(lasso_frac_dict=self.lasso_frac_dict)

        self.sif_free_dict = dict()
        self.all_mod_name_list = list()
        for fun in self.func_info:
            for mod in fun["mod_name_list"]:
                self.all_mod_name_list.append(list(mod.keys())[0])
        for lasso_name in self.lasso_frac_dict.keys():
            xi = 0
            for name in list(sorted(set(self.all_mod_name_list),key=self.all_mod_name_list.index)):
                if re.match(lasso_name+".*", name):
                    xi += 1
            self.sif_free_dict[lasso_name] = [xi*self.lasso_frac_dict[lasso_name][0]]

        # print(self.sif_free_dict)
        # print("mod_name_list\n",mod_name_list)
        # print("func_info\n",self.func_info)

    def get_data_coll(self):
        # 不变质量和振幅的集合
        self.data_collection = list()
        self.sbc_collection = list()
        self.amp_collection = list()
        for func in self.mod_info:
            for sbc in func["Sbc"].values():
                self.sbc_collection.append(sbc)
            self.data_collection.append(func["amp"])
        self.sbc_collection = sorted(set(self.sbc_collection),key=self.sbc_collection.index)
        self.data_collection = sorted(set(self.data_collection),key=self.data_collection.index)
        self.amp_collection = copy.deepcopy(self.data_collection)
        self.data_collection = self.data_collection + self.sbc_collection
        # print("data_collection\n",self.data_collection)
        # print("amp_collection\n",self.amp_collection)
        # print("sbc_collection\n",self.sbc_collection)
        self.render_dict.update(data_collection = self.data_collection)
        self.render_dict.update(sbc_collection = self.sbc_collection)
        self.render_dict.update(amp_collection = self.amp_collection)
        # print(self.data_collection)
        # print(self.amp_collection
    
    def get_args_array(self):
        def flat(nums):
            res = []
            for i in nums:
                if isinstance(i, list):
                    res.extend(flat(i))
                else:
                    res.append(i)
            return res
        # 从mod_info组合args数组
        self.args_collection = dict()
        for key_func, func in self.mods_collection.items():
            temp = dict()
            for key_mod, mod in func.items():
                for key_args, args in mod["args"].items():
                    temp[key_args] = mod["args"][key_args]
                    self.args_collection = {**self.args_collection, **temp}
        self.parameters_list = [self.args_collection[key]["name"] for key in self.args_collection]
        self.parameters_list = sorted(set(self.parameters_list), key=self.parameters_list.index)
        self.args_list_complete = list()
        self.name_list_complete = list()
        self.error_list_complete = list()
        # print("parameters_list: \n",self.parameters_list)
        # print("args_collection: \n",self.args_collection)
        for key_args in self.parameters_list:
            for key, args in self.args_collection.items():
                if re.match(key_args,args["name"]):
                    self.args_list_complete.append(args["value"])
                    self.name_list_complete.append(key)
                    if "error" in args:
                        self.error_list_complete.append(args["error"])
        # print("args_list: \n",self.args_list_complete)
        # print("name_list: \n",self.name_list_complete)
        # print("name_list: \n",self.error_list_complete)
        self.render_dict.update(args_collection = self.args_collection)
        self.render_dict.update(name_list = self.name_list_complete)
        self.render_dict.update(error_list = self.error_list_complete)

    def get_args_dict(self):
        # 简并后提取出参数名字，根据参数名字排序，并得到对应的参数位置
        foo_dict = dict()
        for para in self.parameters_list:
            i = 0
            for key, args in self.args_collection.items():
                if re.match(para, args["name"]):
                    i = i + 1
            foo_dict[para] = i
        begin = 0
        self.args_dict = dict()
        for tag, value in foo_dict.items():
            self.args_dict[tag] = [ begin + i for i in range(value)]
            begin = begin + value
        # print("args_dict\n",self.args_dict)
        # print("args_collection:\n",self.args_collection)
        # 在func_info里添加const的index
        all_const_index = list()
        for func in self.func_info:
            for key_arg, arg in self.args_dict.items():
                if re.match(func["const"], key_arg):
                    func["const_index"] = arg
                    all_const_index = all_const_index + arg
            func["num_mod"] = int(len(func["const_index"])/func["damp"])
        self.render_dict.update(all_const_index = all_const_index)
        self.render_dict.update(args_dict = self.args_dict)

    def checkKey(self, dict, key): 
        if key in dict.keys(): 
            return True
    
    def get_binding_list(self):
        # 数据结构为有向无环图，使用字典实现，因此图的终点的变量的名字不在外层字典的key里
        # 所以可以轻松地将外层的key，也就是除了终点以外的变量设置为不浮动，而终点的变量为浮动
        # 然后利用终点不出现在key里这一特点，可以将带有终点的字典排到字典集合的最前面，方便赋值
        # point 是指向的意思, 比如 A:{"point":"B","value":10}, 这里是A指向B, 即A绑定到B上, A随B变化
        # binding info in pwa json add to external_binding info in ctrl json
        for key, args in self.args_collection.items():
            if self.checkKey(args, "binding"):
                self._binding_point[key] = (args["binding"])
        # prepare binding
        for key, binding in self._binding_point.items():
            i = 0
            if key in self.name_list_complete:
                for name in self.name_list_complete:
                    if re.match(key, name):
                        begin = i
                    if re.match(binding["point"], name):
                        end = i
                    i = i + 1
                binding["goto"] = [begin, end]
        first = dict()
        last = dict()
        str_keys = " ".join(list(self._binding_point.keys()))
        for key, binding in self._binding_point.items():
            if key in self.name_list_complete:
                if re.match(".*"+binding["point"], str_keys):
                    last[key] = binding
                else:
                    first[key] = binding
        self._binding_point = {**first,**last}
        # print(self._binding_point)
        self.binding_point["goto0"] = list()
        self.binding_point["goto1"] = list()
        self.binding_point["bvalue"] = list()
        for key, _binding_point in self._binding_point.items():
            self.binding_point["goto0"].append(_binding_point["goto"][0])
            self.binding_point["goto1"].append(_binding_point["goto"][1])
            self.binding_point["bvalue"].append(_binding_point["value"])
        self.render_dict.update(binding_point = self.binding_point)

    def get_range_list(self):
        key_list = dict()
        for key, args in self.args_collection.items():
            if self.checkKey(args,"range"):
                key_list[key] = args["range"]
        range_dict = dict()
        for key in key_list:
            for name, i in self.name_index_complete.items():
                if re.match(key, name):
                    range_dict[key] = {"index":i, "range":key_list[key]}
        range_info = dict()
        self.range_dict = dict()
        for key, value in range_dict.items():
            range_info[str(value['index'])] = value['range']
        # print("range dict",range_info)
        mw_index = [int(key) for key in list(range_info.keys())]
        mw_index = [a for a in mw_index if a in self.float_index]
        mw_index = [self.float_index.index(idx) for idx in mw_index]
        mw_range = [value for value in list(range_info.values())]
        self.render_dict.update(mw_index = mw_index)
        self.render_dict.update(mw_range = mw_range)
        if self.info["fit"]["boundary"]:
            self.range_dict = range_info

    def get_inintial_para(self):
        # 将binding的除末尾的变量和fix的变量都视为不浮动
        self.initial_parameters = dict()
        self.initial_parameters["all_parameters"] = self.args_list_complete
        self.float_index = list()
        fix_list = list()
        for key, args in self.args_collection.items():
            if self.checkKey(args, "fix"):
                fix_list.append(key)
        binding_list = list(self._binding_point.keys())
        i = 0
        str_fix = " ".join(fix_list + binding_list)
        for name in self.name_list_complete:
            if not re.match(".*"+name, str_fix):
                self.float_index.append(i) 
            i = i + 1
        # print(self.float_index)
        self.initial_parameters["float_index"] = self.float_index
        self.render_dict.update(initial_parameters = self.initial_parameters)
    
    def get_slit_args(self):
        # 参数的解压缩接口
        # print("args_dict:\n",self.args_dict)
        # print(self.args_list_complete)
        # print(self.func_info)
        self.slit_args_dict = dict() # 使用了变换的参数解压缩字典
        self.trans_args_dict = dict() # 惩罚项
        for key, arg in self.args_dict.items():
            temp = list()
            trans_temp = list()
            for i in arg:
                if i in self.float_index:
                    if str(i) in self.range_dict:
                        # 边界条件代码
                        trans_temp.append("np.power({0}-args[{2}],2)/np.power({1},2)/2.0".format(self.range_dict[str(i)][0],self.range_dict[str(i)][1],self.float_index.index(i)))

                    temp.append("args[{}]".format(self.float_index.index(i)))
                elif i in self.binding_point["goto0"]:
                    wz = self.binding_point["goto0"].index(i)
                    igo = self.binding_point["goto1"][wz]
                    if str(igo) in self.range_dict:
                        trans_temp.append("np.power({0}-args[{2}],2)/np.power({1},2)/2.0".format(self.range_dict[str(igo)][0],self.range_dict[str(igo)][1],self.float_index.index(igo)))

                    # goto0 和 goto1 都 fix 的时候不能binding
                    if self.binding_point["goto1"][wz] in self.float_index:
                        temp.append("args[{}]{:+}".format(self.float_index.index(self.binding_point["goto1"][wz]), self.binding_point["bvalue"][wz]))
                    else:
                        temp.append(str(self.args_list_complete[i]))
                else:
                    if str(i) in self.range_dict:
                        trans_temp.append("np.power({0}-args[{2}],2)/np.power({1},2)/2.0".format(self.range_dict[str(i)][0],self.range_dict[str(i)][1],self.args_list_complete(i)))

                    temp.append(str(self.args_list_complete[i]))
            temp = ",".join(temp)
            self.slit_args_dict[key] = "np.array([{}])".format(temp)
            self.trans_args_dict[key] = trans_temp
            if re.match(".*const", key) or re.match(".*theta",key):
                for func in self.func_info:
                    for key_args, args in func["args"].items():
                        if re.match(key, args["name"]):
                            self.slit_args_dict[key] = "{}.reshape(-1,{})".format(self.slit_args_dict[key],func["damp"])
                            break
                    else:
                        continue
                    break
    
    def get_args_index_collection(self):
        who = ["const", "theta", "mass", "width"] 
        temp = dict()
        for w in who:
            temp[w] = list()
            for key, index in self.args_dict.items():
                if re.match(".*"+w, key):
                    for i in index:
                        if i in self.float_index:
                            temp[w].append(self.float_index.index(i))
        # 注释掉的部分是可以得到所有和const，theta，mass无关的变量
        # temp["width"] = list()
        # _who = [".*"+str for str in who]
        # reg_list = list(map(re.compile, _who))
        # for key, index in self.args_dict.items():
        #     if not any(regex.match(key) for regex in reg_list):
        #         for i in index:
        #             if i in self.float_index:
        #                 temp["width"].append(self.float_index.index(i))
        render_temp = temp

        temp = dict()
        for w in who:
            temp[w] = dict()
            for tag in self.info["combine"]["tag"]:
                temp[w][tag] = list()
                for key, index in self.args_dict.items():
                    if re.match(".*"+tag, key):
                        if re.match(".*"+w, key):
                            for i in index:
                                if i in self.float_index:
                                    temp[w][tag].append(self.float_index.index(i))
        temp["flatte"] = dict()
        who_flatte = [".*f980_rg.*",".*f980_g"]
        reg_list = list(map(re.compile, who_flatte))
        for tag in self.info["combine"]["tag"]:
            temp["flatte"][tag] = list()
            for key, index in self.args_dict.items():
                if re.match(".*"+tag,key):
                    if any(regex.match(key) for regex in reg_list):
                        for i in index:
                            if i in self.float_index:
                                temp["flatte"][tag].append(self.float_index.index(i))
        # temp["width"] = dict()
        # for tag in self.info["combine"]["tag"]:
        #     temp["width"][tag] = list()
        #     for key, index in self.args_dict.items():
        #         if re.match(".*"+tag,key):
        #             if not any(regex.match(key) for regex in reg_list):
        #                 for i in index:
        #                     if i in self.float_index:
        #                         temp["width"][tag].append(self.float_index.index(i))
        self.args_index_collection = temp

        render_temp["flatte"] = [i for x in temp["flatte"].values() for i in x]
        self.args_index_coll = render_temp
        self.render_dict.update(args_index_collection = render_temp)

    def write_all_amp_add(self):
        data = list()
        mc = list()
        lasso = list()
        combine_data_add_all_amp = dict()
        combine_lasso_data_add_all_amp = dict()
        combine_mc_add_all_amp = dict()
        self.lasso_frac = dict()
        for func in self.func_info:
            data.append("data_" + func["calculate_func"])
            mc.append("mc_" + func["calculate_func"])
            lasso.append("lasso_data_" + func["calculate_func"])
        for tag in self.info["combine"]["tag"]:
            temp_d = [d for d in data if re.match(".*"+tag,d)]
            combine_data_add_all_amp[tag] = (" + ".join(temp_d))
            temp_l = [d for d in lasso if re.match(".*"+tag,d)]
            self.lasso_frac[tag] = temp_l
            combine_lasso_data_add_all_amp[tag] = (" + ".join(["np.sum(np.sqrt(np.einsum(\"ljk->l\",dplex.dabs({0}))))".format(amp) for amp in temp_l]))
            temp_m = [d for d in mc if re.match(".*"+tag,d)]
            combine_mc_add_all_amp[tag] = (" + ".join(temp_m))
        self.combine_data_add_all_amp = combine_data_add_all_amp
        self.combine_lasso_data_add_all_amp = combine_lasso_data_add_all_amp
        self.combine_mc_add_all_amp = combine_mc_add_all_amp
    
    def get_return(self):
        self.data_return_dict = dict()
        self.lasso_data_return_dict = dict()
        self.mc_return_dict = dict()
        self.weight_return_dict = dict()
        self.wt_data_return_dict = dict()

        for tag in self.info["combine"]["tag"]:
            if re.match(".*"+tag, " ".join(self.sbc_collection)):
                data_size = float(onp.load("data/real_data/{}.npy".format([sbc for sbc in self.sbc_collection if re.match(".*"+tag,sbc)][0])).shape[0])
                self.data_return_dict[tag] = "return -np.sum(np.log(np.sum(dplex.dabs({}),axis=1))) + step_function".format(self.combine_data_add_all_amp[tag])
            else:
                data_size = 0
                self.data_return_dict[tag] = "return -np.sum(np.log(np.sum(dplex.dabs({}),axis=1))) + step_function".format(self.combine_data_add_all_amp[tag])

        for tag in self.info["combine"]["tag"]:
            if re.match(".*"+tag," ".join(self.sbc_collection)):
                data_size = float(onp.load("data/real_data/{}.npy".format([sbc for sbc in self.sbc_collection if re.match(".*"+tag,sbc)][0])).shape[0])
                self.wt_data_return_dict[tag] = "return -np.sum(self.wt_data_{1}*np.log(np.sum(dplex.dabs({0}),axis=1))) + step_function".format(self.combine_data_add_all_amp[tag],tag)
            else:
                data_size = 0
                self.wt_data_return_dict[tag] = "return -np.sum(self.wt_data_{1}*np.log(np.sum(dplex.dabs({0}),axis=1))) + step_function".format(self.combine_data_add_all_amp[tag],tag)

        for tag in self.info["combine"]["tag"]:
            self.lasso_data_return_dict[tag] = "return -np.sum(np.log(np.sum(dplex.dabs({0}),axis=1))) + np.power(10,{2})*({1})".format(self.combine_data_add_all_amp[tag], self.combine_lasso_data_add_all_amp[tag], self.info["fit"]["lambda_tfc"])
        for tag in self.info["combine"]["tag"]:
            self.mc_return_dict[tag] = "return np.sum(dplex.dabs({}))".format(self.combine_mc_add_all_amp[tag])
            # self.mc_return_dict[tag] = "return np.power(np.average(dplex.dabs({})),(1.0 + step_function))".format(self.combine_mc_add_all_amp[tag])
        for tag in self.info["combine"]["tag"]:
            self.weight_return_dict[tag] = "return np.sum(dplex.dabs({}),axis=1)".format(self.combine_data_add_all_amp[tag])

    def get_lh_collection(self):
        slit_args_dict = dict()
        trans_args_dict = dict()
        for tag in self.info["combine"]["tag"]:
            slit_args_dict[tag] = dict()
            trans_args_dict[tag] = dict()
            for key in self.slit_args_dict:
                if re.match(".*"+self.info["merge"], key):
                    slit_args_dict[tag][key] = self.slit_args_dict[key]
                    trans_args_dict[tag][key] = self.trans_args_dict[key]
                if re.match(".*"+tag, key):
                    slit_args_dict[tag][key] = self.slit_args_dict[key]
                    trans_args_dict[tag][key] = self.trans_args_dict[key]
        
        # print(self.func_info)
        func_differ = dict()
        for tag in self.info["combine"]["tag"]:
            func_differ[tag] = list()
            for func in self.func_info:
                if re.match(".*"+tag, func["mod"]):
                    func_differ[tag].append(func)
        
        bounding = dict()
        calc_wt = dict()
        for tag in self.info["combine"]["tag"]:

            trans_temp = list()
            for value in trans_args_dict[tag].values():
                if value:
                    trans_temp += value
            if self.info["fit"]["boundary"]:
                Param_limits = "+".join(trans_temp)
            else:
                Param_limits = "0.0"

            sum_frac = "sum_frac = " + "np.sum(dplex.dabs({})) \n".format("+".join(["np.einsum(\"mljk->mjk\", {})".format(l) for l in self.lasso_frac[tag]]))

            # 独立的 fit_frac 约束
            temp_frac = ["np.sum(np.heaviside(np.einsum(\"ljk->l\",dplex.dabs({0}))/sum_frac - 1.0,1.0))".format(amp) for amp in self.lasso_frac[tag]]

            # total_frac 约束
            temp_l = list()
            temp_r = list()
            temp_f = list()
            for amp in self.lasso_frac[tag]:
                if re.match(".*_l_",amp):
                    temp_l.append(amp)
                elif re.match(".*_r_",amp):
                    temp_r.append(amp)
                else:
                    temp_f.append(amp)
            temp_x = [temp_l[i]+"+"+temp_r[i] for i in range(len(temp_l))]
            temp_frac = temp_f + temp_x

            smooth_add_frac = "np.power({}-{},2.0)".format("+".join(["np.sum(np.einsum(\"ljk->l\",dplex.dabs({0}))/sum_frac)".format(amp) for amp in temp_frac]), self.info["fit"]["total_frac"][tag])

            step_function = "step_function = (" + "{0})*{1} ".format(smooth_add_frac, self.info["fit"]["lambda_tfc"]) + " + " + Param_limits

            bounding[tag] = sum_frac + "\n        " + step_function 

            # save mc weight
            total_wt = "total_wt = " + "np.sum(dplex.dabs({}),axis=1)".format("+".join(["np.einsum(\"mljk->mjk\", {})".format(l) for l in self.lasso_frac[tag]]))
            mod_wt = "wt_list = [total_wt,{}]".format(",".join(["np.einsum(\"ljk->lj\",dplex.dabs({0}))".format(amp) for amp in temp_frac]))
            calc_wt[tag] = [total_wt,mod_wt]

        lh_collection = list() 
        for tag in self.info["combine"]["tag"]:
            if not func_differ[tag] :
                continue
            temp = dict()
            temp.update(slit_args_dict = slit_args_dict[tag])
            temp.update(trans_args_dict = trans_args_dict[tag])
            temp.update(func_differ = func_differ[tag])
            temp.update(data_return_dict = self.data_return_dict[tag])
            temp.update(lasso_data_return_dict = self.lasso_data_return_dict[tag])
            temp.update(mc_return_dict = self.mc_return_dict[tag])
            temp.update(weight_return_dict = self.weight_return_dict[tag])
            temp.update(wt_data_return_dict = self.wt_data_return_dict[tag])
            temp.update(bounding = bounding[tag])
            temp.update(calc_wt = calc_wt[tag])
            temp.update(tag=tag)
            if self.info["fit"]["use_weight"]:
                temp.update(data_size = float(onp.sum(onp.load("data/weight/weight_{}.npy".format(tag)))))
            else:
                temp.update(data_size = float(onp.load("data/real_data/{}.npy".format([sbc for sbc in self.sbc_collection if re.match(".*"+tag,sbc)][0])).shape[0]))
            temp.update(mc_size = float(onp.load("data/mc_truth/{}.npy".format([sbc for sbc in self.sbc_collection if re.match(".*"+tag,sbc)][0])).shape[0]))
            # temp.update(mc_size = float(self.CacheTensor[tag]["mc"]))
            lh_collection.append(temp)
        self.render_dict.update(lh_coll = lh_collection)

    def read_result_json(self, path, _id):
        self.result_parameters[_id] = list()      
        files = os.listdir(path)
        files.sort()
        for tag in self.info["combine"]["tag"]:
            temp = list()
            for file in files:
                if os.path.isfile(path+"/"+file):
                    if re.match(".*{}.*{}".format(tag,self.generator_id), file):
                        with open(path+"/"+file, encoding='utf-8') as f:
                            dict_json = json.loads(f.read())
                            temp.append(self.prepare_args_list(dict_json["mod_info"]))
            self.result_parameters[_id].append(temp)

    def prepare_args_list(self, mod_info):
        mods_collection = dict()
        for mod in mod_info:
            mod.update(calculate_func = "{}_{}_{}".format(mod["amp"],mod["prop"]["prop_phi"]["name"],mod["prop"]["prop_f"]["name"]))
        for func in self.calculate_func_coll:
            mods_collection[func] = dict()
        for mod in mod_info:
            mods_collection[mod["calculate_func"]][mod["mod"]] = mod
        args_collection = dict()
        for key_func, func in mods_collection.items():
            temp = dict()
            for key_mod, mod in func.items():
                for key_args, args in mod["args"].items():
                    temp[key_args] = mod["args"][key_args]
                    args_collection = {**args_collection, **temp}
        parameters_list = [args_collection[key]["name"] for key in args_collection]
        parameters_list = sorted(set(parameters_list), key=parameters_list.index)
        args_list_complete = list()
        for key_args in parameters_list:
            for key, args in args_collection.items():
                if re.match(key_args,args["name"]):
                    args_list_complete.append(args["value"])
        return args_list_complete
    
    def read_all_result_json(self, path, _id):
        self.all_result_parameters[_id] = list()      
        temp_mod = list()
        files = os.listdir(path)
        files.sort()
        for file in files:
            if os.path.isfile(path+"/"+file):
                with open(path+"/"+file, encoding='utf-8') as f:
                    dict_json = json.loads(f.read())
                    temp_mod.append(dict_json["mod_info"])
                    f.close()