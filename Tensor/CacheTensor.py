import re
import os
import sys
import time

import numpy as onp
from jax.config import config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import FUNC as func


#########################################################
#                                                       #
# use multi GPU to calculate the Tensor of AMP class    # 
# use single U output but two same U function           #
#                                                       #
#########################################################

class data_info(object):
    def __init__(self):
        self.address = ""
        self.filename = ""
        self.id = ""
        self.size = 100
        self.begain = 0
        self.end = 100
        self.slices = 1

class CompoundDataForTensor(object):
    def __init__(self,data):
        self.Mom = onp.load(data.address+data.filename)
        self.momentum_mass = dict()
        self.event_num = data.slices
        self.infile(data.begain,data.end,self.event_num)
        self.g_id = None
        self.para_size = None
        self.id = data.id
        self.address = data.address

    def invm(self, Pbc):
        _Pbc = Pbc * onp.array([-1,-1,-1,1])
        return onp.sum(Pbc * _Pbc,axis=1)

    def infile(self,begin,end,num):
        Kp = self.Mom["Kp"][begin:end,:]
        Km = self.Mom["Km"][begin:end,:]
        Pip = self.Mom["Pip"][begin:end,:]
        Pim = self.Mom["Pim"][begin:end,:]
        phi = Kp + Km
        f = Pip + Pim
        b123 = phi + Pip
        b124 = phi + Pim
        # 保存四动量的动量和不变质量
        self.momentum_mass["phi_p"] = phi
        self.momentum_mass["f_p"] = f
        self.momentum_mass["b123_p"] = b123
        self.momentum_mass["b124_p"] = b124 
        self.momentum_mass["phi"] = self.invm(phi)
        self.momentum_mass["f"] = self.invm(f)
        self.momentum_mass["b123"] = self.invm(b123)
        self.momentum_mass["b124"] = self.invm(b124)
        self.momentum_mass["kst2_123"] = self.invm(b123)
        self.momentum_mass["kst2_124"] = self.invm(b124)
        print(Kp.shape)
        # 切片四动量为了计算张量的时候不报错
        self._Kp = onp.array_split(Kp, num, axis=0)
        self._Km = onp.array_split(Km, num, axis=0)
        self._Pip = onp.array_split(Pip, num, axis=0)
        self._Pim = onp.array_split(Pim, num, axis=0)
        self._phi = onp.array_split(phi, num, axis=0)
        self._f = onp.array_split(f, num, axis=0)

    def set_data_tensors(self, data_id):
        self.Kp = onp.squeeze(self._Kp[data_id],axis=None)
        self.Km = onp.squeeze(self._Km[data_id],axis=None)
        self.Pip = onp.squeeze(self._Pip[data_id],axis=None)
        self.Pim = onp.squeeze(self._Pim[data_id],axis=None)
        self.phi = onp.squeeze(self._phi[data_id],axis=None)
        self.f = onp.squeeze(self._f[data_id],axis=None)
    
        

class pwa_func():
    def __init__(self):
        self._dict = dict()
        self.output_num = None
        self.classification_list = ["phif0","phif2","u_b_l","u_b_r","u_rho_l","u_rho_r","u_kst0_l","u_kst0_r","u_kst2_l","u_kst2_r"]
        self.classification_dict = dict()
    
    def fill_func(self):
        my_func = func.FUNCTION()
        my_func.Modifiers()
        self._dict["phif001"] = [my_func.phif001,3]
        self._dict["phif021"] = [my_func.phif021,4]
        self._dict["phif201"] = [my_func.phif201,6]
        self._dict["phif221"] = [my_func.phif221,6]
        self._dict["phif222"] = [my_func.phif222,6]
        self._dict["phif223"] = [my_func.phif223,6]
        self._dict["phif243"] = [my_func.phif243,6]

        self._dict["u_rho_l"]= [my_func.u_rho_1,5]

        self._dict["u_b_l_ss"] = [my_func.u_SS_1,5]
        self._dict["u_b_l_sd"] = [my_func.u_SD_1,5]
        self._dict["u_b_l_ds"] = [my_func.u_DS_1,5]
        self._dict["u_b_l_dd"] = [my_func.u_DD_1,5]

        self._dict["u_rho_r"]= [my_func.u_rho_2,5]

        self._dict["u_b_r_ss"] = [my_func.u_SS_2,5]
        self._dict["u_b_r_sd"] = [my_func.u_SD_2,5]
        self._dict["u_b_r_ds"] = [my_func.u_DS_2,5]
        self._dict["u_b_r_dd"] = [my_func.u_DD_2,5]

        self._dict["u_kst0_l"] = [my_func.u_kst0_l,5]
        self._dict["u_kst0_r"] = [my_func.u_kst0_r,5]

        self._dict["u_kst2_l_12"] = [my_func.u_kst2_12_l,5]
        self._dict["u_kst2_r_12"] = [my_func.u_kst2_12_r,5]
        self._dict["u_kst2_l_32"] = [my_func.u_kst2_32_l,5]
        self._dict["u_kst2_r_32"] = [my_func.u_kst2_32_r,5]

    def get_func(self,str):
        func = (self._dict[str])[0]
        return func
    
    def get_paras(self,str):
        paras = (self._dict[str])[1]
        return paras
    
    def get_dict(self):
        _dict = self._dict
        return _dict

    def get_classification(self):
        for characters in self.classification_list:
            self.classification_dict[characters] = list()#遍历所有的振幅

        for characters in self.classification_list:
            for name in self._dict:
                if re.match(characters,name):
                    self.classification_dict[characters].append(name)

    def classification_info(self):
        for list_name in self.classification_dict:
            print("{} in the classification dictionary".format(list_name))
            for func_name in self.classification_dict[list_name]:
                print("function name : {}".format(func_name))



class TensorFunc(object):
    def __init__(self, cdl=None, myfunc=None):
        self._cdl = cdl
        self.para_size = self._cdl.para_size
        self.func = myfunc

    def init_para(self,_cdl):
        self.para_dict = dict()
        Kp = onp.array(_cdl.Kp)
        Km = onp.array(_cdl.Km)
        Pip = onp.array(_cdl.Pip)
        Pim = onp.array(_cdl.Pim)
        phi = onp.array(_cdl.phi)
        f = onp.array(_cdl.f)

        self.para_dict["3"] = [phi, Kp, Km]
        self.para_dict["4"] = [phi, f, Kp, Km]
        self.para_dict["5"] = [phi, Kp, Km, Pip,Pim]
        self.para_dict["6"] = [phi, f, Kp, Km, Pip, Pim]
    
    def get_para(self,n):
        return self.para_dict[str(n)]
        
    def run(self):
        g_id = self._cdl.g_id
        max_num = self._cdl.event_num - 1
        tensor = []
        while True:
            if g_id > max_num :
                break
            if g_id % 10 == 0:
                # print("run in : {} round".format(g_id))
                pass
            self._cdl.set_data_tensors(g_id)
            g_id += 1
            self.init_para(self._cdl)
            parameter = self.get_para(self.para_size)
            # print("parameter",onp.array(parameter).shape)
            _tensor = self.func(*parameter)
            tensor.append(_tensor)
            if onp.isnan(onp.sum(_tensor)):
                print("$$$$$$$$ nan $$$$$$$$$$")
        return (onp.array(tensor)).reshape(-1,4)


def Calculate(my_func, cdl):
    gpu_id = cdl.gpu_id
    data_id = cdl.id
    address = cdl.address
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config.update("jax_enable_x64", True)

    my_func.get_classification()
    my_func.classification_info()
    for character_name in my_func.classification_dict:
        total_tensor = list()
        # print(total_tensor)
        for func_name in my_func.classification_dict[character_name]: 
            print("run begain, run {} ".format(func_name))
            cdl.para_size = my_func.get_paras(func_name)
            function = my_func.get_func(func_name)
            Tf = TensorFunc(cdl,function)
            start = time.time()
            tensor = Tf.run()
            end = time.time()
            print("time {}".format(end-start))
            total_tensor.append(tensor[:,0:2])
        
        total_tensor = onp.array(total_tensor)
        print("tensor shape",total_tensor.shape)
        file_name = "{}{}_{}".format(address,character_name, data_id)
        print("save in {}".format(file_name))
        onp.save(file_name,total_tensor)
    
    for name in cdl.momentum_mass:
        file_name = "{}{}_{}".format(address,name,data_id)
        onp.save(file_name,cdl.momentum_mass[name])

def MergeMomentum(file1,file2):
    data1 = onp.load(file1)
    data2 = onp.load(file2)
    mergedata = onp.append(data1,data2)
    onp.save(file1,mergedata)
    onp.save(file2,mergedata)