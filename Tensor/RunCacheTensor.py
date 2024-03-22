import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
from Tensor import CacheTensor as calc

# demo of use the calculate module

if __name__ == "__main__":
    real_phikk = False
    mc_phikk = False
    draw_phikk = False
    truth_phikk = False

    truth_phikk = True

    if real_phikk == True :
        print("phikk real data")
        phikk_data = calc.data_info()
        phikk_data.address = "data/real_data/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 10000
        phikk_data.begain = 0
        phikk_data.end = 999999
        phikk_data.slices = 1
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)
        os.system("cp data/real_data/*kk* data/draw_mc/real_data/")
        calc.MergeMomentum("data/draw_mc/real_data/b123_kk.npy","data/draw_mc/real_data/b124_kk.npy")

    if  mc_phikk == True :
        print("phikk mc integral")
        phikk_data = calc.data_info()
        phikk_data.address = "data/mc_int/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 1429100
        phikk_data.begain = 0
        phikk_data.end = {{CacheTensor.kk.mc}}
        phikk_data.slices = 100
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

    if draw_phikk == True :
        print("draw phikk mc integral and real data")
        phikk_data = calc.data_info()
        # calculate data
        phikk_data.address = "data/draw_data/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 1429100
        phikk_data.begain = 500000
        phikk_data.end = 1000000
        phikk_data.slices = 10
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

        # calculate mc
        ########################################################
        phikk_data.address = "data/draw_mc/"
        phikk_data.begain = 500000
        phikk_data.end = 1000000
        phikk_data.slices = 50
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)
        # merg mc used to draw b123 b124 combine
        calc.MergeMomentum("data/draw_mc/b123_kk.npy","data/draw_mc/b124_kk.npy")

    
    if truth_phikk == True :
        print("pre phikk mc truth, pre toy mc")
        phikk_data = calc.data_info()
        # calculate data
        phikk_data.address = "data/mc_truth/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 2000000
        phikk_data.begain = 0
        phikk_data.end = 400000
        phikk_data.slices = 100
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)