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
    real_phipipi = False
    mc_phipipi = False
    draw_phikk = False
    draw_phipipi = False
    truth_phipipi = False
    truth_phikk = False

    real_phikk = True
    mc_phikk = True
    draw_phikk = True
    truth_phikk = True

    real_phipipi = True
    mc_phipipi = True
    draw_phipipi = True
    truth_phipipi = True

    if real_phikk == True :
        print("phikk real data")
        phikk_data = calc.data_info()
        phikk_data.address = "data/real_data/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 7602
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

    if real_phipipi == True :
        print("phipipi real data")
        phipipi_data = calc.data_info()
        phipipi_data.address = "data/real_data/"
        phipipi_data.filename = "Momentum_pipi.npz"
        phipipi_data.id = "pipi"
        phipipi_data.size = 11016
        phipipi_data.begain = 0
        phipipi_data.end = 999999
        phipipi_data.slices = 1
        mycdl = calc.CompoundDataForTensor(phipipi_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)
        os.system("cp data/real_data/*pipi* data/draw_mc/real_data/")
        calc.MergeMomentum("data/draw_mc/real_data/b123_pipi.npy","data/draw_mc/real_data/b124_pipi.npy")

    if mc_phipipi == True :
        print("phipipi mc integral")
        phipipi_data = calc.data_info()
        phipipi_data.address = "data/mc_int/"
        phipipi_data.filename = "Momentum_pipi.npz"
        phipipi_data.id = "pipi"
        phipipi_data.size = 3385941
        phipipi_data.begain = 0
        phipipi_data.end = {{CacheTensor.pipi.mc}}
        phipipi_data.slices = 100
        mycdl = calc.CompoundDataForTensor(phipipi_data)
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

    if draw_phipipi == True :
        print("draw phipipi mc integral and real data")
        phipipi_data = calc.data_info()
        # calculate data
        phipipi_data.address = "data/draw_data/"
        phipipi_data.filename = "Momentum_pipi.npz"
        phipipi_data.id = "pipi"
        phipipi_data.size = 3385941
        phipipi_data.begain = 500000
        phipipi_data.end = 1000000
        phipipi_data.slices = 10
        mycdl = calc.CompoundDataForTensor(phipipi_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

        # calculate mc
        ########################################################
        phipipi_data.address = "data/draw_mc/"
        phipipi_data.begain = 500000
        phipipi_data.end = 1000000
        phipipi_data.slices = 50
        mycdl = calc.CompoundDataForTensor(phipipi_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)
        # merg mc used to draw b123 b124 combine
        calc.MergeMomentum("data/draw_mc/b123_pipi.npy","data/draw_mc/b124_pipi.npy")
    
    if truth_phikk == True :
        print("pre phikk mc truth, pre toy mc")
        phikk_data = calc.data_info()
        # calculate data
        phikk_data.address = "data/mc_truth/"
        phikk_data.filename = "Momentum_kk.npz"
        phikk_data.id = "kk"
        phikk_data.size = 2000000
        phikk_data.begain = 0
        phikk_data.end = 8000000
        phikk_data.slices = 100
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

        phikk_data.address = "data/mc_truth/candidate/"
        phikk_data.begain = 0
        phikk_data.end = 2170000
        phikk_data.slices = 20
        mycdl = calc.CompoundDataForTensor(phikk_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

    if truth_phipipi == True :
        print("pre phipipi mc truth, pre toy mc")
        phipipi_data = calc.data_info()
        # calculate data
        phipipi_data.address = "data/mc_truth/"
        phipipi_data.filename = "Momentum_pipi.npz"
        phipipi_data.id = "pipi"
        phipipi_data.size = 2000000
        phipipi_data.begain = 0
        phipipi_data.end = 2000000
        phipipi_data.slices = 10
        mycdl = calc.CompoundDataForTensor(phipipi_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)

        phipipi_data.address = "data/mc_truth/candidate/"
        phipipi_data.begain = 0
        phipipi_data.end = 600000
        phipipi_data.slices = 10
        mycdl = calc.CompoundDataForTensor(phipipi_data)
        mycdl.g_id = 0
        mycdl.gpu_id = 1
        my_func = calc.pwa_func()
        my_func.fill_func()
        calc.Calculate(my_func,mycdl)