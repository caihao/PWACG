import numpy as onp
import matplotlib.pyplot as plt
import json
import glob
import os
from scipy.optimize import curve_fit


def readData(filename,batch_size):
    files_list = glob.glob(filename+"*")
    freedom_list = [int(_file.split("_")[-1]) for _file in files_list]
    result_info = sorted(zip(freedom_list,files_list))
    print("result info of patch :",result_info)
    print("patch size :",len(result_info))

    data = dict()
    for freedom, file_dir in result_info:
        file_dir = file_dir + "/result_info.json"
        if not os.path.exists(file_dir):
            print("file: " + file_dir + " not exists")
            continue

        with open(file_dir,encoding='utf-8') as f:
            result_json = json.loads(f.read())

        likelihood = list()        
        for i in range(batch_size):
            likelihood.append(float(result_json["result_"+str(i)]["fcn"]))

        data[str(freedom)] = likelihood

    return data

class PrePross(object):
    # return 的是 data 的字典
    def __init__(self,data):
        self.freedom_list = list()
        self.likelihood_list = list()
        for freedom, likelihood in data.items():
            likelihood = onp.array(likelihood)
            freedom = int(freedom)
            self.freedom_list.append(freedom)
            self.likelihood_list.append(likelihood)

    # 产生最小的 data dict 
    def miniLikelihood(self):
        mini_data = dict()
        for n, likelihood in enumerate(self.likelihood_list):
            mini_data[str(self.freedom_list[n])] = float(likelihood.min())

        return mini_data


# 画图的方法和数据处理
class Draw(object):
    def __init__(self,x,y,ymin=None,ymax=None):
        self.freedom = x
        self.likelihood = y
        self.ymin = ymin
        self.ymax = ymax
        
    def aic(self,ax,mylabel,ylabel,lineshape="bs-",n=2.0):
        ax.set_xlabel("DOF",fontdict={"size":"30"})
        ax.set_ylabel(ylabel,fontdict={"size":"30"})
        plt.tick_params(labelsize=20)
        plt.tick_params(which='major',length=15)
        plt.tick_params(which='minor',length=6)
        _aic = 2.0*onp.array(self.likelihood) + n*onp.array(self.freedom)
        ax.plot(onp.array(self.freedom)+f0, _aic, lineshape, label=mylabel, linewidth=1, markersize=3)
        if not self.ymin is None:
            ax.ylim((self.ymin,self.ymax))
        return ax

    def lh(self,ax,mylabel,lineshape="bs-"):
        ax.set_xlabel("DOF",fontdict={"size":"30"})
        ax.set_ylabel("Likelihood",fontdict={"size":"30"})
        plt.tick_params(labelsize=20)
        plt.tick_params(which='major',length=15)
        plt.tick_params(which='minor',length=6)
        # ax.scatter(self.freedom, self.likelihood)
        print(onp.array(self.freedom)+f0)
        ax.plot(onp.array(self.freedom)+f0, self.likelihood, lineshape, label=mylabel)
        if not self.ymin is None:
            ax.ylim((self.ymin,self.ymax))
        return ax

    def scan_tfc(self,ax,mylabel,lineshape="bs"):
        ax.set_xlabel("${\mathbf{SF}}$")
        ax.set_ylabel("Likelihood")
        ax.plot(self.freedom, self.likelihood, lineshape, label=mylabel)
        if not self.ymin is None:
            ax.ylim((self.ymin,self.ymax))
        return ax
    
    def sigma(self,ax,mylabel,ylabel,lineshape="bs-",sigmfile=None):
        if sigmfile is None:
            print("error, no sigmafile")
            return 1
        else:
            with open(sigmfile,"r") as f:
                sigma_list = json.loads(f.read())
        ax.set_xlabel("DOF",fontdict={"size":"30"})
        ax.set_ylabel(ylabel,fontdict={"size":"23"})
        plt.tick_params(labelsize=20)
        plt.tick_params(which='major',length=15)
        plt.tick_params(which='minor',length=6)
        sigmadiff = list()
        for _n, fn in enumerate(self.freedom):
            if fn > 0:
                sigmadiff.append(sigma_list[str(int(abs(fn-self.freedom[_n-1])))])
            elif fn == 0:
                sigmadiff.append(0)
            else:
                sigmadiff.append(-1*sigma_list[str(int(abs(fn-self.freedom[_n+1])))])
        _sigmadiff = list()
        def calculatep(temp,n):
            if sigmadiff[n] != 0:
                temp += sigmadiff[n]
                temp = calculatep(temp,n-1)
            return temp
        def calculaten(temp,n):
            if sigmadiff[n] != 0:
                temp += sigmadiff[n]
                temp = calculaten(temp,n+1)
            return temp
        for n, fn in enumerate(self.freedom):
            temp = 0
            if fn > 0:
                temp = calculatep(temp,n)
                _sigmadiff.append(temp)
            elif fn == 0:
                _sigmadiff.append(0)
            else:
                temp = calculaten(temp,n)
                _sigmadiff.append(temp)
        _sigma = onp.array(self.likelihood) + onp.array(_sigmadiff)
        ax.plot(onp.array(self.freedom)+f0, _sigma, lineshape, label=mylabel)
        if not self.ymin is None:
            ax.ylim((self.ymin,self.ymax))
        return ax

def runSigma(sigmfile,output,drawdist,ylabel):
    fig, axes = plt.subplots(1,1,figsize=(1200*px,900*px))
    for draw in drawdist:
        axes = draw[0].sigma(axes,draw[1]["info"],ylabel,draw[1]["lineshape"],sigmfile=sigmfile)
    axes.legend(fontsize=20)
    plt.savefig(output)

def runAIC(logn,ylabel,output,drawdist):
    fig, axes = plt.subplots(1,1,figsize=(1200*px,900*px))
    for draw in drawdist:
        axes = draw[0].aic(axes,draw[1]["info"],ylabel,draw[1]["lineshape"],n=logn)
    axes.legend(fontsize=20)
    plt.savefig(output)

def find_character_in_file(file_path, target_character):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        
    position = file_content.find(target_character)
    
    if position != -1:
        print(f"找到字符 {target_character} 在位置 {position}")
    else:
        print(f"未找到字符 {target_character}")

    print(file_content[position+18:position+30])
    return float(file_content[position+18:position+30])

def read_scan_tfc_result(filename,frac_tfc):
    target_character = "the minist value "
    R0_fcn = list()
    for _n, n in enumerate(frac_tfc):
        R0_fcn.append(find_character_in_file("./output"+filename+"/fit_result_"+str(_n)+"/fit_{0}.log".format(_n), target_character))
    draw_R0 = Draw(frac_tfc,R0_fcn)
    return draw_R0

def draw_scanSF(axes,scanSF_dir_list,frac_tfc):
    for scanSF_dir in scanSF_dir_list:
        draw = read_scan_tfc_result(scanSF_dir[0],frac_tfc)
        axes = draw.scan_tfc(axes,mylabel=scanSF_dir[1],lineshape=scanSF_dir[2])
    return axes

def prepare_random_info(dir_list,random_num):
    random_fcn = list()
    for _dir in dir_list:
        random_sof = onp.load("output/scanSF/"+_dir+".npy")

        for n in range(random_num):
            with open("./output/scanSF/"+ _dir +"/fit_result_{}/result_info.json".format(n),"r") as _file:
                data = json.load(_file)
                for key, value in data.items():
                    random_fcn.append(value["fcn"])

    return onp.array(random_sof), onp.array(random_fcn)

def format_axis(ax,xlabel,ylabel):
    # 设置轴标签的字体和大小
    ax.tick_params(axis='both', labelsize=28)
    ax.set_xlabel(xlabel, labelpad=18) 
    ax.set_ylabel(ylabel, labelpad=18)  # 设置y轴标签与轴的距离为15点

    # 设置轴的刻度线的方向
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')

    # 居中轴标题
    ax.xaxis.label.set_horizontalalignment('center')
    ax.yaxis.label.set_horizontalalignment('center')

    ax.legend(fontsize=18)

def set_style():
    # 设置全局样式参数
    plt.rcParams['axes.linewidth'] = 0.8  # 设置轴线宽度
    plt.rcParams['axes.facecolor'] = 'white'  # 设置画布背景颜色为白色
    plt.rcParams['axes.edgecolor'] = 'black'  # 设置轴线颜色为黑色
    plt.rcParams['axes.labelsize'] = 34  # 设置x, y轴标签字体大小
    plt.rcParams['axes.labelweight'] = 'normal'  # 设置x, y轴标签字体粗细
    plt.rcParams['xtick.major.size'] = 18  # 设置x轴主刻度线长度
    plt.rcParams['xtick.minor.size'] = 9  # 设置x轴次刻度线长度
    plt.rcParams['ytick.major.size'] = 18  # 设置y轴主刻度线长度
    plt.rcParams['ytick.minor.size'] = 9  # 设置y轴次刻度线长度
    plt.rcParams['legend.frameon'] = False  # 设置图例无边框
    plt.rcParams['figure.subplot.left'] = 0.17  # 设置绘图区域左边距
    plt.rcParams['figure.subplot.bottom'] = 0.17  # 设置绘图区域底边距
    plt.rcParams['figure.subplot.right'] = 0.95  # 设置绘图区域右边距
    plt.rcParams['figure.subplot.top'] = 0.95  # 设置绘图区域顶边距
    plt.rcParams['errorbar.capsize'] = 0  # 在x方向上不显示误差线

f0 = 67


if __name__ == "__main__":
    # pre setting
    # px = 1/plt.rcParams['figure.dpi']
    plt.style.use(['science','no-latex'])
    outdir = "output/pictures/scan/"
    set_style()

    # fix dof in different sf
    frac_tfc = onp.round(onp.arange(1.0,1.25,0.05), decimals=2)
    print(frac_tfc)

    fig, axes = plt.subplots(figsize=(12, 9), dpi=100)
    label = ["$R_{-1}$","$R_{0}$","$R_{1}$","$R_{2}$"]
    info = {"R0":["/scanSF/SF_Scan_Results/scan_R0","$R_{0}$","s"],
            "R1":["/scanSF/SF_Scan_Results/scan_R1","$R_{1}$","o"],
            "R2":["/scanSF/SF_Scan_Results/scan_R2","$R_{2}$","x"],
            "R-1":["/scanSF/SF_Scan_Results/scan_R-1","$R_{-1}$","s"],
            "R0_new":["/scanSF/SF_Scan_Results/scan_R0_new","$R_{0}$","s"]}

    scanSF_dir_list = [info["R0"],info["R1"],info["R2"]]
    # scanSF_dir_list = [info["R0"],info["R1"],info["R-1"],info["R2"]]
    draw_scanSF(axes,scanSF_dir_list,frac_tfc)
    format_axis(axes,"${\mathbf{SF}}$","Likelihood")
    plt.savefig(outdir + "sf.jpg")

    # random_sof
    # dir_list = ["Loose_Distribution/random-r1"]
    dir_list = ["Unconstrained_Data/random-r3"]
    # dir_list = ["Constraint_Levels_Data/random-e0"]
    # dir_list = ["random-r3-e2"]
    random_sof, random_fcn = prepare_random_info(dir_list,6)
    random_sof = random_sof[random_sof!=None]
    random_fcn = random_fcn[random_fcn!=None]

    num_bins = 100
    x = random_sof
    y = random_fcn
    hist, xedges, yedges = onp.histogram2d(x, y, bins=num_bins)
    weights = onp.zeros_like(x)
    for i in range(num_bins):
        for j in range(num_bins):
            x_mask = (x >= xedges[i]) & (x < xedges[i + 1])
            y_mask = (y >= yedges[j]) & (y < yedges[j + 1])
            weights[x_mask & y_mask] = hist[i, j]
        
    colors = weights / onp.max(weights)

    fig, axes = plt.subplots(figsize=(12, 9), dpi=100)
    print(x.shape)
    plt.scatter(x, y, c="k", cmap='viridis', alpha=0.6)

    plt.xlim(0,4)
    # plt.ylim(-8100.0,-7500)
    plt.ylim(-8100.0,-8020)
    count = onp.count_nonzero(random_fcn < -8020)
    print("< -8020",count)
    count = onp.count_nonzero(random_fcn < -8030)
    print("< -8030",count)
    count = onp.count_nonzero(random_fcn < -8034)
    print("< -8032",count)

    format_axis(axes,"${\mathbf{SF}}$","Likelihood")
    plt.savefig(outdir + "random_sof.jpg")


    # dalitz plot
    b123 = onp.sqrt(onp.load("data/real_data/b123_kk.npy"))
    b124 = onp.sqrt(onp.load("data/real_data/b124_kk.npy"))
    fig, axes = plt.subplots(figsize=(12, 9), dpi=100)
    plt.scatter(b123,b124,label="event",s=1.8,marker="o")
    # plt.xlim(1.0,2.5)
    format_axis(axes,"$\phi K^{+}$ (GeV)","$\phi K^{-}$ (GeV)")
    plt.savefig(outdir + "dalitz.jpg")

    # scan select draw
    # sarr = onp.loadtxt("./scan_select.txt")
    # print(sarr.shape)
    # sarr = sarr.reshape(80,-1,3)
    # sorted_arr = onp.array([a[a[:,0].argsort()] for a in sarr])
    # print(sorted_arr[:,0,1])
    # print(sorted_arr.shape)

    fig, axes = plt.subplots(figsize=(12, 9), dpi=100)
    # 高斯函数.
    def gauss(x, a, b, c):
        return a * onp.exp(-(x - b)**2 / (2 * c**2))
    # 创建直方图
    sf_data = onp.load("output/scanSF/scan-select-5.npy")
    print(sf_data.shape)
    hist, bin_edges = onp.histogram(sf_data, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    params, _ = curve_fit(gauss, bin_centers, hist, maxfev=5000)
    print(params)
    x_fit = onp.linspace(bin_edges[0], bin_edges[-1], 100)
    y_fit = gauss(x_fit, *params)

    bin_width = 0.005
    bins = onp.arange(1.04, 1.24+bin_width, bin_width)

    plt.hist(sf_data, bins=bins, density=True, label='Histogram', alpha=0.6, edgecolor='black')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')

    axes.tick_params(axis='both', which='major', labelsize=22)
    # plt.title('The Minimal Point of the SF')
    plt.xlabel('SF')
    plt.ylabel('Events/0.005')
    plt.savefig('output/pictures/scan/sf_mini.jpg')

    # fig, axes = plt.subplots(figsize=(12, 9), dpi=100)
    # plt.scatter(sorted_arr[:,0,1],sorted_arr[:,0,2],label="event",s=1.8,marker="o")
    # format_axis(axes,"SF","events")
    # plt.savefig(outdir + "min_events.jpg")
    

    # # fix sf in different dof
    # info_10 = {"0":-9400.429746984839,"6":-9409.842294111018,"18":-9422.595406016553,"30":-9433.281456941098,"-12":-9374.57863037425,"-18":-9085.100736428925,"-30":-8907.064317225464}
    # draw_10 = Draw(info_10)

    # info_12 = {"0":-9407.136788522039,"6":-9415.520810724338,"18":-9429.567718526509,"30":-9440.092198538608,"-12":-9380.41209849011,"-18":-9123.618998266858,"-30":-8985.147050085172}
    # draw_12 = Draw(info_12)

    # info_14 = {"0":-9404.174456152192,"6":-9409.85057701671,"18":-9423.71730970328,"30":-9435.657605258326,"-12":-9373.427112603298,"-18":-9122.358203854732,"-30":-8968.935749227967}
    # draw_14 = Draw(info_14)

    # drawdist = [
    #     [draw_10,{"info":"${\mathbf{SF}} \quad 100\%$","lineshape":"bs-"}],
    #     [draw_12,{"info":"${\mathbf{SF}} \quad 120\%$","lineshape":"go-"}],
    #     [draw_14,{"info":"${\mathbf{SF}} \quad 140\%$","lineshape":"r^-"}],
    # ]

    # ## likelihood
    # fig, axes = plt.subplots(1,1,figsize=(1200*px,900*px))
    # for draw in drawdist:
    #     axes = draw[0].lh(axes,draw[1]["info"],draw[1]["lineshape"])
    # axes.legend(fontsize=20)
    # plt.savefig(outdir + "lh.jpg")

    # # aic
    # output = outdir + "aic.jpg"
    # runAIC(2,"AIC",output,drawdist)

    # # bic
    # logn = onp.log(13777.0)
    # output = outdir + "/bic.jpg"
    # runAIC(logn,"BIC",output,drawdist)

    # # sigma-5
    # sigmfile = "picture/sigma/5-sigma.json"
    # output = "output/pictures/scan/sigma5.jpg"
    # runSigma(sigmfile,output,drawdist,"Likelihood with $5\sigma$ compensation")

    # # sigma-4
    # sigmfile = "picture/sigma/4-sigma.json"
    # output = "output/pictures/scan/sigma4.jpg"
    # runSigma(sigmfile,output,drawdist,"Likelihood with $4\sigma$ compensation")

    # # sigma-3
    # sigmfile = "picture/sigma/3-sigma.json"
    # output = "output/pictures/scan/sigma3.jpg"
    # runSigma(sigmfile,output,drawdist,"Likelihood with $3\sigma$ compensation")
