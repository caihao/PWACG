{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer, npz with context -%}
{% from "pwa/pwa_select.py" import PWAFunc with context -%}
import ROOT
from ROOT import TH1D, TLorentzVector, TCanvas, TGraph, TMultiGraph, gStyle, gPad, TPaveText, TLegend, TObject, TLatex, TH2D
from picture import my_plot as mpl
from Tensor import CacheTensor as calc

########################################################################
#  这里的挑选程序是用来产生 toy mc 的                                      #
#  通过先舍选 truth 然后用舍选后的 truth 编号得到对应的经过 cut 的 phsp       #
########################################################################

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_fit.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fit")

{{args()}}

{% call ProcessInitializer(data_path="data/phase_mc",mc_path="data/mc_truth") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def generate_random(self):
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        print("seed=",num_seed)
        # onp.random.seed(42)
        # print(num_seed)
        onp.random.seed(num_seed)
        return 0

    def run_select(self,args_float):
        print("run begin")
        # self.jit_request()
        {% for lh in lh_coll %}
        self.jit_weight_{{lh.tag}} = jit(self.weight_{{lh.tag}})
        {% endfor %}
        {% for lh in lh_coll %}
        wt_list = self.jit_weight_{{lh.tag}}(args_float)
        {% endfor %}
        self.wt = wt_list[0]
        w_max = self.wt.max()
        w_min = self.wt.min()
        print("max=", w_max)
        print("min=", w_min)
        print("all data number=",self.wt.shape)
        self.generate_random()
        rand = onp.random.rand(self.wt.shape[0]) * (w_max - w_min) + w_min
        index = onp.squeeze(onp.where(self.wt > rand), axis=None)
        print("pass filter data number=",index.shape)
        return index
{% endcall %}

{% call Control(jinja_draw_info.select.ResultFile) %}
        self.args_float = self.args_list[self.float_list]
        pwaf = PWAFunc(process_initializer[0][0],device_list[0])
        index = pwaf.run_select(self.args_float)
        # 在这里准备select的文件
        {% for lh in lh_coll %}
        cut_index = index
        {% endfor %}
        {% for sbc in sbc_collection %}
        mc_{{sbc}} = onp.load("data/phase_mc/{{sbc}}.npy")
        select_{{sbc}} = mc_{{sbc}}[cut_index]
        onp.save("data/select/{{sbc}}",select_{{sbc}})
        {% endfor %}
        print(cut_index.shape)
        {% for tensor in amp_collection %}
        mc_{{tensor}} = onp.load("data/phase_mc/{{tensor}}.npy")
        select_{{tensor}} = mc_{{tensor}}[:,cut_index,:]
        onp.save("data/select/{{tensor}}",select_{{tensor}})
        {% endfor %}

        {% for lh in lh_coll %}
        os.system("cp data/select/*{{lh.tag}}*.npy data/real_data/")
        os.system("cp data/select/*{{lh.tag}}*.npy data/draw_mc/real_data/")
        calc.MergeMomentum("data/draw_mc/real_data/b123_{{lh.tag}}.npy","data/draw_mc/real_data/b124_{{lh.tag}}.npy")
        {% endfor %}
        fvalue = [0]
        ferror = [0]
        min_fcn = [cut_index.shape[0]]
{% endcall %}