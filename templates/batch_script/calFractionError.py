{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa_draw_wt.py" import PWAFunc, fit_run with context -%}

logger = logging.getLogger("draw")

{{args()}}

{% call ProcessInitializer(data_path="data/draw_data",mc_path="data/mc_int") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def run_weight(self, args_list, float_list):
        check_args = onp.load("output/draw/fraction_paras.npy")
        self.args_list = args_list
        self.float_list = float_list
        print("calculate fraction")
        self.jit_request()
        # args_float = onp.array(self.args_list[self.float_list])
        fraction_sample = list()
        for paras in check_args:
            # args_float = paras
            print("fraction calculate")
            fraction = list()
            {% for lh in lh_coll %}
            wt_list = self.jit_weight_{{lh.tag}}(paras)
            self.sum_wt = onp.sum(wt_list[0])
            # print("args_list: \n",args_float)
            {% for func in lh.func_differ %}
            num_mod = int(len({{func.const_index}})/{{func.damp}})
            for i in range(num_mod):
                wt = wt_list[{{loop.index}}][i]
                fraction.append(onp.sum(wt)/self.sum_wt)
                print("{{func.calculate_func}} frac", onp.sum(wt)/self.sum_wt)
            {% endfor %}
            {% endfor %}
            if np.isnan(self.sum_wt):
                print("nan")
            else:
                fraction_sample.append(fraction)
                print(onp.sum(onp.array(fraction)))
        values = [0]
        ferror = [0]
        onp.save("output/draw/fraction_sample",fraction_sample)
        return values, ferror
{% endcall %}

{% call Control(jinja_draw_info.draw_wt.ResultFile) %}
        mass_index = onp.array({{args_index_collection.mass}})
        width_index = onp.array({{args_index_collection.width}})
        flatte_index = onp.array({{args_index_collection.flatte}})
        # my_args = onp.append(onp.append(mass_index,width_index),flatte_index)
        # my_args = onp.array([0,1,2,7,9])
        pwaf = self.pwaf_list[0][0]
        fvalue, ferror = pwaf.run_weight(self.args_list, self.float_list)
        min_fcn = [0]
{% endcall %}