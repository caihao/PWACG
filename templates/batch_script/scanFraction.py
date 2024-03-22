{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa_draw_wt.py" import PWAFunc, fit_run with context -%}

logger = logging.getLogger("draw")

{{args()}}

{% call ProcessInitializer(data_path="data/mc_truth",mc_path="data/mc_truth") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def run_weight(self, args_list, float_list):
        all_const_index = {{all_const_index}}
        self.args_list = args_list
        self.float_list = float_list
        print("run weight")
        self.jit_request()
        args_float = np.array(self.args_list[self.float_list])
        wt_list = self.jit_weight_{{lh_coll[0].tag}}(args_float)
        self.sum_wt = onp.sum(wt_list[0])
        print("run is over, next!")
        wt_list = wt_list[1:]
        frac_list = [onp.einsum("lj->l",wt)/self.sum_wt for wt in wt_list]
        ferror = [0]
        values = frac_list
        
        return values, ferror
{% endcall %}

{% call Control(jinja_draw_info.draw_wt.ResultFile) %}
        lasso_frac_dict = {{lasso_frac_dict}}
        # print(lasso_frac_dict)

        pwaf = self.pwaf_list[0][0]
        fvalue, ferror = pwaf.run_weight(self.args_list, self.float_list)
        wt_list = list()
        for mod in fvalue:
            for frac in mod:
                wt_list.append(frac)

        # print(wt_list)
        for n, key in enumerate(list(lasso_frac_dict.keys())):
            lasso_frac_dict[key] = [float(wt_list[n]),lasso_frac_dict[key]]
        # print(lasso_frac_dict)

        min_fcn = lasso_frac_dict
{% endcall %}