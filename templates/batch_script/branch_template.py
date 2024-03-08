{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa_draw_wt.py" import PWAFunc, fit_run with context -%}

logger = logging.getLogger("draw")

{{args()}}

{% call ProcessInitializer(data_path,mc_path) %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def run_weight(self, args_list, float_list):
        all_const_index = {{all_const_index}}
        self.args_list = args_list
        self.float_list = float_list
        print("run branch")
        self.jit_request()
        args_float = np.array(self.args_list[self.float_list])
        wt_list = self.jit_weight_{{lh_coll[0].tag}}(args_float)
        sum_wt = onp.sum(wt_list[0])
        return sum_wt
{% endcall %}

{% call Control(jinja_draw_info.draw_wt.ResultFile) %}
        pwaf = self.pwaf_list[0][0]
        sum_wt = pwaf.run_weight(self.args_list, self.float_list)
        min_fcn = sum_wt
{% endcall %}