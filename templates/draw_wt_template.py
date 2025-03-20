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
    def mod_weight(self, total_weight, mod_name, args_result, const_index, num_iamps, all_const_index):
        for n, i in enumerate(range(0, len(const_index), num_iamps)):
            _args_result = copy.deepcopy(args_result)
            _const_index = const_index[i:i+num_iamps]
            leftover = [x for x in all_const_index if x not in _const_index]
            for i in leftover:
                _args_result[i] = -100.0 
            args_float = np.array(_args_result[self.float_list])
            wt = self.jit_weight_{{lh_coll[0].tag}}(args_float)
            # print(mod_name, wt)
            frac = onp.sum(wt)/self.sum_wt
            print(mod_name, frac)
            total_weight[mod_name+"_"+str(n)] = wt
        return

    def run_weight(self, args_list, float_list,mode="pass"):
        all_const_index = {{all_const_index}}
        self.args_list = args_list
        self.float_list = float_list
        print("run weight")
        self.jit_request()
        args_float = np.array(self.args_list[self.float_list])
        if mode == "pass":
            wt_list = self.jit_weight_{{lh_coll[0].tag}}(args_float)
        if mode == "truth":
            wt_list = self.jit_weight_truth_{{lh_coll[0].tag}}(args_float)
        self.sum_wt = onp.sum(wt_list[0])
        total_weight = dict()
        # print("args_list: \n",args_float)
        total_weight["all_mods_wt"] = wt_list[0]
        total_fit_frac = 0
        loop_index = 0
        temp_cal_cul_func = list()
        {% for func in func_info %}
        num_mod = int(len({{func.const_index}})/{{func.damp}})
        for i in range(num_mod):
            if not "{{func.calculate_func|replace("_l_","_")|replace("_r_","_")}}" in temp_cal_cul_func:
                loop_index += 1
            wt = wt_list[loop_index][i]
            total_weight["{{func.calculate_func}}_"+str(i)] = wt
            print("{{func.calculate_func}} frac", onp.sum(wt)/self.sum_wt)
            total_fit_frac += onp.sum(wt)/self.sum_wt
            temp_cal_cul_func.append("{{func.calculate_func|replace("_l_","_")|replace("_r_","_")}}")
        {% endfor %}
        values = np.array(self.args_list)
        total_weight["fit_value"] = values
        ferror = ["{{error_list|join('\",\"')}}"]
        total_weight["fit_error"] = ferror
        if mode == "pass":
            onp.savez("{{draw_result_file}}", **total_weight)
        if mode == "truth":
            onp.savez("{{draw_result_file}}".replace(".","_truth."), **total_weight)
        print("total fit fraction:",total_fit_frac)
        print("run over , save data to weight.npz")
        return values, ferror, total_fit_frac
{% endcall %}

{% call Control(jinja_draw_info.draw_wt.ResultFile) %}
        pwaf = self.pwaf_list[0][0]
        fvalue, ferror, sof = pwaf.run_weight(self.args_list, self.float_list,mode="pass")
        self.save_in_json(fvalue,ferror,"output/error",str(0))
        fvalue, ferror, sof = pwaf.run_weight(self.args_list, self.float_list,mode="truth")
        sof = onp.array(sof)
        fvalue = onp.array(fvalue)
        min_fcn = [sof]
{% endcall %}