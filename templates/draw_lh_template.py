{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa_draw_lh.py" import PWAFunc, fit_run with context -%}

####################################################################
#  this script is to calculate the BIC of all mods                 #
####################################################################

logger = logging.getLogger("draw")

{{args()}}

{% call ProcessInitializer(data_path="data/real_data",mc_path="data/mc_int") %}
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def mod_weight(self, mod_name, args_list, float_list, const_index, num_iamps, all_const_index):
        for n, i in enumerate(range(0, len(const_index), num_iamps)):
            _args_list = copy.deepcopy(args_list)
            _const_index = const_index[i:i+num_iamps]
            leftover = [x for x in all_const_index if x not in _const_index]
            for i in leftover:
                _args_list[i] = -100.0 
            args_float = onp.array(_args_list[float_list])

            wt = self.jit_weight_{{lh_coll[0].tag}}(args_float)
            frac = onp.sum(wt)/self.sum_wt
            # print(mod_name, frac)
            self.frac_list.append(frac)
        return

    def run_weight(self, args_list, float_list, bic_index):
        self.jit_request()
        all_const_index = {{all_const_index}}
        args_float = onp.array(args_list[float_list])
        args_float[bic_index] = -100.0
        total_wt = self.jit_weight_{{lh_coll[0].tag}}(args_float)
        self.sum_wt = onp.sum(total_wt)

        self.frac_list = list()
        {% for func in func_info %}
        self.mod_weight("{{func.calculate_func}}", args_list, float_list, {{func.const_index}}, {{func.damp}}, all_const_index)
        {% endfor %}
        result = self.frac_list
        return result
{% endcall %}

{% call Control(jinja_fit_info.lasso.ResultFile) %}
        # pwaf = self.pwaf_list[0][0]
        # all_const = list()
        # {% for func in func_info %}
        # temp_list = {{func.const_index}}
        # for j in range(0, len(temp_list), {{func.damp}}):
        #     all_const.append(temp_list[j:j+{{func.damp}}])
        # {% endfor %}
        # lasso_frac_dict = {{lasso_frac_dict}}
        # for n, name in enumerate(lasso_frac_dict.keys()):
        #     lasso_frac_dict[name].append(all_const[n])
        # bic_index = list()
        # for name in  self.bic_delete_mods:
        #     if name in lasso_frac_dict:
        #         for index in lasso_frac_dict[name][1]:
        #             bic_index.append(index)
        # frac_list = pwaf.run_weight(self.args_list, self.float_list, bic_index)
        # args_float = self.args_list[self.float_list]
        # args_float[bic_index] = -100.0
        # fcn = self.thread_likelihood(args_float)
        # for n, name in enumerate(lasso_frac_dict.keys()):
        #     lasso_frac_dict[name].append(frac_list[n])

        # fvalue = lasso_frac_dict
        # ferror = {"data_size":{{lh_coll[0].data_size}}, "fcn":fcn}

        args_float = self.args_list[self.float_list]
        fvalue = [0]
        ferror = [0]
        likelihood = self.thread_likelihood(args_float)
        print("part of likelihood : {}".format(likelihood))
        min_fcn = likelihood
{% endcall %}