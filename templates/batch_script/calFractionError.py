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
        fraction_paras = onp.load("output/draw/fraction_paras.npz")
        check_args_list = fraction_paras["args_sample"]
        source_args = fraction_paras["source_args"]
        self.args_list = args_list
        self.float_list = float_list
        print("calculate fraction")
        self.jit_request()

        fraction_pull = list()
        for check_args in check_args_list:
            fraction_sample = list()
            check_args = onp.concatenate((source_args[onp.newaxis, :], check_args), axis=0)
            for paras in check_args:
                print("fraction calculate")
                fraction = list()
                {% for lh in lh_coll %}
                wt_list = self.jit_weight_{{lh.tag}}(paras)
                self.sum_wt = onp.sum(wt_list[0])
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

            fraction_sample = onp.array(fraction_sample)
            fraction_error = list()
            source_frac = onp.array(fraction_sample[0,:])
            og_frac = onp.array(fraction_sample[1,:])
            fraction_sample = fraction_sample[2:,:]
            for i in range((fraction_sample.shape)[1]):
                sample = fraction_sample[:,i]
                print(sample.shape)
                sample = sample[sample>(og_frac[i]*0.05)]
                sigma = onp.sqrt(onp.var(sample))
                print(sample.shape)
                # sigma = onp.sqrt(onp.var(fraction_sample[:,i]))
                fraction_error.append(sigma)
                print(i)
                print(sigma)
                print("***"*18)
            fraction_error = onp.array(fraction_error) + 1e-20
            fraction_pull.append((source_frac-og_frac)/(fraction_error))

        onp.save("output/draw/fraction_pull",onp.array(fraction_pull))

        values = [0]
        ferror = [0]
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