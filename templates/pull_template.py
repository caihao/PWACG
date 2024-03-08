{% from "templates/head.py" import head %}
{{head()}}
{% from "dmultiprocess/control.py" import Control with context -%}
{% from "dmultiprocess/multiprocess_interface.py" import args, ProcessReturns, ProcessInitializer with context -%}

{% from "pwa/pwa.py" import PWAFunc with context -%}

{{args()}}

{% call ProcessInitializer(data_path="data/real_data",mc_path="data/mc_int") %}
    def process_initializer_generator(self):
        self.data_npz(self.data_slices)
        self.mc_npz(self.mc_slices)
        self.regular()
        print("============w i t h  n e x t===============")

        data_numbering = 0
        event_num = self.mini_run
        while data_numbering<event_num:
            _ = ProcessInitializers()
            {% for sbc in sbc_collection %}
            _.data_{{sbc}} = self.all_data_{{sbc}}[data_numbering]
            _.mc_{{sbc}} = self.all_mc_{{sbc}}[data_numbering%self.mc_slices]
            {% endfor %}
            {% for tensor in amp_collection %}
            _.data_{{tensor}} = self.all_data_{{tensor}}[data_numbering]
            _.mc_{{tensor}} = self.all_mc_{{tensor}}[data_numbering%self.mc_slices]
            {% endfor %}
            _.data_numbering = data_numbering
            yield _
            data_numbering += 1
        return
{% endcall %}

{{ProcessReturns()}}

{% call PWAFunc() %}
    def fit_run(self):
        print("run begin")
        self.jit_request()

        print("defult likelihood",self.likelihood(self.args_list))
        args_float = self.args_list[self.float_list]

        values, errors = Doptimization.optimize(self.jit_likelihood_float, args_float, self.jit_grad_likelihood_float)
        # _errors = np.sqrt(np.diag(np.linalg.inv(self.hess(values))))

        all_values = copy.deepcopy(self.args_list)
        for i, locate in enumerate(self.float_list):
            all_values[locate] = values[i]

        print("result likelihood",self.likelihood(all_values))

        all_errors = copy.deepcopy(self.args_list)
        for i, locate in enumerate(self.float_list):
            all_errors[locate] = errors[i]

        return all_values, all_errors
{% endcall %}

{% call Control(jinja_fit_info.pull.ResultFile) %}
        fvalue, ferror = pwaf.fit_run()
{% endcall %}