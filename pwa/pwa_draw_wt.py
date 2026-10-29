{% import "pwa/calculate_annex.py" as annex with context%}
{% extends "pwa/calculate_frame.py" %}

{% block likelihood %}
        {{annex.data_likelihood()}}
        {{annex.mc_likelihood()}}
{% endblock %}

{% block weight %}
        {{annex.weight_wt()}}
        {{annex.weight_wt_truth()}}
{% endblock %}

{% block jit_request %}
        if getattr(PWAFunc, "_compiled_weight_kernels", None) is None:
            PWAFunc._compiled_weight_kernels = {}
            {% for lh in lh_coll %}
            def _raw_weight_{{lh.tag}}(args, data_pack):
                ctx = object.__new__(PWAFunc)
                (
                    {% for data in data_collection %}
                    ctx.data_{{data}},
                    ctx.truth_{{data}},
                    {% endfor %}
                    ctx.wt_data_{{lh.tag}},
                ) = data_pack
                return ctx.weight_{{lh.tag}}(args)

            def _raw_weight_truth_{{lh.tag}}(args, data_pack):
                ctx = object.__new__(PWAFunc)
                (
                    {% for data in data_collection %}
                    ctx.data_{{data}},
                    ctx.truth_{{data}},
                    {% endfor %}
                    ctx.wt_data_{{lh.tag}},
                ) = data_pack
                return ctx.weight_truth_{{lh.tag}}(args)

            PWAFunc._compiled_weight_kernels["{{lh.tag}}"] = (
                jit(_raw_weight_{{lh.tag}}),
                jit(_raw_weight_truth_{{lh.tag}}),
            )
            {% endfor %}

        {% for lh in lh_coll %}
        _wt_{{lh.tag}}, _wt_truth_{{lh.tag}} = PWAFunc._compiled_weight_kernels["{{lh.tag}}"]
        self.jit_weight_{{lh.tag}} = lambda args_float, _f=_wt_{{lh.tag}}, _p=self._data_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        self.jit_weight_truth_{{lh.tag}} = lambda args_float, _f=_wt_truth_{{lh.tag}}, _p=self._data_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        {% endfor %}
{% endblock %}