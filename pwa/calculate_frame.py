{%- from "pwa/calculate_annex.py" import calculate_core, lasso_calculate_core with context %}
{% import "pwa/propogator.py" as prop %}
{%- macro PWAFunc() %}
{% block log %}
{% endblock %}
class PWAFunc():
    _compiled_kernels = None

    def __init__(self, cdl=None, device_id=None):
        self.device = device_id
        {% for data in data_collection %}
        self.data_{{data}} = device_put(np.array(cdl.data_{{data}}),device=self.device)
        self.mc_{{data}} = device_put(np.array(cdl.mc_{{data}}),device=self.device)
        self.truth_{{data}} = device_put(np.array(cdl.truth_{{data}}),device=self.device)
        {% endfor %}
        {% for lh in lh_coll %}
        self.wt_data_{{lh.tag}} = device_put(np.array(cdl.wt_data_{{lh.tag}}),device=self.device)
        {% endfor %}

        {% for lh in lh_coll %}
        self._data_pack_{{lh.tag}} = (
            {% for data in data_collection %}
            self.data_{{data}},
            self.truth_{{data}},
            {% endfor %}
            self.wt_data_{{lh.tag}},
        )
        self._mc_pack_{{lh.tag}} = (
            {% for data in data_collection %}
            self.mc_{{data}},
            self.truth_{{data}},
            {% endfor %}
        )
        {% endfor %}
    
    {% block likelihood %}
    {% endblock %}

    {% block weight %}
    {% endblock %}

    {%- for func in prop_coll %}
    {{calculate_core(func)}}
    {%- endfor %}

    {%- for func in prop_coll %}
    {{lasso_calculate_core(func)}}
    {%- endfor %}

    def phase(self, theta):
        return vmap(self._phase)(theta)

    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))
    
    {{prop.Prop()}}

    {% for lh in lh_coll %}
    def hvp_data_fwdrev_{{lh.tag}}(self, args_float, any_vector):
        return self.jit_hvp_data_fwdrev_{{lh.tag}}(args_float, any_vector)

    def hvp_mc_fwdrev_{{lh.tag}}(self, args_float, any_vector):
        return self.jit_hvp_mc_fwdrev_{{lh.tag}}(args_float, any_vector)
    {% endfor %}

    def jit_request(self):
        {%- block jit_request %}
        {%- endblock %}
        if PWAFunc._compiled_kernels is None:
            PWAFunc._compiled_kernels = {}
            {% for lh in lh_coll %}
            def _raw_data_lh_{{lh.tag}}(args, data_pack):
                ctx = object.__new__(PWAFunc)
                (
                    {% for data in data_collection %}
                    ctx.data_{{data}},
                    ctx.truth_{{data}},
                    {% endfor %}
                    ctx.wt_data_{{lh.tag}},
                ) = data_pack
                return ctx.data_likelihood_{{lh.tag}}(args)

            def _raw_mc_lh_{{lh.tag}}(args, mc_pack):
                ctx = object.__new__(PWAFunc)
                (
                    {% for data in data_collection %}
                    ctx.mc_{{data}},
                    ctx.truth_{{data}},
                    {% endfor %}
                ) = mc_pack
                return ctx.mc_likelihood_{{lh.tag}}(args)

            # 1. 编译原似然函数（不硬编码 device，由输入张量所在设备自动决定）
            _jit_data_lh_{{lh.tag}} = jit(_raw_data_lh_{{lh.tag}})
            _jit_mc_lh_{{lh.tag}} = jit(_raw_mc_lh_{{lh.tag}})

            # 2. 基于已编译的原函数构建并编译一阶导数
            _jit_grad_data_{{lh.tag}} = jit(grad(_jit_data_lh_{{lh.tag}}, argnums=0))
            _jit_grad_mc_{{lh.tag}} = jit(value_and_grad(_jit_mc_lh_{{lh.tag}}, argnums=0))

            # 3. 直接复用已编译的一阶导数构建并编译二阶导数 (HVP)
            @jit
            def _jit_hvp_data_{{lh.tag}}(args_float, any_vector, data_pack):
                return jvp(
                    lambda x: _jit_grad_data_{{lh.tag}}(x, data_pack),
                    [args_float],
                    [any_vector]
                )
            # return mc_grad, mc_hvp 返回三元组：
            @jit
            def _jit_hvp_mc_{{lh.tag}}(args_float, any_vector, mc_pack):
                (mc_val, mc_grad), (_, mc_hvp) = jvp(
                    lambda x: _jit_grad_mc_{{lh.tag}}(x, mc_pack),
                    [args_float],
                    [any_vector]
                )
                return mc_val, mc_grad, mc_hvp

            PWAFunc._compiled_kernels["{{lh.tag}}"] = (
                _jit_data_lh_{{lh.tag}},
                _jit_mc_lh_{{lh.tag}},
                _jit_grad_data_{{lh.tag}},
                _jit_grad_mc_{{lh.tag}},
                _jit_hvp_data_{{lh.tag}},
                _jit_hvp_mc_{{lh.tag}},
            )
            {% endfor %}

        {% for lh in lh_coll %}
        (
            _data_lh_{{lh.tag}},
            _mc_lh_{{lh.tag}},
            _grad_data_{{lh.tag}},
            _grad_mc_{{lh.tag}},
            _hvp_data_{{lh.tag}},
            _hvp_mc_{{lh.tag}},
        ) = PWAFunc._compiled_kernels["{{lh.tag}}"]

        self.jit_data_likelihood_{{lh.tag}} = lambda args_float, _f=_data_lh_{{lh.tag}}, _p=self._data_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        self.jit_mc_likelihood_{{lh.tag}} = lambda args_float, _f=_mc_lh_{{lh.tag}}, _p=self._mc_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        self.jit_grad_data_likelihood_{{lh.tag}} = lambda args_float, _f=_grad_data_{{lh.tag}}, _p=self._data_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        self.jit_grad_mc_likelihood_{{lh.tag}} = lambda args_float, _f=_grad_mc_{{lh.tag}}, _p=self._mc_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), _p)
        self.jit_hvp_data_fwdrev_{{lh.tag}} = lambda args_float, any_vector, _f=_hvp_data_{{lh.tag}}, _p=self._data_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), device_put(any_vector, device=_d), _p)
        self.jit_hvp_mc_fwdrev_{{lh.tag}} = lambda args_float, any_vector, _f=_hvp_mc_{{lh.tag}}, _p=self._mc_pack_{{lh.tag}}, _d=self.device: _f(device_put(args_float, device=_d), device_put(any_vector, device=_d), _p)
        {% endfor %}

    {{caller()}}
{% endmacro %}