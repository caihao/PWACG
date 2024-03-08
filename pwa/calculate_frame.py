{%- from "pwa/calculate_annex.py" import calculate_core, lasso_calculate_core with context %}
{% import "pwa/propogator.py" as prop %}
{%- macro PWAFunc() %}
{% block log %}
{% endblock %}
class PWAFunc():
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
        return jvp(grad(self.data_likelihood_{{lh.tag}}), [args_float], [any_vector])

    def hvp_mc_fwdrev_{{lh.tag}}(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_{{lh.tag}}), [args_float], [any_vector])
    {% endfor %}

    def jit_request(self):
        {%- block jit_request %}
        {%- endblock %}
        {% for lh in lh_coll %}
        self.jit_data_likelihood_{{lh.tag}} = jit(self.data_likelihood_{{lh.tag}}, device=self.device)
        self.jit_mc_likelihood_{{lh.tag}} = jit(self.mc_likelihood_{{lh.tag}}, device=self.device)
        self.jit_grad_data_likelihood_{{lh.tag}} = jit(grad(self.data_likelihood_{{lh.tag}}), device=self.device)
        self.jit_grad_mc_likelihood_{{lh.tag}} = jit(value_and_grad(self.mc_likelihood_{{lh.tag}}), device=self.device)
        self.jit_hvp_data_fwdrev_{{lh.tag}} = jit(self.hvp_data_fwdrev_{{lh.tag}}, device=self.device)
        self.jit_hvp_mc_fwdrev_{{lh.tag}} = jit(self.hvp_mc_fwdrev_{{lh.tag}}, device=self.device)
        {% endfor %}

    {{caller()}}
{% endmacro %}