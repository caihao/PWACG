{% macro data_likelihood() %}
    {% for lh in lh_coll %}
    def data_likelihood_{{lh.tag}}(self, args):
        {% for arg, value in lh["slit_args_dict"].items() %}
        {{arg}} = {{value}}
        {% endfor %}
        {% for func in lh["func_differ"] %}
        data_{{func.calculate_func}} = self.calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.data_{{sbc}}, 
                {%- endfor -%}
                self.data_{{func.amp}})
        {% endfor %}
        {% for func in lh["func_differ"] %}
        lasso_data_{{func.calculate_func}} = self.lasso_calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.truth_{{sbc}}, 
                {%- endfor -%}
                self.truth_{{func.amp}})
        {% endfor %}
        {{lh.bounding}}
        # jax_exper.id_print(step_function)
        {% if info.fit.use_weight %}
        {{lh.wt_data_return_dict}}
        {% else %}
        {{lh.data_return_dict}}
        {% endif %}
    {% endfor %}
{% endmacro %}

{% macro mc_likelihood() %}
    {% for lh in lh_coll %}
    def mc_likelihood_{{lh.tag}}(self, args):
        {% for arg, value in lh["slit_args_dict"].items() %}
        {{arg}} = {{value}}
        {% endfor %}
        {% for func in lh["func_differ"] %}
        mc_{{func.calculate_func}} = self.calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.mc_{{sbc}}, 
                {%- endfor -%}
                self.mc_{{func.amp}})
        {% endfor %}
        {{lh.mc_return_dict}}
    {% endfor %}
{% endmacro %}

{% macro weight_wt() %}
    {% for lh in lh_coll %}
    def weight_{{lh.tag}}(self, args):
        {% for arg, value in lh["slit_args_dict"].items() %}
        {{arg}} = {{value}}
        {% endfor %}
        {% for func in lh["func_differ"] %}
        lasso_data_{{func.calculate_func}} = self.lasso_calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.data_{{sbc}}, 
                {%- endfor -%}
                self.data_{{func.amp}})
        {% endfor %}
        {{lh.calc_wt[0]}}
        {{lh.calc_wt[1]}}
        return wt_list
    {% endfor %}
{% endmacro %}

{% macro weight_wt_truth() %}
    {% for lh in lh_coll %}
    def weight_truth_{{lh.tag}}(self, args):
        {% for arg, value in lh["slit_args_dict"].items() %}
        {{arg}} = {{value}}
        {% endfor %}
        {% for func in lh["func_differ"] %}
        lasso_data_{{func.calculate_func}} = self.lasso_calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.truth_{{sbc}}, 
                {%- endfor -%}
                self.truth_{{func.amp}})
        {% endfor %}
        {{lh.calc_wt[0]}}
        {{lh.calc_wt[1]}}
        return wt_list
    {% endfor %}
{% endmacro %}

{% macro weight_lh() %}
    {% for lh in lh_coll %}
    def weight_{{lh.tag}}(self, args):
        {% for arg, value in lh["slit_args_dict"].items() %}
        {{arg}} = {{value}}
        {% endfor %}
        {% for func in lh["func_differ"] %}
        data_{{func.calculate_func}} = self.calculate_{{func.prop_name}}(
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                self.mc_{{sbc}}, 
                {%- endfor -%}
                self.mc_{{func.amp}})
        {% endfor %}
        {{lh.weight_return_dict}}
    {% endfor %}
{% endmacro %}

{%- macro calculate_core(func) %}
    def calculate_{{func.prop_name}}(self,
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                {{sbc}}, 
                {%- endfor -%}
                {{func.amp}}):

        # ph = np.moveaxis(self.phase({{func.theta}}), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.{{func.prop_name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        # print("bw", bw.shape)
        # phif = dplex.dtomine({{func.amp}})
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct({{func.const}}, {{func.theta}})
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp({{func.const}}), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", {{func.amp}}, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        return phif 

    def {{func.prop_name}}(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        {%- if info.merge == 'phi' %}
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        b = np.moveaxis(vmap(partial(self.{{func.prop.prop_f.name}},Sbc={{func.Sbc.f}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
        {%- endif %}

        {%- if info.merge == 'f' %}
        a = np.moveaxis(vmap(partial(self.{{func.prop.prop_phi.name}},Sbc={{func.Sbc.phi}}))({{func["prop"]["prop_phi"]["_paras"]|join(',')}}),1,0)
        b = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_f"]["paras"]|join(',')}})
        return dplex.deinsum("ij, j->ij",a,b)
        {%- endif %}

        {%- if info.merge == "None" %}
        a = np.moveaxis(vmap(partial(self.{{func.prop.prop_phi.name}},Sbc={{func.Sbc.phi}}))({{func["prop"]["prop_phi"]["_paras"]|join(',')}}),1,0)
        b = np.moveaxis(vmap(partial(self.{{func.prop.prop_f.name}},Sbc={{func.Sbc.f}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("ij, ij->ij",a,b)
        {%- endif %}
{%- endmacro %}

{%- macro lasso_calculate_core(func) %}
    def lasso_calculate_{{func.prop_name}}(self,
                {%- for para in func["all_paras"] -%}
                {{para}}, 
                {%- endfor %}
                {%- for key, sbc in func["Sbc"].items() -%}
                {{sbc}}, 
                {%- endfor -%}
                {{func.amp}}):

        # ph = np.moveaxis(self.phase({{func.theta}}), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.{{func.prop_name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        # print("bw", bw.shape)
        # phif = dplex.dtomine({{func.amp}})
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct({{func.const}}, {{func.theta}})
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp({{func.const}}), ph)
        # print("const_ph",const_ph.shape)
        phif = dplex.deinsum_ord("ijk,li->ljk", {{func.amp}}, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        return phif
{%- endmacro %}