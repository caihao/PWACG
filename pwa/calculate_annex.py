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

        {% if func.prop_name == "BW_BWb" or func.prop_name == "BW_BWk"%}

        {%- if func.Sbc.b1 == 'b124_kk' %}
        bw1 = self.BW_BW_f0_k0({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        bw2 = self.BW_BW_f2_k0({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        {%- endif %}

        {%- if func.Sbc.b1 == 'b124_pipi' %}
        bw1 = self.BW_BW_f0_b1({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        bw2 = self.BW_BW_f2_b1({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        {%- endif %}

        amp_caled_1 = {{func.amp}}[[0, 2], :, :]  # SS and DS
        amp_caled_2 = {{func.amp}}[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct({{func.const}}, {{func.theta}})
        {%- if func.inta_const is defined %}
        {# is defined是jinja2的语法，判断func.inta_const是否在func里面被定义了 #}
        inta_ph = dplex.dconstruct({{func.inta_const}}, {{func.inta_theta}})
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        {%- endif %}
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        {% else %}
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
        {%- if func.inta_const is defined %}
        inta_ph = dplex.dconstruct({{func.inta_const}}, {{func.inta_theta}})
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        {%- endif %}
        phif = dplex.deinsum_ord("ijk,li->ljk", {{func.amp}}, const_ph)#给每个振幅乘上相位因子
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        {% endif %}
        return phif

{%- if func.prop_name != "BW_BWb" and func.prop_name != "BW_BWk"%}
    def {{func.prop_name}}(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        {%- if info.merge == 'phi' %}
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        # a = self.BW_phi({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        {%- if func.Sbc.f == 'f_kk' or func.Sbc.f == 'f_pipi' %}
        b = np.moveaxis(vmap(partial(self.{{func.prop.prop_f.name}},Sbc={{func.Sbc.f}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        {%- endif %}

        {%- if func.Sbc.b1 == 'b124_pipi' or func.Sbc.b1 == 'b124_kk'%}
        b = np.moveaxis(vmap(partial(self.{{func.prop.prop_f.name}},Sbc={{func.Sbc.b1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        {%- endif %}

        {%- if func.Sbc.kst2_1 == 'kst2_124_kk'%}
        b = np.moveaxis(vmap(partial(self.{{func.prop.prop_f.name}},Sbc={{func.Sbc.kst2_1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        {%- endif %}

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
{% else %}
    {%- if func.Sbc.b1 == 'b124_pipi' %} 
    def BW_BW_f0_b1(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        b = np.moveaxis(vmap(partial(self.BW_f0_b1,Sbc={{func.Sbc.b1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    def BW_BW_f2_b1(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        b = np.moveaxis(vmap(partial(self.BW_f2_b1,Sbc={{func.Sbc.b1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    {%- endif %}
    {%- if func.Sbc.b1 == 'b124_kk' %} 
    def BW_BW_f0_k0(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        b = np.moveaxis(vmap(partial(self.BW_f0_k0,Sbc={{func.Sbc.b1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    def BW_BW_f2_k0(self, {{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}}):
        a = self.{{func.prop.prop_phi.name}}({{func["prop"]["prop_phi"]["paras"]|join(',')}})
        b = np.moveaxis(vmap(partial(self.BW_f2_k0,Sbc={{func.Sbc.b1}}))({{func["prop"]["prop_f"]["_paras"]|join(',')}}),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    {%- endif %}
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

        {% if func.prop_name == "BW_BWb" or func.prop_name == "BW_BWk"%}

        {%- if func.Sbc.b1 == 'b124_kk' %}
        bw1 = self.BW_BW_f0_k0({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        bw2 = self.BW_BW_f2_k0({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        {%- endif %}

        {%- if func.Sbc.b1 == 'b124_pipi' %}
        bw1 = self.BW_BW_f0_b1({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        bw2 = self.BW_BW_f2_b1({{func["prop"]["prop_phi"]["paras"]|join(',')}},{{func["prop"]["prop_f"]["paras"]|join(',')}})
        {%- endif %}

        amp_caled_1 = {{func.amp}}[[0, 2], :, :]  # SS and DS
        amp_caled_2 = {{func.amp}}[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct({{func.const}}, {{func.theta}})
        {%- if func.inta_const is defined %}
        inta_ph = dplex.dconstruct({{func.inta_const}}, {{func.inta_theta}})
        # const_ph = dplex.deinsum("lij,l->lij", const_ph, inta_ph)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        {%- endif %}
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        {% else %}
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
        {%- if func.inta_const is defined %}
        inta_ph = dplex.dconstruct({{func.inta_const}}, {{func.inta_theta}})
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        {%- endif %}
        phif = dplex.deinsum_ord("ijk,li->ljk", {{func.amp}}, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        {% endif %}
        return phif
{%- endmacro %}