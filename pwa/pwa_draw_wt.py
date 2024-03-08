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
        {% for lh in lh_coll %}
        self.jit_weight_{{lh.tag}} = jit(self.weight_{{lh.tag}})
        self.jit_weight_truth_{{lh.tag}} = jit(self.weight_truth_{{lh.tag}})
        {% endfor %}
{% endblock %}