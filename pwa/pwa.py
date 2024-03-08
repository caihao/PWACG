{% import "pwa/calculate_annex.py" as annex with context%}
{% extends "pwa/calculate_frame.py" %}

{% block likelihood %}
    {{annex.data_likelihood()}}
    {{annex.mc_likelihood()}}
{% endblock %}