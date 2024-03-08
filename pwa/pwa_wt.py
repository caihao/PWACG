{% import "pwa/calculate_annex.py" as annex with context%}
{% extends "pwa/calculate_frame.py" %}

{% block likelihood %}
    {{annex.likelihood()}}
        return -np.sum(self.wt*(np.log(d_tmp) - np.log(m_tmp)))
{% endblock %}

{% block weight %}
    {{annex.weight()}}
        wt = d_tmp/m_tmp
        wt_ave = np.average(wt)
        return wt/wt_ave
{% endblock %}

{% block jit_request %}
        self.jit_weight = jit(self.weight)
        self.wt = self.jit_weight(self.args_list)
{% endblock %}