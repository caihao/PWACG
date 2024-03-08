import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

from rendered_scripts import {{jinja_fit_info.pull.CodeScript|replace('.py','')}} as pull_object

if __name__ == '__main__':
    args = pull_object.args()
    {% for key, value in run_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    {% for key, value in data_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    
    cl = pull_object.Control(args)
    cl.run_multiprocess()
    {% if info.pull.write %}
    cl.save_in_npz()
    {% endif %}