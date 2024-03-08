import os
import sys
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
from ROOT import Math as root_math
from ROOT import TMath

from rendered_scripts import {{jinja_fit_info.lasso.CodeScript|replace('.py','')}} as lasso_object

if __name__ == '__main__':
    logger = lasso_object.Logger("lasso")
    args = lasso_object.args()
    {% for key, value in run_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    {% for key, value in data_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    
    cl = lasso_object.Control(args)
    cl.run_multiprocess()