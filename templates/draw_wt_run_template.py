import os
import sys
import logging
import json
import logging.config
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_draw.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("draw")

{% for codescript in jinja_draw_info.draw_wt.CodeScript %}
from rendered_scripts import {{codescript|replace('.py','')}} as draw_object_{{loop.index0}}
{% endfor %}

args = draw_object_0.args()
{% for key, value in run_config.items() -%}
args.{{key}} = {{value}}
{% endfor %}
{% for key, value in data_config.items() -%}
args.{{key}} = {{value}}
{% endfor %}

likelihood = 0.0
{% for codescript in jinja_draw_info.draw_wt.CodeScript %}
draw = draw_object_{{loop.index0}}.Control(args)
draw.run_multiprocess()
draw.get_result_dict()
{% endfor %}