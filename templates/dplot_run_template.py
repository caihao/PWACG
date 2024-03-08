import os
import sys
import time
import logging
import logging.config
import json
foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)
os.system('rm -rf output/pictures/*/*')

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_draw.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("draw")

{% for codescript in jinja_draw_info.dplot.CodeScript %}
from rendered_scripts import {{codescript|replace('.py','')}} as dplot_object

####################################################################################
{% if draw_config.switch.likelihood %}
likelihood = dplot_object.Draw_Likelihood()
likelihood.draw_likelihood()
{% endif %}

{% if draw_config.switch.weight%}
weight = dplot_object.Draw_Weight()
weight.draw_weight()
{% endif %}

{% if draw_config.switch.pull%}
pull = dplot_object.Draw_Pull()
pull.draw_pull()
{% endif %}

{% if draw_config.switch.mods%}
## 拟合结果
mods = dplot_object.Draw_Mods()
mods.draw_mods()
## 预设值
# mods = dplot_object.Draw_Mods(args=args, defult=True)
# mods.draw_mods()
{% endif %}
####################################################################################
{% endfor %}

# dplot_object.Draw_correlation()

localtime = time.localtime(time.time())
new_dir = 'result_repo/{:0>2d}{:0>2d}-{:0>2d}{:0>2d}'.format(localtime[1],localtime[2],localtime[3],localtime[4])
os.system('rm -rf {}'.format(new_dir))
os.system('mkdir -p {}/output'.format(new_dir))
os.system('cp -r output/draw/*.md {}/output/'.format(new_dir))
os.system('cp -r output/draw/*.latex {}/output/'.format(new_dir))
os.system('cp -r output/fit/* {}/output'.format(new_dir))
os.system('cp -r output/pictures/ {}'.format(new_dir))
os.system('cp -r config/ {}'.format(new_dir))