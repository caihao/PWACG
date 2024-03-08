import os
import sys
import logging
import json
import logging.config
import numpy as onp
from functools import reduce

foo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(foo_path)
sys.path.append(foo_path)

{% for codescript in jinja_draw_info.draw_lh.CodeScript %}
from rendered_scripts import {{codescript|replace('.py','')}} as draw_object_{{loop.index0}}
{% endfor %}

class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_draw.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)

logger = Logger("draw")
logger = logging.getLogger("draw")



class run_bic(object):
    def __init__(self):
        self.args = draw_object_0.args()
        {% for key, value in run_config.items() -%}
        self.args.{{key}} = {{value}}
        {% endfor %}
        {% for key, value in data_config.items() -%}
        self.args.{{key}} = {{value}}
        {% endfor %}
        self.BIC = 0.0
        self.AIC = 0.0
        self.fcn = 0.0
        self.all_mod_dict = dict()

    def run_obj(self,draw):
        draw.run_multiprocess()
        draw.get_result_dict()
        logger.debug("{}".format(draw.frac_dict))
        logger.debug("{}".format(draw.fcn))
        new_dict = dict(filter(lambda item: item[1][2] > {{info.draw.frac_cut}}, draw.frac_dict.items()))
        logger.debug("{}".format(new_dict))
        bic = 2.0*draw.fcn["fcn"] + onp.log(draw.fcn["data_size"])*onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0))
        aic = 2.0*draw.fcn["fcn"] + 2.0*onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0))
        # print(2.0*draw.fcn["fcn"])
        # print(onp.log(draw.fcn["data_size"]))
        # print(onp.array(reduce(lambda x,y:x+y[0], new_dict.values(), 0)))
        return bic, aic

    def run_bic(self):
        {% for codescript in jinja_draw_info.draw_lh.CodeScript %}
        draw = draw_object_{{loop.index0}}.Control(self.args)
        bic, aic = self.run_obj(draw)
        logger.info("one part of BIC : {}".format(bic))
        self.BIC += bic
        self.AIC += aic
        self.all_mod_dict = {**self.all_mod_dict, **draw.frac_dict}
        self.fcn += draw.fcn["fcn"]
        {% endfor %}
        self.all_mod_name = list(self.all_mod_dict.keys())
        logger.info("BIC : {}".format(self.BIC))
        logger.info("AIC : {}".format(self.AIC))

    def run_select_mods(self):
        name_temp = list()
        new_BIC = self.BIC/2.0
        delete_name = ""
        while True:
            for name in self.all_mod_name:
                logger.info("cycle delete mod : {}".format(name))
                temp_BIC = 0.0
                self.args.bic_delete_mods = list()
                for name in name_temp:
                    self.args.bic_delete_mods.append(name)
                self.args.bic_delete_mods.append(name)
                {% for codescript in jinja_draw_info.draw_lh.CodeScript %}
                draw = draw_object_{{loop.index0}}.Control(self.args)
                bic = self.run_obj(draw)
                logger.info("one part of BIC : {}".format(bic))
                temp_BIC += bic
                {% endfor %}
                logger.info("new BIC : {}".format(temp_BIC))
                if new_BIC > temp_BIC:
                    new_BIC = temp_BIC     
                    delete_name = name
                logger.info("mini BIC : {}".format(new_BIC))
                logger.info("delete mod : {}".format(delete_name))

            if new_BIC > self.BIC:
                name_temp.append(delete_name)
                break
            name_temp.append(delete_name)
            self.BIC = new_BIC
        logger.info("delete mods : {}".format(name_temp))
    
if __name__ == "__main__":
    # A = run_bic()
    # A.run_bic()
    # A.run_select_mods()
    args = draw_object_0.args()
    {% for key, value in run_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    {% for key, value in data_config.items() -%}
    args.{{key}} = {{value}}
    {% endfor %}
    likelihood = 0.0
    {% for codescript in jinja_draw_info.draw_lh.CodeScript %}
    draw = draw_object_{{loop.index0}}.Control(args)
    draw.run_multiprocess()
    draw.get_result_dict()
    likelihood += draw.fcn
    {% endfor %}
    print("likelihood :",likelihood)