#!/usr/bin/env python3
# coding: utf-8
import json
import os
import re
import sys
import jinja2
import glob
from create_code import prepare_all_collection

class Create_Code(prepare_all_collection.Prepare_All):
    def __init__(self, dict_generator):
        super().__init__(dict_generator)
    
    def read_pwa(self, key):
        self.all_mod_info = list()
        for addr_pwa_info in self.json_pwa[key]:
            filename = glob.glob(addr_pwa_info+"*")
            if filename:
                with open(filename[0], encoding='utf-8') as f:
                    dict_json = json.loads(f.read())
                    if "mod_info" in dict_json:
                        self.all_mod_info.append(dict_json["mod_info"])
                    if "external_binding" in dict_json:
                        self._binding_point = {**self._binding_point, **dict_json["external_binding"]}
            else:
                print(" Warning! No such file \"{}\", You should run fit create such file".format(addr_pwa_info))

    def jinja_fit(self):
        print("jinja_fit:")
        self.mod_info = sum(self.all_mod_info, [])
        self.prepare_all()
        for module in self.jinja_fit_info.keys():
            self.render_dict.update(run_config = {**self.parameters["base"]["run_config"], **self.parameters[module]["run_config"]})
            self.render_dict.update(data_config = {**self.parameters["base"]["data_config"], **self.parameters[module]["data_config"]})
            address = self.jinja_fit_info[module]
            env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
            template = env.get_template("templates/" + address["CodeTemplate"])
            template_out = template.render(**self.render_dict)
            with open("rendered_scripts/" + address["CodeScript"], "w",encoding="utf-8") as f:
                f.writelines(template_out)
            run = env.get_template("templates/" + address["RunTemplate"])
            run_out = run.render(**self.render_dict)
            with open("run/" + address["RunScript"], "w",encoding="utf-8") as f:
                f.writelines(run_out)

    def jinja_draw(self):
        print("jinja_draw:")
        for n, mod_info in enumerate(self.all_mod_info):
            self.mod_info = mod_info
            self.prepare_all()
            for module in self.jinja_draw_info.keys():
                self.render_dict.update(run_config = {**self.parameters["base"]["run_config"], **self.parameters[module]["run_config"]})
                self.render_dict.update(data_config = {**self.parameters["base"]["data_config"], **self.parameters[module]["data_config"]})
                temp = self.render_dict
                address = self.jinja_draw_info[module]
                if module == "dplot" or module == "select":
                    with open("config/latex.json", encoding='UTF-8') as f:
                        latexjson = json.loads(f.read())
                        self.render_dict["sbc_collection"] = [sbc for sbc in list(latexjson["Sbc"].keys()) if re.match(".*"+self.render_dict["lh_coll"][0]["tag"],sbc)]
                if "ResultFile" in address:
                    self.render_dict.update(draw_result_file = address["ResultFile"][n])
                if "LassoResultFile" in address:
                    self.render_dict.update(lasso_result_file = address["LassoResultFile"][n])
                env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
                template = env.get_template("templates/" + address["CodeTemplate"])
                template_out = template.render(**self.render_dict)
                with open("rendered_scripts/" + address["CodeScript"][n], "w",encoding="utf-8") as f:
                    f.writelines(template_out)
                run = env.get_template("templates/" + address["RunTemplate"])
                run_out = run.render(**self.render_dict)
                with open("run/" + address["RunScript"], "w",encoding="utf-8") as f:
                    f.writelines(run_out)
                self.render_dict = temp
    
    def jinja_tensor(self):
        print("jinja_tensor:")
        self.initial_prepare()
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        template = env.get_template("Tensor/RunCacheTensor.py")
        template_out = template.render(**self.render_dict)
        with open("run/RunCacheTensor.py", "w",encoding="utf-8") as f:
            f.writelines(template_out)
