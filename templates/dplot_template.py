import numpy as onp
import matplotlib.pyplot as plt
import sys
import os
import re
import json
import copy
import logging
import logging.config
import glob
import pandas as pd
import ROOT
from ROOT import TH1D, TLorentzVector, TCanvas, TGraph, TMultiGraph, gStyle, gPad, TPaveText, TLegend, TObject, TLatex, TH2D
from picture import my_plot as mpl

logger = logging.getLogger("draw")

class Dxplot(object):
    def invm(self,Pbc):
        _Pbc = Pbc * onp.array([-1,-1,-1,1])
        return onp.sum(Pbc * _Pbc,axis=1)
    
    def numpy2lorentz(self,P4):
        lorentz_list = list()
        for P in P4:
            lorentz_list.append(TLorentzVector(P[0],P[1],P[2],P[3]))
        return lorentz_list

    def calculate_costheta(self,LorentzVector):
        costheta_list = [L.CosTheta() for L in LorentzVector]
        return onp.asarray(costheta_list)

    def data_reader(self):
        {% for sbc in sbc_collection %}
        self.data_{{sbc}} = onp.sqrt(onp.load("data/real_data/{{sbc}}.npy"))
        # self.data_lorentz_{{sbc}} = self.numpy2lorentz(self.P_{{sbc}})
        # self.data_costheta_{{sbc}} = self.calculate_costheta(self.data_lorentz_{{sbc}})
        {% endfor %}

    def mc_reader(self):
        {% for sbc in sbc_collection %}
        self.mc_{{sbc}} = onp.sqrt(onp.load("data/mc_truth/{{sbc}}.npy"))
        # self.mc_lorentz_{{sbc}} = self.numpy2lorentz(self.mc_P_{{sbc}})
        # self.mc_costheta_{{sbc}} = self.calculate_costheta(self.mc_lorentz_{{sbc}})
        {% endfor %}

    def read_pull(self):
        all_pull = onp.load("output/pull/fit_result_pull.npz",allow_pickle=True)
        self.value = all_pull["fvalue"]
        self.error = all_pull["ferror"]
        for i, _ in enumerate(self.value):
            if _ is not None:
                if i == 0:
                    value = _
                else:
                    value = onp.vstack((value, _))
        for i, _ in enumerate(self.error):
            if _ is not None:
                if i == 0:
                    error = _
                else:
                    error = onp.vstack((error, _))

        self.value = value
        self.error = error
        print("pull value shape",self.value.shape)
        print("pull error shape",self.error.shape)
    
    def read_lasso(self):
        return onp.load("{{lasso_result_file}}")

    def read_weight(self):
        return onp.load("{{draw_result_file}}")

    def read_truth_weight(self):
        return onp.load("{{draw_result_file}}".replace(".","_truth."))


class Draw_Likelihood(Dxplot):
    def __init__(self):
        self.likelihood = self.read_likelihood()
    
    def draw(self, arg_name):
        c = TCanvas(arg_name, 'likelihood of ' + arg_name, 200, 50, 700, 500 )
        c.SetFillColor( 42 )
        c.SetGrid()
        gr = TGraph((self.likelihood[arg_name].shape)[0], self.likelihood[arg_name], self.likelihood[arg_name+"_lh"])
        gr.SetLineColor( 2 )
        gr.SetLineWidth( 4 )
        gr.SetMarkerColor( 4 )
        gr.SetMarkerStyle( 21 )
        gr.SetTitle( arg_name + ' likelihood scan graph' )
        gr.GetXaxis().SetTitle( 'scan mass' )
        gr.GetYaxis().SetTitle( 'likelihood' )
        gr.Draw( 'ACP' )
        c.cd()
        c.Draw()
        c.SaveAs("output/pictures/likelihood_pictures/" +arg_name+ ".png")
    
    def draw_likelihood(self):
        pass
        {% for mod in mods_collection %}
        {%- for num in mod.mass_index %}
        self.draw("{{mod.mass}}{{num}}")
        {%- endfor %}
        {%- for num in mod.width_index %}
        self.draw("{{mod.width}}{{num}}")
        {%- endfor %}
        {%- for num in mod.const_index %}
        self.draw("{{mod.const}}{{num}}")
        {%- endfor %}
        {%- for num in mod.theta_index %}
        # self.draw("{{mod.theta}}{{num}}")
        {%- endfor %}
        {% endfor %}


class Draw_Weight(Dxplot):
    def __init__(self):
        self.data_reader()
        self.mc_reader()
        self.all_wt = self.read_weight()
        self.all_mods_wt = self.all_wt["all_mods_wt"]
        self.all_mods_truth_wt = self.all_truth_wt["all_mods_wt"]
        self.all_data = dict()
    
    def fill_hist(self, arg_name, data, Fit):
        max_value = data.max() + 0.05
        min_value = data.min() - 0.05
        hist = TH1D(arg_name, arg_name + " weight", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(data.shape[0]):
            hist.Fill(data[i], self.all_mods_wt[i])
        if Fit:
            hist.Fit('gaus','S','Q')
        hist.GetXaxis().SetTitle(arg_name)
        hist.GetYaxis().SetTitle("events")
        return hist

    def draw(self, arg_name, hist, address):
        c = TCanvas(arg_name, arg_name, 900, 600)
        c.SetGrid()
        c.SetFillColor( 42 )
        c.GetFrame().SetFillColor( 21 )
        c.GetFrame().SetBorderSize( 6 )
        c.GetFrame().SetBorderMode( -1 )
        gStyle.SetOptStat(11)
        gStyle.SetOptFit(1011)
        c.cd()
        hist.Draw("E1")    
        c.Draw()
        c.SaveAs(address +arg_name+ ".png")

    def draw_weight(self):
        max_value = self.all_mods_wt.max() + 0.05
        min_value = self.all_mods_wt.min() - 0.05
        print("wt max", max_value)
        print("wt min", min_value)
        hist = TH1D("wt", "wt distribution", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(self.all_mods_wt.shape[0]):
            hist.Fill(self.all_mods_wt[i])
        self.draw("all_mods_wt", hist, "output/pictures/weight_pictures/")
        {% for sbc in sbc_collection %}
        hist_{{sbc}}_m = self.fill_hist("{{sbc}}_mass", self.mc_{{sbc}}, False)
        self.draw("{{sbc}}_mass", hist_{{sbc}}_m, "output/pictures/weight_pictures/")

        hist_{{sbc}}_t = self.fill_hist("{{sbc}}_theta", self.mc_costheta_{{sbc}}, False)
        self.draw("{{sbc}}_theta", hist_{{sbc}}_t, "output/pictures/weight_pictures/")
        {% endfor %}


class Draw_Mods(Dxplot):
    def __init__(self):
        self.data_reader()
        self.mc_reader()
        self.all_wt = self.read_weight()
        self.all_truth_wt = self.read_truth_weight()
        print(self.all_wt.files)
        self.fit_value = self.all_wt["fit_value"]
        self.error = self.all_wt["fit_error"]
        {% for lh in lh_coll %}
        self.wt_data = onp.load("data/weight/weight_{{lh.tag}}.npy")
        {% endfor %}
        with open("config/latex.json", encoding='UTF-8') as f:
            self.latexjson = json.loads(f.read())
            
    def draw(self, arg_name, hist_list, legend_info, max_value, min_value):
        for tag, value in self.latexjson["Sbc"].items():
            if re.match(tag+".*", arg_name):
                latex_tag = value
        mpl.NameAxes(hist_list[0], "M_{0}{1}{2} (GeV)".format("{",latex_tag,"}"), "Events/{:.2f} GeV".format((max_value - min_value)/{{draw_config.weight_option.bin}}))

        mpl.PlotDataMC("output/pictures/partial_mods_pictures/{}_weight".format(arg_name), hist_list, 0, legend_info)

    def fill_hist(self, data_hist, arg_name, data, max_value, min_value):
        coinfo = dict()
        coinfo["data"] = [data_hist,0]

        fit_result_wt = self.all_wt["all_mods_wt"]
        sum_wt = onp.sum(fit_result_wt)
        hist = TH1D("fit_result", "fit result", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(data.shape[0]):
            hist.Fill(data[i], fit_result_wt[i])
        hist.Scale(self.data_size/sum_wt)
        coinfo["mc"] = [hist,0]
        {% for func in func_info %}
        {%- for num in range(func.num_mod) %}
        {{func.calculate_func}}_wt = self.all_wt["{{func.calculate_func}}_{{num}}"]
        hist = TH1D("{{func.calculate_func}}_{{num}}", "{{func.calculate_func}}_{{num}} partial wave", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(data.shape[0]):
            hist.Fill(data[i], {{func.calculate_func}}_wt[i])
        hist.Scale(self.data_size/sum_wt)
        mod_name_list = {{func.mod_name_list[loop.index0]}}
        coinfo[list(mod_name_list.keys())[0]] = [hist]
        for tag, value in self.latexjson["mod"].items():
            if re.match(tag, list(mod_name_list.keys())[0]):
                coinfo[list(mod_name_list.keys())[0]].append(value)
        {%- endfor %}
        {% endfor %}
        all_hist = [i[0] for i in list(coinfo.values())]
        legend_info = [i[1] for i in list(coinfo.values())]
        self.draw(arg_name, all_hist, legend_info, max_value, min_value)
    
    def draw_single(self, arg_name, frac, hist_list, legend_info, max_value, min_value):
        for tag, value in self.latexjson["Sbc"].items():
            if re.match(tag+".*", arg_name):
                latex_tag = value
        mpl.NameAxes(hist_list[0], "M_{0}{1}{2} (GeV)".format("{",latex_tag,"}"), "Events/{:.2f} GeV".format((max_value - min_value)/{{draw_config.weight_option.bin}}))

        legend_text = list()
        for key, value in legend_info.items():
            legend_text.append("{} result:{:.6f}".format(key,self.fit_value[value]))
        legend_text.append("fit fraction : {:.6f}".format(frac))

        mpl.PlotDataMC("output/pictures/partial_mods_pictures/{}_weight".format(arg_name), hist_list, 1, legend_text)

    def fill_single_hist(self, data_hist, arg_name, data, max_value, min_value):
        all_hist = list()
        fit_fraction_dict = dict()
        self.fit_table = dict()

        fit_result_wt = self.all_wt["all_mods_wt"]
        fit_result_truth_wt = self.all_truth_wt["all_mods_wt"]
        sum_wt = onp.sum(fit_result_wt)
        sum_truth_wt = onp.sum(fit_result_truth_wt)
        hist_result = TH1D("fit_result", "fit result", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(data.shape[0]):
            hist_result.Fill(data[i], fit_result_wt[i])
        hist_result.Scale(self.data_size/sum_wt)
        {% for func in func_info %}
        {%- for num in range(func.num_mod) %}
        {{func.calculate_func}}_wt = self.all_wt["{{func.calculate_func}}_{{num}}"]
        hist = TH1D("{{func.calculate_func}}_{{num}}", "{{func.calculate_func}}_{{num}} partial wave", {{draw_config.weight_option.bin}}, min_value, max_value)
        for i in range(data.shape[0]):
            hist.Fill(data[i], {{func.calculate_func}}_wt[i])
        hist.Scale(self.data_size/sum_wt)
        all_hist.append(data_hist)
        all_hist.append(hist_result)
        all_hist.append(hist)
        mod_name_list = {{func.mod_name_list[loop.index0]}}
        self.fit_table = {**self.fit_table,**mod_name_list}
        p_sum_wt = onp.sum({{func.calculate_func}}_wt)
        # frac = p_sum_wt / sum_wt
        frac = onp.sum(self.all_truth_wt["{{func.calculate_func}}_{{num}}"])/sum_truth_wt
        fit_fraction_dict[list(mod_name_list.keys())[0]] = frac
        self.draw_single(arg_name+list(mod_name_list.keys())[0], frac, all_hist, list(mod_name_list.values())[0], max_value, min_value)
        all_hist.clear()
        {%- endfor %}
        {% endfor %}
        self.table_temp = sorted(fit_fraction_dict.items(), key=lambda item:item[1],reverse=True)

    def draw_mods(self):
        {{sbc_collection}}
        fit_fraction_coll = list()
        {% for sbc in sbc_collection %}
        max_value = self.mc_{{sbc}}.max() + 0.15
        min_value = self.mc_{{sbc}}.min() - 0.15
        save_all_wt = self.all_wt
        save_wt_data = self.wt_data
        if re.match("b.*","{{sbc}}"):
            temp_wt = dict()
            for key in self.all_wt.files:
                temp_wt[key] = onp.append(self.all_wt[key],self.all_wt[key])
            self.all_wt = temp_wt
            self.wt_data = onp.append(self.wt_data,self.wt_data)
        hist_{{sbc}} = TH1D("{{sbc}} data", "{{sbc}} data distribution ", {{draw_config.weight_option.bin}}, min_value, max_value)
        {% if info.fit.use_weight %}
        for i in range(self.data_{{sbc}}.shape[0]):
            hist_{{sbc}}.Fill(self.data_{{sbc}}[i],self.wt_data[i])
        self.data_size = onp.sum(self.wt_data)
        {% else %}
        for i in range(self.data_{{sbc}}.shape[0]):
            hist_{{sbc}}.Fill(self.data_{{sbc}}[i])
        self.data_size= self.data_{{sbc}}.shape[0]
        {% endif %}
        self.fill_hist(hist_{{sbc}}, "{{sbc}}", self.mc_{{sbc}}, max_value, min_value)
        self.fill_single_hist(hist_{{sbc}}, "{{sbc}}", self.mc_{{sbc}}, max_value, min_value)
        self.all_wt = save_all_wt
        self.wt_data = save_wt_data

        # max_value = self.mc_costheta_{{sbc}}.max() + 0.1
        # min_value = self.mc_costheta_{{sbc}}.min() - 0.1
        # hist_costheta_{{sbc}} = TH1D("{{sbc}} data", "{{sbc}} data distribution ", {{draw_config.weight_option.bin}}, min_value, max_value)
        # hist_costheta_{{sbc}}.GetXaxis().SetTitle("mass")
        # hist_costheta_{{sbc}}.GetYaxis().SetTitle("event")
        # # hist_{{sbc}}.SetMarkerStyle(12)
        # for i in range(self.data_costheta_{{sbc}}.shape[0]):
        #     hist_costheta_{{sbc}}.Fill(self.data_costheta_{{sbc}}[i])
        # self.fill_hist(hist_costheta_{{sbc}}, "{{sbc}}_theta", self.mc_costheta_{{sbc}})
        {% endfor %}
        sum_frac = 0
        aic_r = 0
        cut = 0.1
        for tu in self.table_temp:
            sum_frac += tu[1]
            table = self.fit_table[tu[0]]
            _temp = {"mod_name":tu[0],"fraction":tu[1],"mass":self.fit_value[list(table.values())[0]],"mass error":float(self.error[list(table.values())[0]]),"width":self.fit_value[list(table.values())[1]],"width error":float(self.error[list(table.values())[1]])}
            fit_fraction_coll.append(_temp)
            if tu[1] > cut:
                aic_r += 1
        fit_fraction_coll.append({"mod_name":"sum_fraction","fraction":sum_frac})
        logger.info("# table of fit result sort by frac")
        # logger.info("{}".format(json.dumps(fit_fraction_coll, ensure_ascii=False)))

        with open("output/draw/fit_fraction_table_{{lh_coll[0].tag}}.json","w") as json_file:
            json.dump(fit_fraction_coll, json_file)

        # add latex mod name
        for i, _mod in enumerate(fit_fraction_coll):
            for tag, value in self.latexjson["mod"].items():
                if re.match(tag+".*", _mod["mod_name"]):
                    fit_fraction_coll[i]["mod_name"] = "${}$".format(value)
        
        json_object = pd.DataFrame(fit_fraction_coll)
        # json_object = pd.read_json("output/draw/fit_fraction_table_{{lh_coll[0].tag}}.json")
        md_string = json_object.to_markdown()
        latex_string = json_object.to_latex(escape=False)
        logger.info("markdown file:\n{}".format(md_string))
        with open("output/draw/fit_fraction_table_{{lh_coll[0].tag}}.md","w") as md_file:
            md_file.write(md_string)
        with open("output/draw/fit_fraction_table_{{lh_coll[0].tag}}.latex","w") as latex_file:
            latex_file.write(latex_string)


class Draw_Pull(Dxplot):
    def __init__(self):
        self.read_pull()
        self.all_args = dict()
        {% for arg_name, arg in args_collection.items() %}
        self.all_args["{{arg_name}}"] = {{arg.value}}
        {% endfor %}
    
    def fill_draw(self, arg_name, all_arg_index):
        print(arg_name)
        for n, arg_index in enumerate(all_arg_index):
            value = onp.asarray(self.value[:, arg_index:arg_index+1])
            error = onp.asarray(self.error[:, arg_index:arg_index+1])
            pull_value = (value - self.all_args[arg_name][n]) / error
            hist = TH1D(arg_name+"_"+str(n), "pull distribution of number.{1} {0}".format(arg_name, str(n)), {{draw_config.pull_option.bin}}, {{draw_config.pull_option.min}}, {{draw_config.pull_option.max}})
            for i in range((pull_value.shape)[0]):
                hist.Fill(pull_value[i])
            hist.Fit('gaus','S','Q')
            hist.GetXaxis().SetTitle("pull")
            hist.GetYaxis().SetTitle("events")
            self.draw(arg_name+"_number"+str(n), hist, "output/pictures/pull_pictures/")

    def draw_pull(self):
        self.fill_draw("phi_m", [0])
        self.fill_draw("phi_w", [1])
        {% for key_arg, arg in args_dict.items() %}
        self.fill_draw("{{key_arg}}", {{arg}})
        {% endfor %}

# def Draw_correlation():
#         error_file = glob.glob("{{jinja_fit_info.fit.ResultFile}}/correlation"+"*")
#         if error_file:
#             correlation = onp.load(error_file[0])
#             print(correlation.shape)
#             num = (correlation.shape)[0]
#             correlation = onp.nan_to_num(correlation)
#             hist2d = TH2D("hist2d","data",num,0,num,num,0,num)
#             for i in range(num):
#                 for j in range(num):
#                     hist2d.Fill(i+0.5,num-j-0.5,correlation[i,j])
#                 hist2d.GetXaxis().SetBinLabel(i+1,"biname")
#                 hist2d.GetYaxis().SetBinLabel(num-i,"biname")
#             hist2d.SetMinimum(-1)
#             hist2d.SetMaximum(+1)
#             mpl.PloatScatter(hist2d, "output/pictures/likelihood_pictures/correlation_error")
#         else:
#             print("not exit correlation file")
