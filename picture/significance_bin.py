import numpy as onp
import ROOT
from ROOT import gStyle, gPad

def count_significance(h1, h2, bins, z_value):
    significance = onp.empty(bins)
    for i in range(bins):
        h1_bin_content = h1.GetBinContent(i)
        h2_bin_content = h2.GetBinContent(i)
        if h1_bin_content <= h2_bin_content:
            p_value_bin_content = 1 - ROOT.Math.inc_gamma_c(h2_bin_content, h1_bin_content)
            significance[i] = z_value(p_value_bin_content)
        else:
            p_value_bin_content = ROOT.Math.inc_gamma_c(h2_bin_content + 1, h1_bin_content)
            significance[i] = - z_value(p_value_bin_content)
        if p_value_bin_content >= 0.5:
            significance[i] = 0.
    print(significance)
    return significance

def draw_bin_sig(args_name, hist_list, hist_sig):
    c = ROOT.TCanvas("c", "canvas", 800, 800)
    gStyle.SetOptTitle(True)
    gStyle.SetOptStat(11)
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0)
    pad1.SetGridx()
    for n, hist in enumerate(hist_list):
        if n == 0:
            hist.Draw("E")
        else:
            hist.Draw("SAME HIST PLC PMC")
    legend = gPad.BuildLegend()
    legend.Draw()
    pad1.Draw()
    c.cd()
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.25)
    pad2.SetGridx()
    hist_sig.Draw("HIST")
    pad2.Draw()
    c.SaveAs("output/pictures/partial_mods_pictures/{}_weight.png".format(args_name))