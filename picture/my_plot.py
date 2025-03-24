import ROOT
import numpy as onp
from ROOT import TH1D, TLorentzVector, TCanvas, TGraph, TMultiGraph, gStyle, gPad, TPaveText, TLegend, TObject, TLatex, TPaveText, TH2F

# Format for data points
def FormatData(datahist):
    datahist.SetMarkerStyle(20)
    datahist.SetMarkerSize(1)
    datahist.SetLineWidth(2)

    FormatAxis(datahist.GetXaxis())
    FormatAxis(datahist.GetYaxis())


def FormatAxis(axis):
    axis.SetLabelFont(42)
    axis.SetLabelSize(0.06)
    axis.SetLabelOffset(0.01)
    axis.SetNdivisions(510)
    axis.SetTitleFont(42)
    axis.SetTitleColor(1)
    axis.SetTitleSize(0.07)
    axis.SetTitleOffset(1.15)
    axis.CenterTitle()
  

def NameAxes(datahist, xname, yname):
    if xname:
        datahist.GetXaxis().SetTitle(xname)
    if yname:
        datahist.GetYaxis().SetTitle(yname)


# Format for main MC (red line)
def FormatMC1(mc1hist, n = 0):
    mc1hist.SetLineColor(6+6*(n+1))
    mc1hist.SetLineWidth(2)


# Format for second MC or background
# (Blue shaded area)
def FormatMC2(mc2hist):
    mc2hist.SetLineColor(4)
    # mc2hist.SetFillColor(4)
    mc2hist.SetLineWidth(3)
    # mc2hist.SetFillStyle(3001)


# Write "BESIII" in the upper right corner
def WriteBes3():
    bes3 = TLatex(0.94,0.94, "BESIII")
    bes3.SetNDC()
    bes3.SetTextFont(72)
    bes3.SetTextSize(0.1)
    bes3.SetTextAlign(33)
    return bes3


# Write "Preliminary" below BESIII -
# to be used together with WriteBes3()
def WritePreliminary():
    prelim = TLatex(0.94,0.86, "Preliminary")
    prelim.SetNDC()
    prelim.SetTextFont(62)
    prelim.SetTextSize(0.055)
    prelim.SetTextAlign(33)
    return prelim
  

# Make a legend; 
# position will have to change depending on the data shape
def MakeLegend(hist_list,hist_info,xlow=0.87,ylow=0.45,xhi=0.945,yhi=0.945):
    leg = TLegend(xlow, ylow, xhi, yhi)
    leg.AddEntry(hist_list[0], "Data", "LEP")
    leg.AddEntry(hist_list[1], "MC", "L")

    for i in range(2,len(hist_list)):
        leg.AddEntry(hist_list[i], hist_info[i], "L")

    leg.SetFillColor(0)
    leg.SetTextFont(22)
    return leg

def MakeLegendSingle(hist_list,legend_info,xlow=0.65,ylow=0.7,xhi=0.95,yhi=0.95):
    pt = TPaveText(xlow, ylow, xhi, yhi, "BRNDC")
    SetPaveText(pt)
    if legend_info:
        for value in legend_info:
            pt.AddText(value)

    return pt

def SetPaveText(pt):
     pt.SetFillStyle(0)
     pt.SetBorderSize(0)
     pt.SetTextAlign(10)
     pt.SetTextFont(22)
    #  pt.SetTextSize(0.04)

# Set the general style options
def SetStyle():
    # No Canvas Border
    gStyle.SetCanvasBorderMode(0)
    gStyle.SetCanvasBorderSize(0)
    # White BG
    gStyle.SetCanvasColor(10)
    # Format for axes
    gStyle.SetLabelFont(42,"xyz")
    gStyle.SetLabelSize(0.06,"xyz")
    gStyle.SetLabelOffset(0.01,"xyz")
    gStyle.SetNdivisions(510,"xyz")
    gStyle.SetTitleFont(42,"xyz")
    gStyle.SetTitleColor(1,"xyz")
    gStyle.SetTitleSize(0.07,"xyz")
    gStyle.SetTitleOffset(1.15,"xyz")
    # No pad borders
    gStyle.SetPadBorderMode(0)
    gStyle.SetPadBorderSize(0)
    # White BG
    gStyle.SetPadColor(10)
    # Margins for labels etc.
    gStyle.SetPadLeftMargin(0.17)
    gStyle.SetPadBottomMargin(0.17)
    gStyle.SetPadRightMargin(0.05)
    gStyle.SetPadTopMargin(0.05)
    # No error bars in x direction
    gStyle.SetErrorX(0)
    # Format legend
    gStyle.SetLegendBorderSize(0)
    # statistics box
    gStyle.SetOptStat(0)


# Style options for "final" plots
# (no stat/fit box)
def SetPrelimStyle():
    gStyle.SetOptDate(0)
    gStyle.SetOptStat(0)
    gStyle.SetOptFit(0)
    gStyle.SetOptTitle(0)


# Style options for internal meetings
# (stat/fit box)
def SetMeetingStyle():
    gStyle.SetOptDate(0)
    gStyle.SetOptTitle(0)
    gStyle.SetOptStat(1111)
    gStyle.SetStatBorderSize(1)
    gStyle.SetStatColor(10)
    gStyle.SetStatFont(42)
    gStyle.SetStatFontSize(0.03)
    gStyle.SetOptFit(1111)

def PlotSimp(filename,_hist):
    SetStyle()
    c1 = TCanvas("bes3plots","BESIII Plots", 1200,900)
    FormatData(_hist)

    _hist.Draw("axis")
    _hist.Draw("E1")
    leg = TLegend(0.65, 0.7, 0.95, 0.95)
    leg.AddEntry(_hist, "Toy MC", "LEP")
    leg.SetFillColor(0)
    leg.SetTextFont(22)
    leg.Draw()

    print("{}.png".format(filename))
    c1.SaveAs("{}.png".format(filename))

# Plot a data MC plot
def PlotDataMC(filename,  # Name for the output files, without extension 
        hist_list, # a list carrer histogram
        singleORset=0, # 0 draw set of mods, 1 draw single mods
        legend_info=None,
		prelim=1, # Use 1 for Preliminary plot, 2 for a publication plot, and 0 for a meeting plot with 
		):
      
    SetStyle()

    if prelim:
        SetPrelimStyle()
    else:
        SetMeetingStyle()

    c1 = TCanvas("bes3plots","BESIII Plots", 1200,900)

    y_max = max([hist.GetMaximum() for hist in hist_list])  # 找到所有直方图的最大值
    for hist in hist_list:
        hist.GetYaxis().SetRangeUser(0, y_max * 1.1)  # 统一设置 Y 轴范围

    FormatData(hist_list[0])
    FormatMC2(hist_list[1])
    for i in range(2,len(hist_list)):
        FormatMC1(hist_list[i],i)
    
    hist_list[0].Draw("axis")
    hist_list[0].Draw("E1")
    for i in range(1,len(hist_list)):
        hist_list[i].Draw("SAME HIST PMC PLC")
    hist_list[0].Draw("axissame")

    # if prelim:
    #     bes = WriteBes3()
    #     bes.Draw()
    #     if prelim == 1:
    #         prel = WritePreliminary()
    #         prel.Draw()

    if singleORset:
        pt = MakeLegendSingle(hist_list, legend_info)
        pt.Draw()
    else:
        leg = MakeLegend(hist_list, legend_info)
        leg.Draw()

    # print("{}.eps".format(filename))
    # c1.SaveAs("{}.eps".format(filename))
    print("{}.png".format(filename))
    c1.SaveAs("{}.png".format(filename))


def PloatScatter(datahist,filename):
    SetStyle()
    # ROOT.gStyle.SetPaintTextFormat("4.1f")
    # ROOT.gStyle.SetTextFont(42)
    # ROOT.gStyle.SetPalette(55,0)
    c1 = TCanvas("bes3plots","BESIII Plots",1200,900)
    FormatData(datahist)
    datahist.SetMarkerStyle(2)
    datahist.SetStats(0)
    datahist.Draw("col text")
    c1.SaveAs("{}.png".format(filename))
