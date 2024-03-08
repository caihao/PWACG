import seaborn as sns
import matplotlib.pyplot as plt
import numpy as onp
import jax
import jax.numpy as np
import time
import scipy.optimize as opt
import ROOT as rt
from jax import device_put


def readroot(name):
    # rt.ROOT.EnableImplicitMT()
    f = rt.TFile(name)
    tree = f.Get("tr")
    # tree = f.tr
    # entries = tree.GetEntriesFast()
    mom = onp.asarray([[tree.Kp1X, tree.Kp1Y, tree.Kp1Z, tree.Kp1E,
                        tree.Km1X, tree.Km1Y, tree.Km1Z, tree.Km1E,
                        tree.Kp2X, tree.Kp2Y, tree.Kp2Z, tree.Kp2E,
                        tree.Km2X, tree.Km2Y, tree.Km2Z, tree.Km2E
                        ] for event in tree])
    fastor = onp.asarray([[tree.phif001X, tree.phif001Y, tree.phif021X, tree.phif021Y] for event in tree])
    phif001 = (fastor[0:120000,0:2])
    phif021 = (fastor[0:120000,2:4])
    phif001MC = (fastor[80000:400000,0:2])
    phif021MC = (fastor[80000:400000,2:4])
    Kp = (mom[0:120000,0:4])
    Km = (mom[0:120000,4:8])
    Pip = (mom[0:120000,8:12])
    Pim = (mom[0:120000,12:16])
    KpMC = (mom[80000:400000,0:4])
    KmMC = (mom[80000:400000,4:8])
    PipMC = (mom[80000:400000,8:12])
    PimMC = (mom[80000:400000,12:16])
    print("have get tensor")
    return phif001, phif021, phif001MC, phif021MC, Kp, Km, Pip, Pim, KpMC, KmMC, PipMC, PimMC


def get_mom(name):
    f = rt.TFile(name)
    tree = f.tr
    mom = onp.asarray([[tree.Kp1X, tree.Kp1Y, tree.Kp1Z, tree.Kp1E,
                        tree.Km1X, tree.Km1Y, tree.Km1Z, tree.Km1E,
                        tree.Kp2X, tree.Kp2Y, tree.Kp2Z, tree.Kp2E,
                        tree.Km2X, tree.Km2Y, tree.Km2Z, tree.Km2E
                        ] for event in tree])
    Kp = (mom[0:120000,0:4])
    Km = (mom[0:120000,4:8])
    Pip = (mom[0:120000,8:12])
    Pim = (mom[0:120000,12:16])
    
    print("complete get four-momentum")
    return Kp, Km, Pip, Pim

def get_mom_truth(name):
    f = rt.TFile(name)
    tree = f.data_tr
    mom = onp.asarray([[tree.KpX, tree.KpY, tree.KpZ, tree.KpE,
                        tree.KmX, tree.KmY, tree.KmZ, tree.KmE,
                        tree.pipX, tree.pipY, tree.pipZ, tree.pipE,
                        tree.pimX, tree.pimY, tree.pimZ, tree.pimE
                        ] for event in tree])
    Kp = (mom[:,0:4])
    Km = (mom[:,4:8])
    Pip = (mom[:,8:12])
    Pim = (mom[:,12:16])
    
    print("complete get four-momentum")
    return Kp, Km, Pip, Pim