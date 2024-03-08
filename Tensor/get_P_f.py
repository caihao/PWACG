import numpy as onp
import jax.numpy as np
import uproot_methods
import sys
import os

mom = onp.load("../raw_data/Momentum.npz")
Kp = mom["Kp"][:,:]
Km = mom["Km"][:,:]
Pip = mom["Pip"][:,:]
Pim = mom["Pim"][:,:]
data_f = Pip + Pim
P_f = onp.array([ c.p for c in (uproot_methods.TLorentzVectorArray.from_cartesian(data_f[:,0:1],data_f[:,1:2],data_f[:,2:3],data_f[:,3:4]))])
onp.savez("../data/Momentum_P.npz",Kp=Kp,Km=Km,Pip=Pip,Pim=Pim,P_f = P_f)