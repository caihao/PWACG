import numpy as onp
import jax.numpy as np
import uproot_methods
import sys
import os

# 将多个npz文件组和在一起
n = 4
if __name__ == "__main__":
    name = "../raw_data/Momentum_" + str(1) + ".npz"
    mom = onp.load(name)
    _Kp = mom["Kp"][:,:]
    _Km = mom["Km"][:,:]
    _Pip = mom["Pip"][:,:]
    _Pim = mom["Pim"][:,:]
    for i in range(n-1):
        ver = i + 2
        name = "../raw_data/Momentum_" + str(ver) + ".npz"
        mom = onp.load(name)
        _Kp = onp.append(_Kp,mom["Kp"][:,:])
        _Km = onp.append(_Km,mom["Km"][:,:])
        _Pip = onp.append(_Pip,mom["Pip"][:,:])
        _Pim = onp.append(_Pim,mom["Pim"][:,:])

    print(_Kp.shape)
    Kp = onp.array(_Kp).reshape(-1,4)
    print(Kp.shape)
    Km = onp.array(_Km).reshape(-1,4)
    Pip = onp.array(_Pip).reshape(-1,4)
    Pim = onp.array(_Pim).reshape(-1,4)
    onp.savez("../data/Momentum.npz",Kp=Kp,Km=Km,Pip=Pip,Pim=Pim)
