import numpy as onp
import jax.numpy as np
from jax import jit, vmap

def metric():
    G = onp.eye((4))
    G[0, 0] = -1
    G[1, 1] = -1
    G[2, 2] = -1
    return G

def dot(a, b):
    G = metric()
    return np.einsum("i,j,ij->", a, b, G)

def _Qabc(a, b, c):
    Sa = dot(a, a)
    Sb = dot(b, b)
    Sc = dot(c, c)
    return (Sa + Sb - Sc)**2. / (4. * Sa) - Sb

def _psi2s_Qabc(b, c):
    psi2s = np.array([0.0405,0,0,3.686])
    Sa = dot(psi2s, psi2s)
    Sb = dot(b, b)
    Sc = dot(c, c)
    return (Sa + Sb - Sc)**2. / (4. * Sa) - Sb

Qabc = jit(vmap(_Qabc))
psi2s_Qabc = jit(vmap(_psi2s_Qabc))

class mom_filter():
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        load_file = onp.load("../raw_data/Momentum.npz")
        self.Kp  = load_file["Kp"][:,:]
        self.Km  = load_file["Km"][:,:]
        self.Pip = load_file["Pip"][:,:]
        self.Pim = load_file["Pim"][:,:]
        self.phi = self.Kp + self.Km
        self.f = self.Pip + self.Pim
        self.b123 = self.Kp + self.Km + self.Pip
        self.b124 = self.Kp + self.Km + self.Pim
    
    def run(self):
        condition_1 = psi2s_Qabc(self.phi, self.f)
        condition_2 = psi2s_Qabc(self.b123, self.Pim)
        condition_3 = psi2s_Qabc(self.b124, self.Pip)
        address_1 = onp.squeeze(np.where((condition_1 > 0) & (condition_2 > 0) & (condition_3 > 0)), axis=None)
        Kp = self.Kp[address_1,:]
        Km = self.Km[address_1,:]
        Pip = self.Pip[address_1,:]
        Pim = self.Pim[address_1,:]
        print(Pim.shape)
        onp.savez("../data/all_data/Momentum.npz",Kp=Kp,Km=Km,Pip=Pip,Pim=Pim)

if __name__ == '__main__':
    _filter = mom_filter()
    _filter.run()