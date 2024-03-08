import os
import sys

import jax.numpy as np
from jax import jit
from jax import vmap

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import BASE as base

class FUNCTION(object):
    def __init__(self):
        # defind the four momentum of \psi(2s)
        self.psi2s = np.array([0.0,0.0,0.0,3.686])
        # self.psi2s = np.array([0.0405,0,0,3.686])
    
    def Modifiers(self):
        self.phif001 = jit(vmap(self._phif001))
        self.phif021 = jit(vmap(self._phif021))
        self.phif201 = jit(vmap(self._phif201))
        self.phif221 = jit(vmap(self._phif221))
        self.phif222 = jit(vmap(self._phif222))
        self.phif223 = jit(vmap(self._phif223))
        self.phif243 = jit(vmap(self._phif243))

        self.u_rho_1 = jit(vmap(self._u_rho_1))
        self.u_rho_2 = jit(vmap(self._u_rho_2))

        self.u_SS_1 = jit(vmap(self._u_SS_1))
        self.u_SS_2 = jit(vmap(self._u_SS_2))
        self.u_SD_1 = jit(vmap(self._u_SD_1))
        self.u_SD_2 = jit(vmap(self._u_SD_2))
        self.u_DS_1 = jit(vmap(self._u_DS_1))
        self.u_DS_2 = jit(vmap(self._u_DS_2))
        self.u_DD_1 = jit(vmap(self._u_DD_1))
        self.u_DD_2 = jit(vmap(self._u_DD_2))

    # Eq.(36)
    # <\phi f_0 | 01>
    def _phif001(self,phi,Kp,Km):
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{\mu}
        G = base.metric()
        return np.einsum("i,ij->j", t1_cov, G)

    # Eq.(37)
    # <\phi f_0 | 21>
    def _phif021(self,phi,f0,Kp,Km):
        t2_cov = base.T2_cov(self.psi2s,phi,f0) # \tilde{T}^{(2)}_{\mu \nu}
        G = base.metric()
        t2_con = np.einsum("ij,ik,jl->kl", t2_cov, G, G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{\nu}
        return np.einsum("ij,j->i", t2_con, t1_cov)

    # Eq.(41)
    # <\phi f_2 | 01>
    def _phif201(self,phi,f2,Kp,Km,Pip,Pim):
        G = base.metric()
        t2_cov = base.T2_cov(f2,Pip,Pim) # \tilde{t}^{(2)}_{\mu \nu} symmetrisation
        t2_con = np.einsum("ij,ik,jl->kl", t2_cov, G, G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{\nu}
        return np.einsum("ij,j->i",t2_con,t1_cov)

    # Eq.(42)
    # <\phi f_2 | 21>
    def _phif221(self,phi,f2,Kp,Km,Pip,Pim):
        G = base.metric()
        t2_cov = base.T2_cov(self.psi2s,phi,f2) # \tilde{T}^{(2)}_{\mu \alph}
        _t2_cov = base.T2_cov(f2,Pip,Pim) # \tilde{t}^{(2)}_{\alph \nu}
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{\nu}
        t2_con = np.einsum("ij,ik,jl->kl",t2_cov,G,G)
        t1_con = np.einsum("i,ik->k",t1_cov,G)
        return np.einsum("ij,jk,k->i",t2_con,_t2_cov,t1_con)

    # Eq.(43)
    # <\phi f_2 | 22>
    def _phif222(self,phi,f2,Kp,Km,Pip,Pim):
        G = base.metric()
        leci_cov = base.leci_four() # levi-cita
        leci_con = np.einsum("ijkl,im,jn,ko,lp->mnop",leci_cov,G,G,G,G)
        psi2s_cov = np.einsum("i,ij->j",self.psi2s,G) # \P_{\psi \alph}
        t2_phif2 = np.einsum("ij,il->lj",base.T2_cov(self.psi2s,phi,f2),G) # \tilde{T}^{(2) \delta}_{(\phi f_2) \beta}
        t2_34 = np.einsum("ij,ik->kj",base.T2_cov(f2,Pip,Pim),G) # \tilde{t}^{(2)\lamda}_{(34)\delta}
        t1_con = np.einsum("i,ik->k",base.T1_cov(phi,Kp,Km),G) # \tilde{t}^{(1)\nu}_{(12)}
        return (np.einsum("ijkl,j,mk,lnop,nm,o,p->i",leci_con,psi2s_cov,t2_phif2,leci_cov,t2_34,self.psi2s,t1_con) +
                np.einsum("ijkl,j,mk,mnop,nl,o,p->i",leci_con,psi2s_cov,t2_phif2,leci_cov,t2_34,self.psi2s,t1_con))
                # phi and psi2s is different

    # Eq.(44)
    # <\phi f_2 | 23>
    def _phif223(self,phi,f2,Kp,Km,Pip,Pim):
        G = base.metric()
        p3_cov = base.P3(self.psi2s) # \P^{(3)}_{\mmu \alph \ beta \gamma \delta \nu}
        p3_con = np.einsum("ijklmn,io,jp,kq,lr,ms,nt->opqrst",p3_cov,G,G,G,G,G,G)
        t2_phif2 = base.T2_cov(self.psi2s,phi,f2) # \tilde{T}^{(2)}_{(\phi f_2) \alph \beta}
        t2_34 = base.T2_cov(f2,Pip,Pim) # \tile{t}^{(3)}_{(34)\gamma \delta}
        t1_12 = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        return np.einsum("ijklmn,jk,lm,n->i",p3_con,t2_phif2,t2_34,t1_12)

    # Eq.(45)
    # <\phi f_2 | 43>
    def _phif243(self,phi,f2,Kp,Km,Pip,Pim):
        G = base.metric()
        t4_cov = base.T4_cov(self.psi2s,phi,f2) # \tilde{T}^{(4)\mu \nu \lamda \sigma}_{(\phi f_2)}
        t4_con = np.einsum("ijkl,im,jn,ko,lp->mnop",t4_cov,G,G,G,G)
        t2_cov = base.T2_cov(f2,Pip,Pim) # \tilde{t}^{(2)}_{(34)\lamda \sigma}
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        return np.einsum("ijkl,j,kl->i",t4_con,t1_cov,t2_cov)

    # Eq.(46)
    # U^{\mu}_{\rho}
    def _u_rho_1(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        leci_cov = base.leci_four()
        leci_con = np.einsum("ijkl,im,jn,ko,lp->mnop",leci_cov,G,G,G,G)
        # 选择i作为公式里的\nu，不同的指标提升的结果只影响正负号的对称性，因为在做复数取模的时候正负号的影响就消失了，
        # 所以可以制定一个指标为\nu
        leci_mu = np.einsum("ijkl,in->njkl",leci_cov,G)
        psi2s_cov = np.einsum("i,ij->j",self.psi2s,G) # \P_{\psi \alph}

        rho = Kp + Km + Pim
        T1_cov = base.T1_cov(self.psi2s,rho,Pip)
        T1_con = np.einsum("i,ij->j",T1_cov,G)
        t_phi4_cov = base.T1_cov(rho,phi,Pim)
        t_12_cov = base.T1_cov(phi,Kp,Km)
        # \epsilon^{\mu}_{\alpha \beta \gamma} * P^{\alpha}_{\psi} 
        # * \tilde{T}^{(1) \beta}_{\rho^, \Pi+} * \epsilon^{\gamma \delta \sigma \lamda} * P_{\psi \delta} * \tilde{t}^{(1)}_{(\phi \Pi-) \sigma} * \tilde{t}^{(1)}_{(\K+ \K-)}   
        phi4 = np.einsum("ijkl,j,k,lmno,m,n,o->i",leci_mu,self.psi2s,T1_con,leci_con,psi2s_cov,t_phi4_cov,t_12_cov)

        return phi4

    def _u_rho_2(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        leci_cov = base.leci_four()
        leci_con = np.einsum("ijkl,im,jn,ko,lp->mnop",leci_cov,G,G,G,G)
        # 选择i作为公式里的\nu，不同的指标提升的结果只影响正负号的对称性，因为在做复数取模的时候正负号的影响就消失了，
        # 所以可以制定一个指标为\nu
        leci_mu = np.einsum("ijkl,in->njkl",leci_cov,G)
        psi2s_cov = np.einsum("i,ij->j",self.psi2s,G) # \P_{\psi \alph}

        rho = Kp + Km + Pip
        T1_cov = base.T1_cov(self.psi2s,rho,Pim)
        T1_con = np.einsum("i,ij->j",T1_cov,G)
        t_phi4_cov = base.T1_cov(rho,phi,Pip)
        t_12_cov = base.T1_cov(phi,Kp,Km)
        # \epsilon_{\alpha \beta \gamma}^{\mu} p_{\psi}^{\alpha} 
        # \tilde{T}_{\left(\rho^{\prime} \Pi- \right)}^{(1) \beta} \epsilon^{\gamma \delta \sigma \lambda} p_{\psi \delta} \tilde{t}_{(\phi \Pi+) \sigma}^{(1)} \tilde{t}_{(K+ K-) \lambda}^{(1)}
        phi3 = np.einsum("ijkl,j,k,lmno,m,n,o->i",leci_mu,self.psi2s,T1_con,leci_con,psi2s_cov,t_phi4_cov,t_12_cov)

        return phi3

    # Eq.(47)
    # U^{\mu}_{b_1 SS}
    def _u_SS_1(self,phi,Kp,Km,Pip,Pim):
        b1 = Kp + Km + Pip
        gt_con = base.Gt_con(b1)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        amp123 = np.einsum("ij,j->i",gt_con,t1_cov)

        return amp123

    def _u_SS_2(self,phi,Kp,Km,Pip,Pim):
        b1 = Kp + Km + Pim
        gt_con = base.Gt_con(b1)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        amp124 = np.einsum("ij,j->i",gt_con,t1_cov)

        return amp124

    # Eq.(48)
    # U^{\mu}_{b_1 SD}
    def _u_SD_1(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()

        b1 = Kp + Km + Pip
        t2_cov = base.T2_cov(b1,phi,Pip) # \tilde{t}^{(2)}_{(\phi 3)\mu \nu}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        phi_3 = np.einsum("ij,j->i",t2_con,t1_cov)

        return phi_3

    def _u_SD_2(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()

        b1 = Kp + Km + Pim
        t2_cov = base.T2_cov(b1,phi,Pim) # \tilde{t}^{(2)}_{(\phi 4)\mu \nu}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        phi_4 = np.einsum("ij,j->i",t2_con,t1_cov)

        return phi_4

    # Eq.(49)
    # U^{\mu}_{b_1 DS}
    def _u_DS_1(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        
        b1 = Kp + Km + Pip
        Gt_cov = base.Gt_cov(b1)
        t2_cov = base.T2_cov(self.psi2s,b1,Pim) # \tilde{t}^{(2)}_{(b_1 4)\mu \nu}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        t1_con = np.einsum("i,ij->j",t1_cov,G)
        b1_4 = np.einsum("ij,jk,k->i",t2_con,Gt_cov,t1_con)

        return b1_4


    def _u_DS_2(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        
        b1 = Kp + Km + Pim
        Gt_cov = base.Gt_cov(b1)
        t2_cov = base.T2_cov(self.psi2s,b1,Pip) # \tilde{t}^{(2)}_{(b_1 3)\mu \nu}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        t1_con = np.einsum("i,ij->j",t1_cov,G)
        b1_3 = np.einsum("ij,jk,k->i",t2_con,Gt_cov,t1_con)

        return b1_3

    # Eq.(50)
    # U^{\mu}_{b_1 DD}
    def _u_DD_1(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        
        b1 = Kp + Km + Pip
        t2_cov = base.T2_cov(self.psi2s,b1,Pim) # \tilde{t}^{(2)}_{(b_1 4)\mu \lamda}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        _t2_cov = base.T2_cov(b1,phi,Pip) # \tilde{t}^{(2)}_{(\phi 3)\lamda \nu}
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        t1_con = np.einsum("i,ij->j",t1_cov,G)
        b1_4 = np.einsum("ij,jk,k->i",t2_con,_t2_cov,t1_con)

        return b1_4

    def _u_DD_2(self,phi,Kp,Km,Pip,Pim):
        G = base.metric()
        
        b1 = Kp + Km + Pim
        t2_cov = base.T2_cov(self.psi2s,b1,Pip) # \tilde{t}^{(2)}_{(b_1 3)\mu \lamda}
        t2_con = np.einsum("ij,in,jm->nm",t2_cov,G,G)
        _t2_cov = base.T2_cov(b1,phi,Pim) # \tilde{t}^{(2)}_{(\phi 4)\lamda \nu}
        t1_cov = base.T1_cov(phi,Kp,Km) # \tilde{t}^{(1)}_{(12)\nu}
        t1_con = np.einsum("i,ij->j",t1_cov,G)
        b1_3 = np.einsum("ij,jk,k->i",t2_con,_t2_cov,t1_con)

        return b1_3


if __name__ == "__main__":
    G = base.metric()
    leci_cov = base.leci_four()
    leci_con = np.einsum("ijkl,im,jn,ko,lp->mnop",leci_cov,G,G,G,G)
    leci_1 = np.einsum("ijkl,im->mjkl",leci_con,G)
    leci_2 = np.einsum("ijkl,jm->imkl",leci_con,G)
    print(leci_1 - leci_2)
