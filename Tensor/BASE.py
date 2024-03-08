import jax.numpy as np
import numpy as onp
from jax import jit

# definition of R and Q0
R = 1
Q0 = 0.197321 / R

# definition of G_{mu,nu}
def metric():
    G = onp.eye((4))
    G[0, 0] = -1
    G[1, 1] = -1
    G[2, 2] = -1
    return G

# definition of epsilon_{m,nu}
def leci_four():
    leci = onp.zeros((4, 4, 4, 4))
    leci[0, 1, 2, 3] = 1
    leci[0, 1, 3, 2] = -1
    leci[0, 3, 1, 2] = 1
    leci[3, 0, 1, 2] = -1
    leci[3, 0, 2, 1] = 1
    leci[0, 3, 2, 1] = -1
    leci[0, 2, 3, 1] = 1
    leci[0, 2, 1, 3] = -1
    leci[2, 0, 1, 3] = 1
    leci[2, 0, 3, 1] = -1
    leci[2, 3, 0, 1] = 1
    leci[3, 2, 0, 1] = -1
    leci[3, 2, 1, 0] = 1
    leci[2, 3, 1, 0] = -1
    leci[2, 1, 3, 0] = 1
    leci[2, 1, 0, 3] = -1
    leci[1, 2, 0, 3] = 1
    leci[1, 2, 3, 0] = -1
    leci[1, 3, 2, 0] = 1
    leci[3, 1, 2, 0] = -1
    leci[3, 1, 0, 2] = 1
    leci[1, 3, 0, 2] = -1
    leci[1, 0, 3, 2] = 1
    leci[1, 0, 2, 3] = -1
    return leci

# calculation of p.p in four dementions , return a double
# data frame (Px, Py, Pz, E)
@jit
def dot(a, b):
    G = metric()
    return np.einsum("i,j,ij->", a, b, G)

# calculation of invariant mass from four-momentun
@jit
def invm(a):
    return np.sqrt(dot(a, a))

# Eq.(2)
# calculation of \tilde{G}^{mu,nu} contravariance 逆变矢量
@jit
def Gt_con(a):
    G = metric()
    return G - np.einsum("i,j->ij", a, a) / dot(a, a)

# calculation of \tilde{G}_{mu,nu} covariance 协变矢量
@jit
def Gt_cov(a):
    gt_con = Gt_con(a)
    G = metric()
    return np.einsum("ij,ik,jl->kl", gt_con, G, G)

#Eq.(13)
# calculation of Qabc
@jit
def Qabc(a, b, c):
    Sa = dot(a, a)
    Sb = dot(b, b)
    Sc = dot(c, c)
    return (Sa + Sb - Sc)**2. / (4. * Sa) - Sb

#Eq.(14)
# calculation of Blatt-Weisskopf barrier factor. B_1(Qabc)
@jit
def B1(a, b, c):
    m_Qabc2 = Qabc(a, b, c)
    return np.sqrt(2.0 / (m_Qabc2 + Q0**2))

#Eq.(15)
# calculation of Blatt-Weisskopf barrier factor. B_2(Qabc)
@jit
def B2(a, b, c):
    m_Qabc2 = Qabc(a, b, c)
    return np.sqrt(13.0 / (m_Qabc2**2 + 3*m_Qabc2*(Q0**2) + 9*(Q0**4)))

#Eq.(16)
# calculation of Blatt-Weisskopf barrier factor. B_3(Qabc)
@jit
def B3(a, b, c):
    m_Qabc2 = Qabc(a, b, c)
    return np.sqrt(277.0 / (m_Qabc2**3 + 6*(m_Qabc2**2)*(Q0**2) + 45*m_Qabc2*(Q0**4) + 225*(Q0**6)))

#Eq.(17)
# calculation of Blatt-Weisskopf barrier factor. B_4(Qabc)
@jit
def B4(a, b, c):
    m_Qabc2 = Qabc(a, b, c)
    return np.sqrt(12746.0 / (m_Qabc2**4 + 10*(m_Qabc2**3)*(Q0**2) + 135*(m_Qabc2**2)*(Q0**4) + 1575*m_Qabc2*(Q0**6) + 11025*(Q0**8)))

# Eq.(10)
# definition of \tilde{t}^{(1)} covariance.
@jit
def T1_cov(a, b, c):
    gt_cov = Gt_cov(a) # \tilde{g}
    r = b - c  # r = p_b - p_c
    B = B1(a, b, c)
    return np.einsum("ij,j->i", gt_cov, r) * B

# Eq.(11)
# definition of \tilde{t}^{(2)} covariance.
@jit
def T2_cov(a, b, c):
    gt_cov = Gt_cov(a) # \tilde{g}
    r = b - c  # r = p_b - p_c
    t_r = np.einsum("ij,j->i", gt_cov, r) # \tilde{r}
    B = B2(a, b, c)
    return (np.einsum("i,j->ij", t_r, t_r) -
            (1. / 3) * dot(t_r, t_r) * gt_cov) * B

# Eq.(12)
# definition of \tilde{t}^{(3)} covariance.
@jit
def T3_cov(a, b, c):
    gt_cov = Gt_cov(a) # \tilde{g}
    r = b - c  # r = p_b - p_c
    t_r = np.einsum("ij,j->i", gt_cov, r) # \tilde{r}
    B = B3(a, b, c)
    gt_r_1 = np.einsum("ij,k->ijk",gt_cov,t_r) # \tilde{g}*r
    gt_r_2 = np.einsum("jk,i->ijk",gt_cov,t_r) # \tilde{g}*r
    gt_r_3 = np.einsum("ki,j->ijk",gt_cov,t_r) # \tilde{g}*r
    rrr = np.einsum("i,j,k->ijk", t_r, t_r, t_r) # r_{\mu}r_{\nv}r_{\lamda}
    return (rrr - (1. / 5) * dot(t_r, t_r) * (gt_r_1 + gt_r_2 + gt_r_3)) * B

# Eq.(20)
# P^{(2)}_{\mu \nu \mu^" \nu^"}
# {\mu \nu \mu^" \nu^"} to {i k j l}
@jit
def P2(a):
    gt_cov = Gt_cov(a) # \tilde{g}
    g_g_1 = np.einsum("ij,kl->ikjl", gt_cov, gt_cov)
    g_g_2 = np.einsum("il,kj->ikjl", gt_cov, gt_cov)
    g_g_3 = np.einsum("ik,jl->ikjl", gt_cov, gt_cov)
    return 1./2*(g_g_1+g_g_2) - 1./3*g_g_3

# Eq.(21)
# P^{(3)}_{\mu \nu \lamda \mu^" \nu^" \lamda^"}
# {\mu \nu \lamda \mu^" \nu^" \lamda^"} to {i k m j l n}
@jit
def P3(a):
    gt_cov = Gt_cov(a)
    # 6 terms ggg
    get_ggg  = np.einsum("ij,kl,mn->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg += np.einsum("ij,kn,ml->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg += np.einsum("il,kj,mn->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg += np.einsum("il,kn,mj->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg += np.einsum("in,kl,mj->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg += np.einsum("in,kj,ml->ikmjln", gt_cov, gt_cov, gt_cov)
    # 9 terms ggg
    get_ggg_  = np.einsum("ik,jl,mn->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("ik,ln,mj->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("ik,jn,ml->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("im,jn,kl->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("im,jl,kn->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("im,ln,kj->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("km,ln,ij->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("km,jl,in->ikmjln", gt_cov, gt_cov, gt_cov)
    get_ggg_ += np.einsum("km,jn,il->ikmjln", gt_cov, gt_cov, gt_cov)
    return -1./6 * get_ggg + 1./15 * get_ggg_

# Eq.(22)
# P^{(3)}_{\mu \nu \lamda \sigma \mu^" \nu^" \lamda^" \sigma^"}
# {\mu \nu \lamda \sigma \mu^" \nu^" \lamda^" \sigma^"} to {m n o p i j k l}
@jit
def P4(a):
    gt_cov = Gt_cov(a)
    # 24 terms gggg
    get_gggg  = np.einsum("mi,nj,ok,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mi,nj,ol,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mi,nk,oj,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mi,nk,ol,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mi,nl,oj,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mi,nl,ok,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,ni,ok,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,ni,ol,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,nk,oi,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,nk,ol,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,nl,oi,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mj,nl,ok,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,ni,oj,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,ni,ol,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,nj,oi,pl->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,nj,ol,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,nl,oi,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("mk,nl,oj,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,ni,oj,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,ni,ok,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,nj,oi,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,nj,ok,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,nk,oi,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg += np.einsum("ml,nk,oj,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    # 72 terms gggg
    get_g_g_g_g  = np.einsum("ij,mn,pl,ok->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,mn,ol,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,mo,pl,nk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,mo,nl,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,mp,ol,nk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,mp,nl,ok->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,no,pl,mk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,no,ml,pk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,np,ol,mk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,np,ml,ok->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,op,nl,mk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ij,op,ml,nk->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mn,pl,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mn,ol,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mo,pl,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mo,nl,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mp,ol,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,mp,nl,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,no,pl,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,no,ml,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,np,ol,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,np,ml,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,op,nl,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("ik,op,ml,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mn,pk,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mn,ok,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mo,pk,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mo,nk,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mp,ok,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,mp,nk,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,no,pk,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,no,mk,pj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,np,ok,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,np,mk,oj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,op,nk,mj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("il,op,mk,nj->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mn,pl,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mn,ol,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mo,pl,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mo,nl,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mp,ol,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,mp,nl,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,no,pl,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,no,ml,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,np,ol,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,np,ml,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,op,nl,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jk,op,ml,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mn,pk,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mn,ok,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mo,pk,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mo,nk,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mp,ok,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,mp,nk,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,no,pk,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,no,mk,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,np,ok,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,np,mk,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,op,nk,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("jl,op,mk,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mn,pj,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mn,oj,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mo,pj,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mo,nj,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mp,oj,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,mp,nj,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,no,pj,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,no,mj,pi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,np,oj,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,np,mj,oi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,op,nj,mi->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_g_g_g_g += np.einsum("kl,op,mj,ni->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    # 9 terms gggg
    get_gggg_  = np.einsum("ij,kl,mn,op->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("ij,kl,mo,np->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("ij,kl,mp,no->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("ik,jl,mn,op->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("ik,jl,mo,np->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("ik,jl,mp,no->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("il,jk,mn,op->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("il,jk,mo,np->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    get_gggg_ += np.einsum("il,jk,mp,no->mnopijkl", gt_cov,gt_cov,gt_cov,gt_cov)
    return 1./24 * get_gggg - 1./84 * get_g_g_g_g + 1./105 * get_gggg_

# defind by Eq.(23)
# \tilde{t}^(4)
@jit
def T4_cov(a, b, c):
    r = b - c  
    B = B4(a, b, c)
    p4 = P4(a)  # get P^{(4)}
    return np.einsum("ijklmnop,m,n,o,p->ijkl", p4, r, r, r, r) * B # use Eq.(23) calculate \tilde{t}^(4)

# defind Breit-wigner func
@jit
def BF(m, t, bc):
    sbc = invm(bc)
    bf = 1/(m**2 - sbc - i*m*t)
    return bf

# defind by Eq.(51)
# omega^{mu}
@jit
def omega_con(omega0,omega1,omega2):
    m = 0.769
    t = 0.151    
    leci_cov = leci_four()
    G = metric()
    leci_mu = np.einsum("ijkl,in->njkl",leci_cov,G)
    leci_p_p_p = np.einsum("ijkl,j,k,l",leci_mu,omega1,omega2,omega0)
    Omega = omega0 + omega1 + omega2 

    rho = omega1 + omega2 
    bf_rho12 = BF(m,t,rho)
    B1_omega0 = B1(Omega,rho,omega0)
    B1_rho12 = B1(rho,omega1,omega2)

    rho = omega1 + omega0 
    bf_rho10 = BF(m,t,rho)
    B1_omega2 = B1(Omega,rho,omega2)
    B1_rho10 = B1(rho,omega1,omega0)

    rho = omega0 + omega2 
    bf_rho02 = BF(m,t,rho)
    B1_omega1 = B1(Omega,rho,omega1)
    B1_rho02 = B1(rho,omega0,omega2)

    return leci_p_p_p*(B1_omega0*bf_rho12*B1_rho12 + B1_omega2*bf_rho10*B1_rho10 + B1_omega1*bf_rho02*B1_rho02)