import os, glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib as mpl
from iminuit import Minuit

path = "/Users/albaburgosmondejar/Desktop/Dataset2/" 
eta_col1, eta_col2 = "l1_eta", "l2_eta"
qpt_col1, qpt_col2 = "l1_q_over_pt", "l2_q_over_pt"
label_col = "opposite_charge"  # 0 => SS, 1 => OS

PtBinning  = [20, 50, 100, 200, 2600] 
EtaBinning = [0, 1.37, 1.52, 2.0, 2.6]
# PtBinning  = [20, 35, 50, 75, 100,150, 200,1400, 2600] 
# EtaBinning = [0, 0.8, 1.37,1.45, 1.52,1.75, 2.0, 2.3,2.6]
NBINS_QPT = len(PtBinning) - 1
NBINS_ETA = len(EtaBinning) - 1
qpt_edges = PtBinning
eta_edges = EtaBinning

EPS = 1e-18

pkl_files = sorted(glob.glob(os.path.join(path, "*.pkl")))
df = pd.concat([pd.read_pickle(f) for f in pkl_files], ignore_index=True)
need = [eta_col1, eta_col2, qpt_col1, qpt_col2, label_col]

#Background subtraction and selection of Z-peak
# mask_low  = (df["m_l1l2"] >  71000) & (df["m_l1l2"] <  81000)
# mask_high = (df["m_l1l2"] > 101000) & (df["m_l1l2"] < 111000)
mask_Z    = (df["m_l1l2"] >= 81000) & (df["m_l1l2"] <= 101000)
# len_A = mask_low.sum()
# len_B = mask_Z.sum()
# len_C = mask_high.sum()
# N_sig = len_B - (len_A + len_C)/2
# factor_bck = N_sig / len_B
factor_bck = 0.976

df = df[need][mask_Z].copy()
df[qpt_col1] = ( 1/df[qpt_col1]).abs() *1e-3
df[qpt_col2] = (1/ df[qpt_col2]).abs()*1e-3
df[eta_col1] = (df[eta_col1]).abs()
df[eta_col2] = (df[eta_col2]).abs()

def hist4(df_in):
    data = df_in[[eta_col1, eta_col2, qpt_col1, qpt_col2]].to_numpy()
    H, _ = np.histogramdd(
        data,
        bins=(eta_edges, eta_edges, qpt_edges, qpt_edges)
    )
    return H.astype(np.float64)

SS = hist4(df[df[label_col] == 0])  * factor_bck
OS = hist4(df[df[label_col] == 1])  * factor_bck
Array_SS, Array_OS = SS, OS
ALL = SS + OS
# ALL = np.where(ALL <= 0.0, EPS, ALL)

def nll(par, Array_SS, Array_OS, eps=1e-18):
    SS = np.asarray(Array_SS, dtype=np.float64)
    OS = np.asarray(Array_OS, dtype=np.float64)

    E1, E2, P1, P2 = SS.shape
    assert E1 == E2 and P1 == P2
    NBINS_ETA, NBINS_PT = E1, P1

    par = np.asarray(par, dtype=np.float64).reshape(NBINS_ETA, NBINS_PT)

    Psum = par[:, None, :, None] + par[None, :, None, :]

    ALL = SS + OS
    ALL = np.where(ALL <= 0.0, eps, ALL)

    lam = ALL * Psum
    lam = np.where(lam <= 0.0, eps, lam)

    cell_ll = np.where(SS != 0.0, SS * np.log(lam) - lam, -lam)

    tri = np.triu(np.ones((NBINS_ETA, NBINS_ETA), dtype=np.float64))
    ll = np.sum(cell_ll * tri[:, :, None, None])
    ll = np.sum(cell_ll)

    return -ll

E = NBINS_ETA
P = NBINS_QPT  

NPAR = E*P
init = 5e-5
x0 = np.full(NPAR, init, dtype=float)

def nll_flat(*theta_flat):
    x = np.array(theta_flat, dtype=float)
    return nll(x, Array_SS, Array_OS)

m = Minuit(nll_flat, *x0)

for i in range(NPAR):
    m.limits[i] = (0.0, None)

m.migrad(ncall=10_000_000)
m.hesse()

theta = np.array(m.values).reshape(E, P)

eta_centers = 0.5*(np.array(eta_edges[:-1]) + np.array(eta_edges[1:]))
qpt_centers = 0.5*(np.array(qpt_edges[:-1]) + np.array(qpt_edges[1:]))
fig, ax = plt.subplots(figsize=(4,4))

pc = ax.pcolormesh(eta_edges, qpt_edges, theta, shading='auto')

ax.set_ylabel("pt (bin center)")
ax.set_xlabel("eta (bin center)")
ax.set_yscale('log') 

fig.colorbar(pc, ax=ax, label="theta")
ax.set_yticks(qpt_edges)
ax.set_yticklabels([f"{v:.2g}" for v in qpt_edges])

ax.set_xticks(eta_edges)
ax.set_xticklabels([f"{v:.2g}" for v in eta_edges])


for i in range(P):  
    for j in range(E): 
        
        if j != 1:
            ax.text(
                eta_centers[j],
                qpt_centers[i],
                f"{theta[i,j]:.2e}",
                ha="center", va="center"
            )

        print(theta[i,j])

plt.tight_layout()
plt.show()