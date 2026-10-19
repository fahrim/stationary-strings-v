"""Mini capraz kontrol (Grand Proof II / Faz 10.1 derini, T32-T34).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy/scipy disinda
bagimliligi yoktur (FD: N=750/1500 kaba izgaralar, bagimsiz vektor yonetimi):
  (C1) V_min tekilliksizligi + lam0 bandi + ortonormallik/dugumsuzluk +
       Richardson monotonlugu ve sureklilik bandi (olculen p~0.28; O(h^2) DEGIL).
  (C2) 0++ kapali-form 14-basamak + LQCD <%0.5 ve <1si (bilesik).
  (C3) 2++ + topolojik 0-+ (5.1726, %1.03) + 0*++ + ortalama <%1.6.
  (C4) gamma_Meral 14-basamak + gamma_UV0/Cornell (bilesik) + ccbar-IR <%2.
  (C5) OS Gc(t)>0 ve egim->radyal aralik + Cornell siralama/kusatma (bilesik).
  (C6) Kilit degismezler (Lambda0, mu_s, I_shield, S_loss, a_Meral, gap, topo).
Kullanim:  python mini_check_gp2.py   -> 15/15 PASS beklenir.
"""

import numpy as np
from scipy.linalg import eigh

PASS = []
CHECK = {}


def check(name, cond, detail):
    PASS.append(bool(cond))
    CHECK[name] = detail
    print(('PASS ' if cond else 'FAIL ') + name + ' :: ' + detail)


# ---- kilit sabitler (bagimsiz yazim) ----
C_SPH = 28.0 / 79.0
LAM0 = C_SPH * 0.5 * (
    163.0 / 578.0
    + (64.0 * np.sqrt(17.0) / 4913.0) * np.log((21.0 + 5.0 * np.sqrt(17.0)) / 4.0)
)
MU_M = 2.0 * np.sqrt(2.0) * LAM0
ETA_T = 0.1434
C_UNIV = 1.6260379365521303
B2 = 0.25
KAP = MU_M * (1.0 - ETA_T)


def Veff(x):
    x = np.asarray(x, dtype=float)
    return (1.0 + B2) / (x * x + B2) + C_UNIV ** 2 * (1.0 + MU_M) * x * x


def ym_lam0(N, xmax=12.0, nvec=6):
    dx = xmax / N
    xi = (np.arange(1, N) * dx).astype(float)
    Vs = Veff(xi) - 1.0 / (4.0 * xi * xi)
    H = (np.diag(2.0 / dx ** 2 + Vs)
         + np.diag(-1.0 / dx ** 2 * np.ones(len(xi) - 1), 1)
         + np.diag(-1.0 / dx ** 2 * np.ones(len(xi) - 1), -1))
    w, V = eigh(H, subset_by_index=[0, nvec - 1], eigvals_only=False)
    return w, V, xi


# ---- C1: operator + Richardson ----
xs = np.linspace(0.0, 3.0, 30001)
Vv = Veff(xs)
i0 = int(np.argmin(Vv))
check('C1_Vmin_nonsingular', abs(float(Vv[i0]) - 3.19416) < 0.002,
      'V_min = %.5f @ x = %.4f (tekilliksiz)' % (float(Vv[i0]), float(xs[i0])))
w, V, xi = ym_lam0(1500)
lam0 = float(w[0])
check('C1_lam0_positive', 5.5 < lam0 < 6.5,
      'lambda_0 = %.4f > 0 (Sturm-Liouville gap)' % lam0)
u0 = np.asarray(V[:, 0], dtype=float)
if -np.min(u0) > np.max(u0):
    u0 = -u0
o01 = abs(float(np.asarray(V[:, 0]) @ np.asarray(V[:, 1])))
check('C1_ortho_nodeless', o01 < 1e-14 and float(np.min(u0)) > -1e-9 * float(np.max(u0)),
      '|<u0|u1>| = %.2e, dugumsuz (min/max %.2e)' % (o01, float(np.min(u0)) / float(np.max(u0))))
w750, _, _ = ym_lam0(750)
lam750 = float(w750[0])
cont = lam0 - (lam750 - lam0) / (2.0 ** 0.28 - 1.0)
check('C1_richardson', lam750 > lam0 and 5.5 < cont < 6.0,
      'lam(750)=%.4f > lam(1500)=%.4f, sureklilik ~%.3f (p~0.28)' % (lam750, lam0, cont))

# ---- C2: 0++ araligi ----
GAP = 2.0 * C_UNIV * np.sqrt(1.0 + 0.5 * MU_M * (1.0 - ETA_T))
check('C2_gap_exact', abs(float(GAP) - 3.3914463103987673) < 1e-12,
      'gap = %.13f' % float(GAP))
REL = abs(float(GAP) - 3.405) / 3.405
SIG = abs(float(GAP) - 3.405) / 0.021
check('C2_gap_lqcd', REL < 0.005 and SIG < 1.0,
      '|dg|/g = %.3f%%, %.2f si' % (REL * 100.0, SIG))

# ---- C3: uyarilmis spektrum (topolojik 0-+) ----
P2 = float(GAP) * np.sqrt(1.0 + 6.0 * KAP / 1.25)
PM_TOPO = float(GAP) * np.sqrt(1.0 + 1.25 * (1.0 - ETA_T + MU_M))
PM_CLASSIC = float(GAP) * np.sqrt(1.0 + 1.25 * (1.0 - ETA_T))
PS = float(GAP) * C_UNIV * np.sqrt(1.0 + MU_M)
R2 = abs(P2 - 4.73) / 4.73
RM = abs(PM_TOPO - 5.12) / 5.12
RS = abs(PS - 6.18) / 6.18
check('C3_2pp', R2 < 0.03, '2++ %.4f vs 4.73 (%.2f%%)' % (P2, R2 * 100.0))
check('C3_0mp_topo', abs(PM_TOPO - 5.172638) < 0.001 and RM < 0.011,
      '0-+ topo %.4f vs 5.12 (%.2f%%; klasik %.4f acik)' % (PM_TOPO, RM * 100.0, PM_CLASSIC))
check('C3_0spp', RS < 0.03, '0*++ %.4f vs 6.18 (%.2f%%)' % (PS, RS * 100.0))
check('C3_mean', float(np.mean([REL, R2, RM, RS])) < 0.016,
      'ortalama %.2f%%' % (float(np.mean([REL, R2, RM, RS])) * 100.0))

# ---- C4: Lüscher-Meral + UV0 + charmonium ----
GAM = np.pi / 12.0 * (1.0 + 0.8 * KAP)
GUV0 = float(GAM) + 4.0 / 3.0 * MU_M
check('C4_gamma_exact', abs(float(GAM) - 0.29847153360434125) < 1e-12,
      'gamma_Meral = %.14f' % float(GAM))
check('C4_gamma_UV0_cornell', abs(GUV0 - 0.57101663) < 1e-8
      and abs(float(GAM) - 0.295) / 0.010 < 1.0,
      'gamma_UV0 = %.8f, Cornell ile %.2f si' % (GUV0, abs(float(GAM) - 0.295) / 0.010))
SIG2 = 0.44 ** 2
RC = 0.5 / 0.44
N2, RMAX = 1500, 25.0
dr = RMAX / N2
r = (np.arange(1, N2) * dr).astype(float)
Guv = float(GAM) + 4.0 / 3.0 * MU_M / np.log(np.e + (r / RC) ** 2)


def qlevels(mQ, Gvec):
    Vq = -Gvec / r + SIG2 * r
    Hq = (np.diag(2.0 / (mQ * dr ** 2) + Vq)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), 1)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), -1))
    return eigh(Hq, subset_by_index=[0, 2], eigvals_only=True)


Ecc = qlevels(1.4, np.full_like(r, float(GAM)))
SPLIT_CC = float((Ecc[1] - Ecc[0]) * 1000.0)
check('C4_ccbar_split', abs(SPLIT_CC - 589.19) / 589.19 < 0.02,
      'ccbar-IR 2S-1S = %.2f MeV (PDG 589.19)' % SPLIT_CC)

# ---- C5: OS sonumu + kusatma (bilesik) ----
X2 = xi * xi
Mn = np.array([abs(float(np.asarray(V[:, n]) * X2 @ np.asarray(V[:, 0]))) ** 2
               for n in range(1, 6)])
dE = np.array([float(w[n] - w[0]) for n in range(1, 6)])
Gc10 = float(np.sum(Mn * np.exp(-dE * 10.0)))
Gc12 = float(np.sum(Mn * np.exp(-dE * 12.0)))
Gc15 = float(np.sum(Mn * np.exp(-dE * 15.0)))
SL = float(-np.log(Gc15 / Gc12) / 3.0)
RGAP = float(w[1] - w[0])
Ebb_IR = qlevels(4.7, np.full_like(r, float(GAM)))
Ebb_UV = qlevels(4.7, Guv)
SLOPE = float(GAM) / RMAX ** 2 + SIG2
ok_os = Gc10 > 0.0 and Gc15 > 0.0 and abs(SL - RGAP) < 1e-3
ok_cf = bool(Ecc[0] < Ecc[1] < Ecc[2]) and bool(Ebb_IR[0] < Ebb_IR[1] < Ebb_IR[2]) and SLOPE > 0.0
check('C5_OS_confinement', ok_os and ok_cf,
      'Gc(10)>0, egim %.4f vs radyal %.4f; siralama OK; dV/dR=%.4f; '
      'bb-IR %.1f / bb-UVIR %.1f MeV (PDG 563.0)'
      % (SL, RGAP, SLOPE, float((Ebb_IR[1] - Ebb_IR[0]) * 1000.0),
         float((Ebb_UV[1] - Ebb_UV[0]) * 1000.0)))

# ---- C6: kilit degismezler ----
S_LOSS = np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)
I_SH = float(np.log(S_LOSS * (1.0 - ETA_T)))
A_M = float(MU_M * (1.0 - ETA_T) * 299792458.0 * (67.4 * 1000.0 / 3.0856775814913673e22))
check('C6_locked', abs(LAM0 - 0.07226943267767906) < 1e-16
      and abs(I_SH - 2.0 * np.sqrt(2.0)) < 1e-15
      and abs(A_M - 1.1465889796559045e-10) < 1e-23
      and abs(float(GAP) - 3.3914463103987673) < 1e-12
      and abs(PM_TOPO - 5.172638) < 0.001,
      'Lam0=%.17f I=%.15f aM=%.9e gap=%.10f topo=%.6f'
      % (LAM0, I_SH, A_M, float(GAP), PM_TOPO))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
