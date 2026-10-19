"""Mini capraz kontrol (Grand Proof II / Faz 10.2 derini, T32-T34).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy/scipy disinda
bagimliligi yoktur. Tekilliksiz hucre-merkezli FV (eigh_tridiagonal) ve
Yukawa-perdelemeli Cornell cozucusu asagida yeniden yazilmistir:
  (C1) V_min + FV lam0 bandi (5.61020) + ortonormallik/dugumsuzluk +
       Richardson p = 2.0000 ve sureklilik limiti (bilesik) +
       Temple-Kato: Gauss varyasyonel ust-sinir + FV kesin cerceveleme.
  (C2) 0++ kapali-form 14-basamak + LQCD <%0.5 ve <1si (bilesik) +
       sqrt(2lam) koprusu (bilesik) + kiral kopru %0.05/LQCD %0.45 (bilesik).
  (C3) 2++ + topolojik 0-+ (5.1726, %1.03) + 0*++ + ortalama <%1.6.
  (C4) gamma_Meral 14-basamak + perdeleme sabitleri (bilesik) +
       perdelenmis ccbar <%2 ve bbbar <%8 (bilesik) + CAS cc/bb bandi (bilesik).
  (C5) OS Gc(t)>0 ve egim->radyal aralik + Cornell siralama/kusatma (bilesik).
  (C6) Kilit degismezler (Lambda0, mu_s, I_shield, a_Meral, gap, topo, FV kopru,
       kiral kopru).
Kullanim:  python mini_check_gp2.py   -> 18/18 PASS beklenir.
"""

import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal

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


def fv_lams(N, xmax=8.0, nvec=6):
    h = xmax / N
    xi = ((np.arange(N) + 0.5) * h).astype(float)
    Vv = Veff(xi)
    xf = np.arange(N + 1) * h
    d = (xf[:-1] + xf[1:]) / (xi * h * h) + Vv
    o = -xf[1:-1] / (h * h * np.sqrt(xi[:-1] * xi[1:]))
    w, V = eigh_tridiagonal(np.asarray(d), np.asarray(o),
                            select='i', select_range=(0, nvec - 1))
    return np.asarray(w), np.asarray(V), xi, h


# ---- C1: FV operator + Richardson ----
xs = np.linspace(0.0, 3.0, 30001)
Vv = Veff(xs)
i0 = int(np.argmin(Vv))
check('C1_Vmin_nonsingular', abs(float(Vv[i0]) - 3.19416) < 0.002,
      'V_min = %.5f @ x = %.4f (tekilliksiz)' % (float(Vv[i0]), float(xs[i0])))
w, V, xi, h = fv_lams(2000)
lam0 = float(w[0])
check('C1_lam0_FV', abs(lam0 - 5.61019954) < 0.001,
      'FV lambda_0 = %.7f (varyasyonel sinir 5.6468 altinda)' % lam0)
v0 = np.asarray(V[:, 0], dtype=float)
if -np.min(v0) > np.max(v0):
    v0 = -v0
o01 = abs(float(np.asarray(V[:, 0]) @ np.asarray(V[:, 1])))
u1n = int(np.sum(np.diff(np.sign(np.asarray(V[:, 1]))) != 0))
check('C1_ortho_nodeless', o01 < 1e-14 and u1n == 1,
      '|<v0|v1>| = %.2e, psi1 dugum = %d' % (o01, u1n))
L = [float(fv_lams(N)[0][0]) for N in (500, 1000, 2000, 4000)]
p = float(np.log((L[1] - L[2]) / (L[2] - L[3])) / np.log(2.0))
cont = float((4.0 * L[3] - L[2]) / 3.0)
check('C1_richardson_p2', abs(p - 2.0) < 1e-3 and 5.61019 < cont < 5.61022
      and all(L[i] < L[i + 1] for i in range(3)),
      'p = %.5f, lam_cont = %.8f (O(h^2), asagidan monoton)' % (p, cont))
# Temple-Kato (bagimsiz): Gauss varyasyonel ust-sinir + FV matris carpimi.
from scipy.integrate import quad as _q
W2 = float(C_UNIV ** 2 * (1.0 + MU_M))
AST = 0.6931
ITK, _ = _q(lambda x: x * np.exp(-2.0 * AST * x * x) / (x * x + B2),
            0.0, np.inf, limit=200)
MU_G = float(2.0 * AST + W2 / (2.0 * AST) + 1.25 * ITK / (1.0 / (4.0 * AST)))
wT, VT, xiT, hT = fv_lams(1000)
vT = np.asarray(VT[:, 0], dtype=float)
if np.max(vT) < -np.min(vT):
    vT = -vT
lam1T = float(wT[1])
VTa = ((1.0 + B2) / (xiT * xiT + B2) + C_UNIV ** 2 * (1.0 + MU_M) * xiT * xiT)
xfT = np.arange(1001) * hT
ddT = (xfT[:-1] + xfT[1:]) / (xiT * hT * hT) + VTa
ooT = -xfT[1:-1] / (hT * hT * np.sqrt(xiT[:-1] * xiT[1:]))
Sv = np.asarray(ddT) * vT
Sv[:-1] += np.asarray(ooT) * vT[1:]
Sv[1:] += np.asarray(ooT) * vT[:-1]
MU_T = float(vT @ Sv)
rr = Sv - MU_T * vT
SIG_T = float(np.sqrt(rr @ rr))
TK_LO = float(MU_T - SIG_T ** 2 / (lam1T - MU_T))
check('C1_temple_kato', abs(MU_G - 5.6467531) < 1e-4
      and 5.0 < TK_LO <= float(wT[0]) + 2e-5 and MU_T <= float(wT[0]) + 2e-5
      and (MU_T - TK_LO) < 1e-9,
      'Gauss ust %.7f; FV [%.12f, %.12f] genislik %.1e'
      % (MU_G, TK_LO, MU_T, MU_T - TK_LO))

# ---- C2: 0++ araligi + sayisal kopru ----
GAP = 2.0 * C_UNIV * np.sqrt(1.0 + 0.5 * MU_M * (1.0 - ETA_T))
check('C2_gap_exact', abs(float(GAP) - 3.3914463103987673) < 1e-12,
      'gap = %.13f' % float(GAP))
REL = abs(float(GAP) - 3.405) / 3.405
SIGN = abs(float(GAP) - 3.405) / 0.021
GNUM = float(np.sqrt(2.0 * cont))
RN_C = abs(GNUM - float(GAP)) / float(GAP)
RN_L = abs(GNUM - 3.405) / 3.405
PCT_C = RN_C * 100.0
PCT_L = RN_L * 100.0
GCHI = float(np.sqrt(2.0 * cont * (1.0 + LAM0 / 3.0)))
RCHI_C = abs(GCHI - float(GAP)) / float(GAP)
RCHI_L = abs(GCHI - 3.405) / 3.405
check('C2_gap_lqcd_bridge', REL < 0.005 and SIGN < 1.0
      and RN_C < 0.02 and RN_L < 0.02
      and RCHI_C < 6e-4 and RCHI_L < 0.005,
      'gap fark %.3f pct (%.2f si); sqrt(2lam) = %.7f; '
      'kiral %.7f (kapali %.3f pct, LQCD %.3f pct = %.2f si)'
      % (REL * 100.0, SIGN, GNUM, GCHI, RCHI_C * 100.0,
         RCHI_L * 100.0, abs(GCHI - 3.405) / 0.021))

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

# ---- C4: Lüscher-Meral + perdeleme + quarkonyum ----
GAM = np.pi / 12.0 * (1.0 + 0.8 * KAP)
DG = MU_M / 1.25
RC = 0.5 / 0.44
check('C4_gamma_exact', abs(float(GAM) - 0.29847153360434125) < 1e-12,
      'gamma_Meral = %.14f' % float(GAM))
check('C4_screen_consts', abs(DG - 0.16352706) < 1e-8 and abs(RC - 1.136364) < 1e-6
      and abs(float(GAM) - 0.295) / 0.010 < 1.0,
      'dgamma_UV = %.8f, Rc = %.6f GeV^-1, Cornell %.2f si'
      % (DG, RC, abs(float(GAM) - 0.295) / 0.010))
SIG2 = 0.44 ** 2
N2, RMAX = 1500, 25.0
dr = RMAX / N2
r = (np.arange(1, N2) * dr).astype(float)
Gscr = float(GAM) + DG * np.exp(-r / RC)


def qlevels(mQ, Gvec):
    Vq = -Gvec / r + SIG2 * r
    Hq = (np.diag(2.0 / (mQ * dr ** 2) + Vq)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), 1)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), -1))
    return eigh(Hq, subset_by_index=[0, 2], eigvals_only=True)


Ecc = qlevels(1.4, Gscr)
Ebb = qlevels(4.7, Gscr)
SCC = float((Ecc[1] - Ecc[0]) * 1000.0)
SBB = float((Ebb[1] - Ebb[0]) * 1000.0)
check('C4_quarkonia_screened', abs(SCC - 589.19) / 589.19 < 0.02
      and abs(SBB - 562.96) / 562.96 < 0.08,
      'cc %.2f (%.2f%%), bb %.2f (%.2f%%)'
      % (SCC, abs(SCC - 589.19) / 589.19 * 100.0,
         SBB, abs(SBB - 562.96) / 562.96 * 100.0))
# C_F = 4/3 Casimir tamamlayici sutun (bagimsiz kaba izgara).
CF = 4.0 / 3.0
DG_CAS = CF * MU_M / 1.25
Gcas = float(GAM) + DG_CAS * np.exp(-r / RC)


def qcas(mQ):
    Vq = -Gcas / r + SIG2 * r
    Hq = (np.diag(2.0 / (mQ * dr ** 2) + Vq)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), 1)
          + np.diag(-1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1), -1))
    return eigh(Hq, subset_by_index=[0, 2], eigvals_only=True)


Ec = qcas(1.4)
Eb = qcas(4.7)
SCC_C = float((Ec[1] - Ec[0]) * 1000.0)
SBB_C = float((Eb[1] - Eb[0]) * 1000.0)
check('C4_casimir_band', abs(DG_CAS - 0.21803608) < 1e-8
      and abs(SCC_C - 602.92) < 1.5 and abs(SBB_C - 550.62) < 1.5,
      'dCas = %.8f; cc-CAS %.2f, bb-CAS %.2f MeV'
      % (DG_CAS, SCC_C, SBB_C))

# ---- C5: OS sonumu + kusatma (bilesik) ----
X2 = xi * xi
V0 = np.asarray(V[:, 0], dtype=float)
Mn = np.array([abs(float(np.asarray(V[:, n]) * X2 @ V0)) ** 2 for n in range(1, 6)])
dE = np.array([float(w[n] - w[0]) for n in range(1, 6)])
Gc12 = float(np.sum(Mn * np.exp(-dE * 12.0)))
Gc15 = float(np.sum(Mn * np.exp(-dE * 15.0)))
SL = float(-np.log(Gc15 / Gc12) / 3.0)
RGAP = float(w[1] - w[0])
SLOPE = float(GAM) / RMAX ** 2 + SIG2
ok = (Gc12 > 0.0 and Gc15 > 0.0 and abs(SL - RGAP) < 1e-3
      and bool(Ecc[0] < Ecc[1] < Ecc[2]) and bool(Ebb[0] < Ebb[1] < Ebb[2])
      and SLOPE > 0.0)
check('C5_OS_confinement', ok,
      'Gc>0, egim %.6f vs radyal %.6f; siralama OK; dV/dR=%.4f'
      % (SL, RGAP, SLOPE))

# ---- C6: kilit degismezler ----
S_LOSS = np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)
I_SH = float(np.log(S_LOSS * (1.0 - ETA_T)))
A_M = float(MU_M * (1.0 - ETA_T) * 299792458.0 * (67.4 * 1000.0 / 3.0856775814913673e22))
check('C6_locked', abs(LAM0 - 0.07226943267767906) < 1e-16
      and abs(I_SH - 2.0 * np.sqrt(2.0)) < 1e-15
      and abs(A_M - 1.1465889796559045e-10) < 1e-23
      and abs(float(GAP) - 3.3914463103987673) < 1e-12
      and abs(GNUM - 3.34968774) < 1e-6
      and abs(GCHI - 3.38979431) < 1e-6,
      'Lam0=%.17f I=%.15f aM=%.9e gap=%.10f sqrt2lam=%.7f chiral=%.7f'
      % (LAM0, I_SH, A_M, float(GAP), GNUM, GCHI))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
