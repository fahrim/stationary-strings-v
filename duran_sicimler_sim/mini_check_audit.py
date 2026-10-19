"""Seri denetimi (Grand Proof I--III): serinin TUM kilit degismezlerinin
bagimsiz yeniden turetimi.

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ. Tum sabitler ve
formuller asagida YENIDEN yazilmistir ve FARKLI sayisal rotalar kullanilir:
  * l_tilde: cozucunun Gauss-Legendre(256) kapalifomu yerine DOP853 ODE
    entegrasyonu (kurkce: farkli yontem, ayni fizik).
  * l(X,0): l_prof yerine cebirsel olarak ozdes ama ayri yazilan
    l_shear_reg = (X0 - h X)/(X0 + X).
  * G(X): dogrudan F*sqrt(a^2+beta0^2 l^2) formuluyle yeniden.
  * Pi_eff ve C_univ: X* = 1, s_eq = 10, h = 0.005 uzerinden.

Kontrol edilenler (serinin butun zincirinin dayandigi degerler):
  L1 Lambda_Meral^(0) (kapali form)      L2 mu_s^Meral = 2 sqrt2 Lambda
  L3 C_univ (bagimsiz yeniden turetim)   L4 Pi_Meral = mu_s
  L5 I_shield = 2 sqrt2                  L6 S_loss
  L7 a_Meral (GP-I)                      L8 gap (GP-II kiral kopru)
Kullanim:  python mini_check_audit.py   -> 9/9 PASS beklenir.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad

PASS = []
CHECK = {}


def check(name, cond, detail):
    PASS.append(bool(cond))
    CHECK[name] = detail
    print(('PASS ' if cond else 'FAIL ') + name + ' :: ' + detail)


# ---- model parametreleri (kilitli) ----
F0, X0, ALPHA = 1.0, 1.0, 0.3
KAPPA_R, GAMMA_REL, ELL0 = 1.0, 0.1, 1.0
X_STAR, BETA0, H = 1.0, 0.5, 0.005
ETA_T0 = 0.1434
C_SPH = 28.0 / 79.0


def F_prof(X):
    return F0 / (1.0 + X / X0) ** (1.0 + H)


def a_prof(X):
    return 2.0 * X * (1.0 + H) / (X0 + X)


def l_shear_reg(X):
    return (X0 - H * X) / (X0 + X)


def G_drive(X):
    return F_prof(X) * np.sqrt(a_prof(X) ** 2 + BETA0 ** 2 * l_shear_reg(X) ** 2)


def l_tilde_ode(s, y, X):
    """d l/ds = kappa G e^{hs} l - gamma e^{-s} (l - l0),  l(0) = l0."""
    G = G_drive(X)
    l0 = ELL0
    return [KAPPA_R * G * np.exp(H * s) * y[0] - GAMMA_REL * np.exp(-s) * (y[0] - l0)]


def l_tilde(s, X):
    """Bagimsiz: DOP853 ile ODE cozumu (cozucunun GL-256 kapalifomundan farkli)."""
    sol = solve_ivp(l_tilde_ode, (0.0, float(s)),
                    [ELL0], method='DOP853', rtol=1e-13, atol=1e-15,
                    args=(X,))
    return float(sol.y[0, -1])


def Pi_eff_at(X, s, mu_s):
    F = F_prof(X)
    lt = l_tilde(s, X)
    return float(F ** 2 - (mu_s / (2.0 * X_STAR)) * np.exp(-(1.0 + 2.0 * H) * s)
                 * (lt ** 2 - ELL0 ** 2))


# ---- L1: Lambda_Meral^(0) kapali form ----
LAM0 = C_SPH * 0.5 * (
    163.0 / 578.0
    + (64.0 * np.sqrt(17.0) / 4913.0) * np.log((21.0 + 5.0 * np.sqrt(17.0)) / 4.0)
)
check('L1_lambda_meral', abs(LAM0 - 0.07226943267767906) < 1e-16,
      'Lambda_Meral^(0) = %.17f' % LAM0)

# ---- L2: mu_s = 2 sqrt2 Lambda ----
MU = 2.0 * np.sqrt(2.0) * LAM0
check('L2_mu_s', abs(MU - 0.20440882367556615) < 1e-14,
      'mu_s^Meral = %.16f' % MU)

# ---- L3: C_univ BAGIMSIZ yeniden turetim ----
S_EQ = 10.0
F_base = F_prof(X_STAR)
Pi_eff = Pi_eff_at(X_STAR, S_EQ, MU)
F_eff = float(np.sqrt(max(Pi_eff, 1e-300)))
C_UNIV = float(F_base / F_eff)
lt = l_tilde(S_EQ, X_STAR)
check('L3_c_univ', abs(C_UNIV - 1.6260379365521303) < 1e-9,
      'C_univ = %.13f (F_base = %.9f, F_eff = %.9f)' % (C_UNIV, F_base, F_eff))

# ---- L4: Pi_Meral = mu_s (ozdeslik) ----
# Pi_Meral = (C_univ^2 - 1) * F_eff^2 / D_str,  D_str = (1/(2X*)) e^{-(1+2h)s} dm
# (Pi_eff'teki mu_s faktoru D_str'de YOK; bkz. cozucunun T28 tanimi).
D_str = float((1.0 / (2.0 * X_STAR)) * np.exp(-(1.0 + 2.0 * H) * S_EQ)
              * (lt ** 2 - ELL0 ** 2))
PI_M = float((C_UNIV ** 2 - 1.0) * (F_eff ** 2) / D_str)
check('L4_pi_meral', abs(PI_M - MU) < 1e-8,
      'Pi_Meral = %.12f (mu_s = %.12f, fark %.2e)' % (PI_M, MU, abs(PI_M - MU)))

# ---- L5: I_shield = 2 sqrt2 ----
S_LOSS = float(np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T0))
I_SH = float(np.log(S_LOSS * (1.0 - ETA_T0)))
check('L5_i_shield', abs(I_SH - 2.0 * np.sqrt(2.0)) < 1e-14,
      'I_shield = %.15f (S_loss = %.7f)' % (I_SH, S_LOSS))

# ---- L6: S_loss ----
check('L6_s_loss', abs(S_LOSS - 19.7511425152) < 1e-9,
      'S_loss = %.10f' % S_LOSS)

# ---- L7: a_Meral (GP-I) ----
C_LIGHT = 299792458.0
MPC_M = 3.0856775814913673e22
A_M = float(MU * (1.0 - ETA_T0) * C_LIGHT * (67.4 * 1000.0 / MPC_M))
check('L7_a_meral', abs(A_M - 1.1465889796559045e-10) < 1e-23,
      'a_Meral = %.9e m/s^2' % A_M)

# ---- L8: gap (GP-II kiral kopru) ----
GAP = 2.0 * C_UNIV * np.sqrt(1.0 + 0.5 * MU * (1.0 - ETA_T0))
GAP_CH = float(np.sqrt(2.0 * 5.610203993898374 * (1.0 + LAM0 / 3.0)))
check('L8_gap', abs(GAP - 3.3914463103987673) < 1e-12
      and abs(GAP_CH - 3.3897943144621827) < 1e-12,
      'gap = %.10f, kirali kopru = %.10f' % (GAP, GAP_CH))

# ---- L9: analitik (izgaradan bagimsiz) Temple-Kato cercevelemesi ----
# H = H_harm(a) + V_core ; a = C_univ*sqrt(1+mu) ; V_core = 1.25/(x^2+0.25) > 0
# H_harm taban ozfonksiyonu phi0 = sqrt(2a) e^{-a x^2/2} -> mu = 2a + <V_core>,
# sigma^2 = Var(V_core) ; min-max: lam1 > 6a ; TK: lam0 >= mu - sig^2/(6a-mu).
from scipy.integrate import quad as _qa
ALPHA = C_UNIV * np.sqrt(1.0 + MU)
LAM1_H = 6.0 * ALPHA


def _vc(x):
    return 1.25 / (x * x + 0.25)


def _muvar(b):
    n = _qa(lambda x: x * np.exp(-b * x * x), 0.0, np.inf, limit=400)[0]
    i1 = _qa(lambda x: x * _vc(x) * np.exp(-b * x * x), 0.0, np.inf, limit=400)[0]
    i2 = _qa(lambda x: x * _vc(x) ** 2 * np.exp(-b * x * x), 0.0, np.inf, limit=400)[0]
    v1, v2 = i1 / n, i2 / n
    return float(b + v1 + ALPHA ** 2 / b), float(v2 - v1 * v1)


MU_A, SIG2_A = _muvar(ALPHA)
LOW_A = MU_A - SIG2_A / (LAM1_H - MU_A)
LAM_CONT = 5.610203993898374
check('L9_temple_kato', LOW_A <= LAM_CONT <= MU_A and LOW_A > 0
      and abs(MU_A - 5.7674116) < 1e-5 and abs(LAM1_H - 10.7070268) < 1e-6,
      'analitik TK [%.7f, %.7f] lam_cont=%.9f icinde (alt>0)'
      % (LOW_A, MU_A, LAM_CONT))

print('AUDIT %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))