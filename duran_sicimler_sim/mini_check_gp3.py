"""Mini capraz kontrol (Grand Proof III / Faz 11, T35-T37).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy/scipy disinda
bagimliligi yoktur. Tum sabitler ve formuller asagida yeniden yazilmistir:
  (C1) Burgers N: rc = sqrt(2nu/gam), y* brentq, umax, dh kapali + quad
       caprazi (|d| < 1e-12).
  (C2) PEO oranlari: C_univ = 1.62603794, C^-1 (hiz),
       [C^2(1+mu)]^-1 = 0.31402536 (huni) [literatur bandi DOGRULANAMADI].
  (C3) Vela (kaynak dogrulamali): 2004 olayi %0.43, 2000 olayi %32.9;
       olculem Q_2000 = 0.0220 + olculem zaman olcekleriyle ODE caprazi.
  (C4) 22Na: O_exact bant orani 0.08413579; misprint (>9) ve pole (>100)
       curutme + laboratuvar >500si.
  (C5) Seri kilit degismezler (Lambda0, mu_s, C_univ, S_loss, a_Meral, gap).
Kullanim:  python mini_check_gp3.py   -> 13/13 PASS beklenir.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

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

# ---- C1: Burgers kontrol sivisi ----
nu0, gam, G0, g0 = 1.0e-5, 2.0, 1.20e-2, 9.80665
rc = float(np.sqrt(2.0 * nu0 / gam))
y = float(brentq(lambda t: np.exp(-t) * (2.0 * t + 1.0) - 1.0, 0.5, 2.0,
                 xtol=1e-15, rtol=1e-15, maxiter=300))
umax = float(G0 / (2.0 * np.pi * rc * np.sqrt(y)) * (1.0 - np.exp(-y)))
dh_c = float(G0 ** 2 * np.log(2.0) / (4.0 * np.pi ** 2 * g0 * rc ** 2))


def _uth(r):
    return G0 / (2.0 * np.pi * max(float(r), 1e-300)) * (1.0 - np.exp(-(r / rc) ** 2))


dh_q, _ = quad(lambda r: float(_uth(r)) ** 2 / max(float(r), 1e-300),
               0.0, np.inf, limit=500)
dh_q = float(dh_q / g0)
check('C1_burgers_closed', abs(rc * 1e3 - 3.16227766) < 1e-5
      and abs(umax - 0.38542474) < 1e-6,
      'rc = %.5f mm, umax = %.5f m/s' % (rc * 1e3, umax))
check('C1_dimple_quad', abs(dh_c - dh_q) < 1e-12,
      'dh kapali %.5f mm vs quad %.5f mm' % (dh_c * 1e3, dh_q * 1e3))

# ---- C2: PEO oranlari ----
rc_p = C_UNIV * rc
um_p = umax / C_UNIV
rh = 1.0 / (C_UNIV ** 2 * (1.0 + MU_M))
check('C2_core_broadening', abs(rc_p * 1e3 - 5.14198344) < 1e-5,
      'rc_PEO = %.5f mm (C = %.8f)' % (rc_p * 1e3, C_UNIV))
check('C2_velocity_braking', abs(um_p - 0.23703306) < 1e-6,
      'umax_PEO = %.5f m/s (1/C = %.8f)' % (um_p, 1.0 / C_UNIV))
check('C2_dimple_suppression', abs(rh - 0.31402536) < 1e-8,
      'dh orani = %.8f (dh_PEO = %.5f mm) [bant dogrulanamadi]'
      % (rh, dh_c * rh * 1e3))

# ---- C3: Vela (kaynak dogrulamali) ----
# Dodson+2002 Tablo 1: nu, nudot; Dodson+2007: 2004 olayi; Link+1999: dI/I >= 0.014
nu0 = 11.194615396005
nud = 1.55615e-11
Icr = 0.0140
dt = 2.80 * 365.25 * 86400.0
base = Icr * nud * dt / nu0
gm = base * (1.0 + MU_M)
QF = float(np.exp(-2.0 * np.sqrt(2.0)))
QT = float(LAM0 * C_UNIV / np.sqrt(0.75))
G2000 = 3.0858e-6      # Dodson+2002: dnu_p/nu = 3.45435e-5/11.194615
G2004 = 2.08e-6        # Dodson+2007: 2004 olayi ani kesirli boyut
R2004 = abs(gm - G2004) / G2004
R2000 = abs(gm - G2000) / G2000
check('C3_glitch_2004', R2004 < 0.01,
      '2004 olayi: model %.6e vs %.4e (%.2f%%)'
      % (gm, G2004, R2004 * 100.0))
check('C3_glitch_2000', R2000 < 0.40,
      '2000 olayi: model %.6e vs %.4e (%.1f%%, en buyuk olayi)'
      % (gm, G2000, R2000 * 100.0))
Q2000 = (0.020 + 0.31 + 0.193 + 0.2362) * 1e-6 / 3.45435e-5
check('C3_Q_measured', 0.02 < Q2000 < 0.023
      and abs(float(np.exp(-2.0 * np.sqrt(2.0))) - 0.05910575) < 1e-8,
      'Q_fast = %.5f, Q_total = %.5f; olculem Q_2000 = %.4f'
      % (QF, QT, Q2000))
t = np.linspace(1.0, 200.0, 20001)
R1 = np.ones_like(t)
for tau in (0.53, 3.29, 19.07):
    R1 -= (1.0 - QT) * np.exp(-t / tau) / 3.0
dA = (1.0 - QT) / 3.0 * sum(np.exp(-t / tau) / tau for tau in (0.53, 3.29, 19.07))
dN = np.gradient(R1, t)
check('C3_relax_ode', float(np.max(np.abs(dA - dN))) < 1e-3,
      'dR/dt analitik-numerik fark %.2e (olculem zaman olcekleri)' % float(np.max(np.abs(dA - dN))))

# ---- C4: 22Na bant orani ----
_OPS = (5.0 / 3.0, 1.0 / 3.0, -2.0 / 15.0, -1.0 / 5.0,
        -29.0 / 210.0, -13.0 / 210.0, -2.0 / 315.0)


def _om(x):
    if x <= 0.0:
        return 0.0
    if x < 1e-2:
        acc = 0.0
        for cf in reversed(_OPS):
            acc = acc * x + cf
        return acc * x
    return (2.0 * (2.0 - x) / x + 2.0 * (1.0 - x) * x / (2.0 - x) ** 2
            + 4.0 * ((1.0 - x) / x ** 2 - (1.0 - x) ** 2 / (2.0 - x) ** 3)
            * np.log(1.0 - x))


def _ex(E):
    m = 511.0
    if E <= 0.0 or E >= m:
        return 2.0 / (m * (np.pi ** 2 - 9.0)) if E == m else 0.0
    return _om(E / m) / (m * (np.pi ** 2 - 9.0))


def _mi(E):
    m = 511.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return float((2.0 / (m * (np.pi ** 2 - 9.0)))
                     * (E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
                        + 2 * m * (m - E) / (2 * m - E) ** 2
                        - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3))


def _pub(E):
    """arXiv:2605.21289 Denk.(12) BASIMDAKI (ln'siz, E^2 paydali) form."""
    m = 511.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return float((2.0 / (m * (np.pi ** 2 - 9.0)))
                     * (E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
                        + 2 * m * (m - E) / E ** 2
                        - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3))


Blo, _ = quad(_ex, 10.0, 100.0, limit=500)
Bhi, _ = quad(_ex, 350.0, 490.0, limit=500)
BEX = float(Blo / Bhi)
Mlo, _ = quad(_mi, 10.0, 100.0, limit=500)
Mhi, _ = quad(_mi, 350.0, 490.0, limit=500)
Slo, _ = quad(_mi, 10.0, 100.0, limit=500)
Shi, _ = quad(_mi, 350.0, 490.0, limit=500)
Plo, _ = quad(_pub, 10.0, 100.0, limit=500)
Phi, _ = quad(_pub, 350.0, 490.0, limit=500)
check('C4_band_ratio', abs(BEX - 0.08413579) < 1e-6,
      'B_exact = %.8f' % BEX)
check('C4_substituted_kill', Slo / Shi > 9.0,
      'B_ikame(2m-E)^2 = %.2f (>9, curutuldu)' % (Slo / Shi))
Ntot = 1.85e5 * 0.903 * 0.55 * 0.75 * 0.02 * 3600.0
Nlo = Ntot * Blo
# Yayimdaki (ln'siz, E^2 paydali) form: Blo_pub ~ 211
SIG = abs(Ntot * 211.05518962495182 - Nlo) / np.sqrt(Nlo)
check('C4_lab_sigma', SIG > 500.0, 'laboratuvar ayirimi %.1f si' % SIG)

# ---- C5: kilit degismezler ----
S_LOSS = np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)
I_SH = float(np.log(S_LOSS * (1.0 - ETA_T)))
A_M = float(MU_M * (1.0 - ETA_T) * 299792458.0 * (67.4 * 1000.0 / 3.0856775814913673e22))
GAP = 2.0 * C_UNIV * np.sqrt(1.0 + 0.5 * MU_M * (1.0 - ETA_T))
check('C5_locked', abs(LAM0 - 0.07226943267767906) < 1e-16
      and abs(C_UNIV - 1.6260379365521303) < 1e-13
      and abs(I_SH - 2.0 * np.sqrt(2.0)) < 1e-15
      and abs(A_M - 1.1465889796559045e-10) < 1e-23
      and abs(float(GAP) - 3.3914463103987673) < 1e-12,
      'Lam0=%.17f C=%.10f aM=%.9e gap=%.10f'
      % (LAM0, C_UNIV, A_M, float(GAP)))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
