"""Mini capraz kontrol (Makale 9 / Teorem IX, Faz 7).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy/scipy disinda
bagimliligi yoktur. Tum sabitler ve formuller asagida yeniden ve farkli
yontemle yazilmistir:
  (a) n_B: dogrudan CODATA aritmetigi (farkli sabit yazimi).
  (b) Sicim-Brillouin: kapali-form cebir + kuadratik kokun denklemde yerine konmasi.
  (c) Radial ODE: solver Radau kullanir; burada klasik RK4 (farkli integrator).
  (d) Nv=7 kafes: solver minimize_scalar+FD-Hessian kullanir; burada brentq
      kritik-nokta (farkli kok-bulucu) + simetrik nefes-modu analitik egriligi.
  (e) S_loss: e^(2sqrt2)/(1-eta) 15-basamak literal karsilastirma.
  (f) Birikim ODE: solver analitik cozum kullanir; burada RK4 (farkli yol).
Kullanim:  python mini_check_ix.py   -> 6/6 PASS beklenir.
"""
import numpy as np
from scipy.optimize import brentq

PASS = []
CHECK = {}


def check(name, cond, detail):
    PASS.append(bool(cond))
    CHECK[name] = detail
    print(('PASS ' if cond else 'FAIL ') + name + ' :: ' + detail)


# ---- sabitler (bagimsiz yazim) ----
E_CHG = 1.602176634e-19
EPS_0 = 8.8541878128e-12
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
MU_M = 0.2044088236755661  # 2*sqrt(2)*0.07226943267767906 (Teorem VI-VII kilidi)
ETA_T = 0.1434

# ---- C1: klasik Brillouin sinirlari ----
for B, m, tag, ref in [(1.0, M_E, 'ep_1T', 4.858e18),
                       (1.0, M_P, 'pb_1T', 2.646e15),
                       (3.0, M_E, 'ep_3T', 4.372e19),
                       (3.0, M_P, 'pb_3T', 2.381e16)]:
    nB = EPS_0 * B ** 2 / (2.0 * m)
    check('C1_nB_' + tag, abs(nB - ref) / ref < 5e-3,
          'nB=%.6e m^-3 (beklenen ~%.3e)' % (nB, ref))

# ---- C2: Sicim-Brillouin kapali formu ----
for chi, ref in [(0.0, 1.204409), (0.5, 1.459920),
                 (0.75, 1.970942), (0.9, 3.504008)]:
    r = 1.0 + MU_M * (1.0 + 0.25 * chi) / (1.0 - chi)
    check('C2_ratio_chi%.2f' % chi, abs(r - ref) < 1e-6,
          'nmax/nB=%.10f (beklenen %.6f)' % (r, ref))
# kokler kuadratigi sagliyor mu (n = 1.5 nB, chi = 0.75)?
n_op, chi_op = 1.5, 0.75
sg = 0.25 * MU_M * (1.0 + 0.25 * chi_op) / (1.0 - chi_op)
disc = 1.0 + 4.0 * sg - n_op
wp = 0.5 * (1.0 + np.sqrt(disc))
check('C2_quad_residue', abs(wp ** 2 - wp + n_op / 4.0 - sg) < 1e-14,
      'disc=%.6f res=%.2e' % (disc, abs(wp ** 2 - wp + n_op / 4.0 - sg)))


# ---- C3: bagimsiz RK4 cift-kol ----
def rk4(C, T=40.0, N=4000):
    h = T / N
    y = np.array([1.0, 0.0])
    mx = 1.0
    for _ in range(N):
        k1 = np.array([y[1], C * y[0]])
        k2 = np.array([y[1] + h / 2 * k1[1], C * (y[0] + h / 2 * k1[0])])
        k3 = np.array([y[1] + h / 2 * k2[1], C * (y[0] + h / 2 * k2[0])])
        k4 = np.array([y[1] + h * k3[1], C * (y[0] + h * k3[0])])
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        mx = max(mx, abs(y[0]))
    return y[0], mx


C_std = 0.25 * (n_op - 1.0)
C_str = 0.25 * (n_op - 1.0 - 4.0 * sg)
e_std, m_std = rk4(C_std)
_, m_str = rk4(C_str)
check('C3_std_blowup', e_std > 1e3, 'RK4 std r_end=%.3e (patlama)' % e_std)
check('C3_str_bounded', m_str < 2.0, 'RK4 str r_max=%.4f (sinirli)' % m_str)


# ---- C4: Nv=7 brentq + analitik nefes-modu egriligi ----
def dF7(a):
    return (-21.0 / a + 6.0 * (1.0 + MU_M) * a
            + 6.0 * MU_M * a / (1.0 - a * a))


a_star = brentq(dF7, 1e-9, 0.9999, xtol=1e-15, rtol=1e-15)
F2 = (21.0 / a_star ** 2 + 6.0 * (1.0 + MU_M)
      + 6.0 * MU_M * (1.0 + a_star ** 2) / (1.0 - a_star ** 2) ** 2)
check('C4_ring_radius', abs(a_star - 0.9598051782) < 1e-10,
      'a7*=%.10f' % a_star)
check('C4_breathing_curvature', F2 > 0.0, 'F\'\'=%.3f > 0' % F2)


# ---- C5: S_loss 15 basamak ----
S = np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)
check('C5_S_loss', abs(S - 19.75114251524387) < 1e-12,
      'S=%.14f' % S)


# ---- C6: birikim ODE RK4 vs analitik ----
G0, ET = 15.0, 2.0e-3
Ninf = G0 / ET
Na = lambda x: Ninf * (1.0 - np.exp(-ET * x))
h, N, x = 0.5, 0.0, 0.0
for _ in range(int(225.0 / h)):
    k1 = G0 - ET * N
    k2 = G0 - ET * (N + h / 2 * k1)
    k3 = G0 - ET * (N + h / 2 * k2)
    k4 = G0 - ET * (N + h * k3)
    N = N + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    x = x + h
check('C6_accumulation', abs(N - Na(225.0)) < 1e-9,
      'RK4 N(225)=%.6f vs analitik %.6f' % (N, Na(225.0)))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
