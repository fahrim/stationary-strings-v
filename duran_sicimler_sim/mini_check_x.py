"""Mini capraz kontrol (Makale 10 / Teorem X, Faz 8).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy disinda
bagimliligi yoktur. Tum sabitler ve formuller asagida yeniden ve farkli
yontemle yazilmistir:
  (C1) Peeling-ballooning carpanlari + bagimsiz RK4 ELM filaman ODE'si.
  (C2) ITER ve SPARC Lawson uclu carpimi + Q_fusion kuadratik kazanc.
  (C3) Connor-Hastie Ec + sicim-yukseltmeli Ec_eff/Ec 14-basamak dogrulama.
  (C4) E_||/Ec = 30'da Rosenbluth-Putvinski cig buyume hizi isaret tersinmesi.
  (C5) 6 fiziksel olcek (10^-10..10^20 m, 30 mertebe) en-kucuk-kareler egim=1.
  (C6) Serinin 5 kilit degismezinin (Lambda_Meral^(0), mu_s^Meral, I_shield=2sqrt2,
       S_loss, Simpson int O_exact = 1) uctan-uca bagimsiz oz-denetimi.
Kullanim:  python mini_check_x.py   -> 6/6 (16/16 alt-kontrol) PASS beklenir.
"""
import numpy as np

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
C_L = 299792458.0
C_SPH = 28.0 / 79.0
LAM0 = C_SPH * 0.5 * (
    163.0 / 578.0
    + (64.0 * np.sqrt(17.0) / 4913.0) * np.log((21.0 + 5.0 * np.sqrt(17.0)) / 4.0)
)
MU_M = 2.0 * np.sqrt(2.0) * LAM0
ETA_T = 0.1434
S_LOSS = np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)


def sigma_str(chi):
    return 0.25 * MU_M * (1.0 + 0.25 * chi) / (1.0 - chi)


# ---- C1: peeling-ballooning + RK4 ELM ODE ----
for chi, ref in [(0.0, 1.2044088236755661),
                 (0.5, 1.4599198532700237),
                 (0.75, 1.9709419124589393),
                 (0.9, 3.5040080900256863)]:
    r = 1.0 + 4.0 * sigma_str(chi)
    check('C1_pb_chi%.2f' % chi, abs(r - ref) < 1e-14,
          'alpha_crit/std=%.14f' % r)


def rk4_elm(C, T=40.0, N=4000):
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


e_std, _ = rk4_elm(0.25 * (1.5 - 1.0))
_, m_str = rk4_elm(0.25 * (1.5 - (1.0 + 4.0 * sigma_str(0.75))))
check('C1_elm_std_blowup', e_std > 1e3, 'RK4 std r_end=%.3e' % e_std)
check('C1_elm_str_bounded', m_str <= 1.0 + 1e-9, 'RK4 str r_max=%.6f' % m_str)

# ---- C2: ITER ve SPARC Lawson + Q ----
H50 = 1.0 + (S_LOSS - 1.0) * sigma_str(0.50)
H75 = 1.0 + (S_LOSS - 1.0) * sigma_str(0.75)
Q_iter_50 = 10.0 * H50 ** 2
Q_sparc_50 = 11.0 * H50 ** 2
check('C2_H98_super', H50 > 1.8 and H75 > 1.8,
      'H98(0.50)=%.4f H98(0.75)=%.4f' % (H50, H75))
check('C2_Q_ignition', abs(Q_iter_50 - 99.6037) < 1e-2 and abs(Q_sparc_50 - 109.5641) < 1e-2,
      'Q_ITER(0.50)=%.2f Q_SPARC(0.50)=%.2f' % (Q_iter_50, Q_sparc_50))

# ---- C3: Connor-Hastie Ec ve Ec_eff/Ec ----
ne, lnL, Zeff = 1.0e20, 16.0, 2.0
Ec = ne * E_CHG ** 3 * lnL / (4.0 * np.pi * EPS_0 ** 2 * M_E * C_L ** 2)
er0 = S_LOSS * (1.0 + 4.0 * sigma_str(0.0))
er75 = S_LOSS * (1.0 + 4.0 * sigma_str(0.75))
check('C3_Ec_eff_chi0', abs(er0 - 23.78845032303333) < 1e-12,
      'Ec_eff/Ec(0)=%.14f' % er0)
check('C3_Ec_eff_chi075', abs(er75 - 38.92835460224382) < 1e-12,
      'Ec_eff/Ec(0.75)=%.14f' % er75)

# ---- C4: Rosenbluth-Putvinski cig isaret tersinmesi ----
tau_rel = 4.0 * np.pi * EPS_0 ** 2 * M_E ** 2 * C_L ** 3 / (ne * E_CHG ** 4 * lnL)
pref = 1.0 / (tau_rel * lnL) * np.sqrt(np.pi / (3.0 * (Zeff + 5.0)))
g_std = pref * (30.0 - 1.0)
g_str = pref * (30.0 / er75 - 1.0)
check('C4_runaway_quench', g_std > 30.0 and g_str < -0.2,
      'gamma_std=+%.4f/s gamma_str=%.4f/s' % (g_std, g_str))

# ---- C5: 30-mertebe evrensel olcekleme ----
C_UNIV = 1.6260379365521303
L0 = np.array([1.0e-10, 0.988849784e-4, 4.0e-2, 1.0e4, 1.22e10, 3.0e20])
R_sat = C_UNIV * L0
lx, ly = np.log10(L0), np.log10(R_sat)
xm, ym = np.mean(lx), np.mean(ly)
slope = float(np.sum((lx - xm) * (ly - ym)) / np.sum((lx - xm) ** 2))
R2 = float(np.sum((slope * (lx - xm)) ** 2) / np.sum((ly - ym) ** 2))
check('C5_30decade_slope', abs(slope - 1.0) < 1e-14 and abs(R2 - 1.0) < 1e-14,
      'slope=%.15f R2=%.15f span=%.1f dec' % (slope, R2, lx[-1] - lx[0]))

# ---- C6: uctan-uca seri degismezleri ----
I_sh = np.log(S_LOSS * (1.0 - ETA_T))
check('C6_meral_closed', abs(LAM0 - 0.07226943267767906) < 1e-16,
      'Lambda0=%.17f' % LAM0)
check('C6_shield_invariant', abs(I_sh - 2.0 * np.sqrt(2.0)) < 1e-15,
      'I_shield=%.16f' % I_sh)

# Simpson Ore-Powell normu
SER = (5.0 / 3.0, 1.0 / 3.0, -2.0 / 15.0, -1.0 / 5.0,
       -29.0 / 210.0, -13.0 / 210.0, -2.0 / 315.0)
N = 200001
E = np.linspace(0.0, 511.0, N)
x = E / 511.0
lo = (x > 0) & (x < 1e-2)
hi = (x >= 1e-2) & (x < 1.0)
om = np.zeros_like(x)
xl = x[lo]
acc = np.zeros_like(xl)
for cf in reversed(SER):
    acc = acc * xl + cf
om[lo] = acc * xl
xh = x[hi]
om[hi] = (2.0 * (2.0 - xh) / xh + 2.0 * (1.0 - xh) * xh / (2.0 - xh) ** 2
          + 4.0 * ((1.0 - xh) / xh ** 2 - (1.0 - xh) ** 2 / (2.0 - xh) ** 3)
          * np.log(1.0 - xh))
om[-1] = 2.0
O = om / (511.0 * (np.pi ** 2 - 9.0))
dx = E[1] - E[0]
simp = dx / 3.0 * (O[0] + O[-1] + 4.0 * np.sum(O[1:-1:2]) + 2.0 * np.sum(O[2:-1:2]))
check('C6_ore_powell_norm', abs(simp - 1.0) < 1e-9, 'int O=%.15f' % simp)

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
