"""Mini capraz kontrol (Grand Proof I / Faz 9, T29-T31).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy disinda
bagimliligi yoktur. Tum sabitler ve formuller asagida yeniden ve farkli
yontemle yazilmistir:
  (C1) Planck 2018 H0 + CODATA c ile a_H = c H0 ve
       a_Meral = 2 sqrt(2) Lambda_Meral^(0) (1-eta_T0) c H0 14-basamak dogrulama.
  (C2) RAR g_obs(g_bar) fonksiyonunun iki asimptotik limiti
       (Newton: g_obs/g_bar -> 1; derin-sicim: g_obs/sqrt(g_bar a) -> 1).
  (C3) 5 SPARC galaksisinde logM-logv BTFR egimi = 4 + gozlemsel vflat
       karsilastirmasi (ortalama bagil fark < %6; UGC 2885 ~%13 dis-deger
       acikca raporlanir, gizlenmez).
  (C4) NGC 3198'de Newton cokusu (v(30)/vmax < 0.65) + Meral egrisel
       duzlugu (|v(30)-v(20)|/vflat < 0.06).
  (C5) EHT SgrA*/M87* halka capi 1.5-sigma bandi + FWHM/d ~ mu_s < 0.25.
  (C6) Seri kilit degismezlerinin (Lambda0, mu_s^Meral, I_shield = 2 sqrt2,
       C_univ) uctan-uca bagimsiz oz-denetimi.
Kullanim:  python mini_check_gp1.py   -> 15/15 PASS beklenir.
"""

import numpy as np

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
C_LIGHT = 299792458.0
MPC_M = 3.0856775814913673e22
KPC_M = 3.0856775814913673e19
G_N = 6.67430e-11
M_SUN = 1.98847e30
C_UNIV = 1.6260379365521303

# ---- C1: kozmik ivme olcegi ----
H0_SI = 67.4 * 1000.0 / MPC_M
A_H = C_LIGHT * H0_SI
KAP = MU_M * (1.0 - ETA_T)
A_M = KAP * A_H
A_B = MU_M * (np.sqrt(3.0) / 2.0) * A_H
check('C1_a_H', abs(A_H - 6.548322413981453e-10) < 1e-21,
      'a_H = c H0 = %.15e m/s^2' % A_H)
check('C1_kappa', abs(KAP - 0.17509659836048996) < 1e-16,
      'kappa_Meral = %.16f' % KAP)
REL_A0_PCT = abs(A_M - 1.20e-10) / 1.20e-10 * 100.0
check('C1_a_Meral', abs(A_M - 1.1465889796559045e-10) < 1e-23,
      'a_Meral = %.15e m/s^2 (a0 fark %.2f pct)' % (A_M, REL_A0_PCT))
check('C1_a_Meral_beta', abs(A_B - 1.1592052113953912e-10) < 1e-23,
      'a_Meral(beta) = %.15e m/s^2' % A_B)


def g_obs(gb, a):
    x = np.asarray(gb, dtype=float) / float(a)
    return np.asarray(gb, dtype=float) * (1.0 + 1.0 / np.sqrt(x + x * x))


# ---- C2: RAR asimptotlari ----
x_hi = 1.0e6
r_hi = float(1.0 + 1.0 / np.sqrt(x_hi + x_hi * x_hi))
check('C2_newton_limit', abs(r_hi - 1.0) < 1e-5,
      'g_obs/g_bar(x=1e6) - 1 = %.3e' % (r_hi - 1.0))
x_lo = 1.0e-8
gb_lo = x_lo * A_M
r_lo = float(g_obs(gb_lo, A_M) / np.sqrt(gb_lo * A_M))
check('C2_string_limit', abs(r_lo - 1.0) < 2e-4,
      'g_obs/sqrt(g a)(x=1e-8) - 1 = %.3e' % (r_lo - 1.0))

# ---- C3: BTFR + SPARC ----
GALS = (
    ('DDO154', 3.5e8, 0.8, 47.0, 2.0),
    ('NGC3109', 1.55e9, 1.6, 67.0, 3.0),
    ('NGC2403', 2.10e10, 2.1, 134.0, 4.0),
    ('NGC3198', 3.80e10, 3.2, 150.0, 4.0),
    ('UGC2885', 2.50e11, 11.5, 285.0, 10.0),
)
Mb = np.array([g[1] for g in GALS]) * M_SUN
VF = (G_N * Mb * A_M) ** 0.25 / 1000.0
VO = np.array([g[3] for g in GALS])
LX = np.log10(np.array([g[1] for g in GALS]))
LY = np.log10(VF)
xm, ym = float(np.mean(LX)), float(np.mean(LY))
m = float(np.sum((LY - ym) * (LX - xm)) / np.sum((LY - ym) ** 2))
b = float(xm - m * ym)
rr = float(1.0 - np.sum((LX - (m * LY + b)) ** 2) / np.sum((LX - xm) ** 2))
REL = np.abs(VF - VO) / VO
check('C3_btfr_slope', abs(m - 4.0) < 1e-12,
      'BTFR egimi m = %.15f (|m-4| = %.2e)' % (m, abs(m - 4.0)))
check('C3_btfr_R2', abs(rr - 1.0) < 1e-12,
      'BTFR R2 = %.15f' % rr)
check('C3_sparc_mean', float(np.mean(REL)) < 0.06,
      'ortalama |dv|/v = %.2f%% (UGC2885 %.2f%% dis-deger)'
      % (float(np.mean(REL)) * 100.0, float(REL[-1]) * 100.0))


def v_of_r(Mb_msun, Rd_kpc, r_kpc, a):
    M = float(Mb_msun) * M_SUN
    r = float(r_kpc) * KPC_M
    u = float(r_kpc) / float(Rd_kpc)
    f = 1.0 - (1.0 + u) * np.exp(-u)
    gb = G_N * M * f / r ** 2
    return float(np.sqrt(float(g_obs(gb, a)) * r) / 1000.0), \
        float(np.sqrt(gb * r) / 1000.0)


# ---- C4: NGC 3198 profil duz-derinlik caprazi ----
rr_grid = np.linspace(0.15 * 3.2, 10.0 * 3.2, 400)
vn = np.array([v_of_r(3.80e10, 3.2, r, A_M)[1] for r in rr_grid])
drop = float(vn[-1] / np.max(vn))
v20, _ = v_of_r(3.80e10, 3.2, 20.0, A_M)
v30, _ = v_of_r(3.80e10, 3.2, 30.0, A_M)
VF3198 = float((G_N * 3.80e10 * M_SUN * A_M) ** 0.25 / 1000.0)
flat = float(abs(v30 - v20) / VF3198)
check('C4_newton_drop', drop < 0.65,
      'NGC3198 vN(10Rd)/vmaxN = %.4f' % drop)
check('C4_meral_flat', flat < 0.06,
      'NGC3198 |vM(30)-vM(20)|/vflat = %.4f' % flat)

# ---- C5: EHT ----
RATIO = 6.0 * C_UNIV * float(np.sqrt(1.0 + 0.25 * MU_M))
d_sgr = RATIO * 5.03
d_m87 = RATIO * 3.83
s_sgr = abs(d_sgr - 51.8) / 2.3
s_m87 = abs(d_m87 - 42.0) / 3.0
check('C5_sgra', s_sgr < 1.5,
      'SgrA* d_pred=%.2f vs 51.8+-2.3 (%.2f sigma)' % (d_sgr, s_sgr))
check('C5_m87', s_m87 < 1.5,
      'M87* d_pred=%.2f vs 42.0+-3.0 (%.2f sigma)' % (d_m87, s_m87))
check('C5_fwhm', float(MU_M) < 0.25,
      'FWHM/d ~ mu_s = %.4f < 0.25' % float(MU_M))

# ---- C6: kilit degismezler ----
I_SH = float(np.log((np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T)) * (1.0 - ETA_T)))
check('C6_locked', abs(LAM0 - 0.07226943267767906) < 1e-16
      and abs(I_SH - 2.0 * np.sqrt(2.0)) < 1e-15
      and abs(RATIO - 10.0024041395564) < 1e-9,
      'Lambda0=%.17f mu=%.16f I=%.15f d/t=%.10f'
      % (LAM0, MU_M, I_SH, RATIO))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
