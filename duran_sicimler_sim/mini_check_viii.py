"""Mini capraz kontrol (Makale 8 / Teorem VIII, Faz 6).

BAGIMSIZLIK: bu dosya solver_engine.py'yi import ETMEZ; numpy/scipy disinda
bagimliligi yoktur. Tum sabitler ve formuller asagida yeniden ve farkli
yontemle yazilmistir:
  (a) Breit-Rabi: solver Kronecker+ozvektor etiketler; burada elde-yazilmis
      uncoupled-basis matrisi + analitik mF atamasi (gerilmis durumlar baz
      durumudur, orta 2x2 analitik).
  (b) O_exact integrali: solver quad kullanir; burada Simpson (farkli kuadratur).
  (c) T19 zarf dengesi: solver Radau ODE cozer; burada brentq sabit-noktasi
      (cebirsel) + kaba explicit-Euler (farkli integrator).
  (d) rp_ideal: dogrudan aritmetik, farkli sabit yazimi.
Kullanim:  python mini_check_viii.py   -> 4/4 PASS beklenir.
"""
import numpy as np
from scipy.optimize import brentq

PASS = []
CHECK = {}


def check(name, cond, detail):
    PASS.append(bool(cond))
    CHECK[name] = detail
    print(('PASS ' if cond else 'FAIL ') + name + ' :: ' + detail)


# ---- (a) Breit-Rabi, bagimsiz yol ----
A_HZ = 1420404.8e3
MUB = 13.996245042e9
MUN = 7.622593285e6  # Hz/T -- DOGRU deger (proton 42.58 MHz/T)
GJ, GI = 2.00231930436256, 5.5856946893


def fda_indep(B):
    A = A_HZ
    ze, zp = GJ * MUB * B, GI * MUN * B
    E_uu = A / 4 + (ze - zp) / 2      # mF=+1 -> |d>
    E_dd = A / 4 - (ze - zp) / 2      # mF=-1 -> |b>
    M = np.array([[-A / 4 + (ze + zp) / 2, A / 2],
                  [A / 2, -A / 4 - (ze + zp) / 2]])
    Ea, Ec = sorted(np.linalg.eigvalsh(M))
    return (E_uu - Ea) / 1e3  # kHz


D_REF = {1.03: 28.018219607975333, 1.05: 28.018473553750663,
         1.07: 28.018713398240507}  # duzeltme-ONCESI solver degerleri (kayit)
for B in [1.03, 1.05, 1.07]:
    h = 1e-4
    d = (fda_indep(B + h) - fda_indep(B - h)) / (2 * h) / 1e6  # GHz/T
    check('breit_dfdB_%.2fT' % B, abs(d - 28.008) < 0.002,
          'dfdB=%.6f GHz/T (beklenen ~28.008, eski-hatali %.6f)' % (d, D_REF[B]))

# ---- (b) O_exact Simpson integrali ----
SER = (5.0 / 3.0, 1.0 / 3.0, -2.0 / 15.0, -1.0 / 5.0,
       -29.0 / 210.0, -13.0 / 210.0, -2.0 / 315.0)


def omega(x):
    x = np.asarray(x, dtype=float)
    lo = x < 1e-2
    out = np.empty_like(x)
    xl = x[lo]
    acc = np.zeros_like(xl)
    for cf in reversed(SER):
        acc = acc * xl + cf
    out[lo] = acc * xl
    xh = x[~lo]
    out[~lo] = (2.0 * (2.0 - xh) / xh + 2.0 * (1.0 - xh) * xh / (2.0 - xh) ** 2
                + 4.0 * ((1.0 - xh) / xh ** 2 - (1.0 - xh) ** 2 / (2.0 - xh) ** 3)
                * np.log(1.0 - xh))
    return out


m = 511.0
N = 200001
E = np.linspace(0.0, m, N)
O = np.where((E > 0) & (E < m), omega(E / m) / (m * (np.pi ** 2 - 9.0)), 0.0)
O[-1] = 2.0 / (m * (np.pi ** 2 - 9.0))  # x -> 1 limiti (Omega -> 2)
dx = E[1] - E[0]
simp = dx / 3.0 * (O[0] + O[-1] + 4.0 * np.sum(O[1:-1:2]) + 2.0 * np.sum(O[2:-1:2]))
check('ore_simpson_norm', abs(simp - 1.0) < 1e-9,
      'Simpson integrali=%.15f (|1-I|=%.2e)' % (simp, abs(simp - 1.0)))

# ---- (c) T19 zarf: explicit-Euler + brentq sabit-noktasi ----
e, mp = 1.602176634e-19, 1.67262192369e-27
nuz = 453e3 * np.sqrt(16.0 / 3.911)
gdisp = 2.0 * np.pi * 122.0 * (122.0 / nuz)
a_rms = 0.15 * 9.80665
R, req, mu_s, gs0 = 5e-3, 0.5e-3, 0.20317683698063987, 1.0
wc = e * 0.993 / mp


def rhs(r):
    den = max(1.0 - (r / R) ** 2, 1e-6)
    return gdisp * r - mu_s * gs0 * r * (1.0 + (r / req) ** 2) / den + a_rms / wc


r_star = brentq(rhs, 1e-12, R * 0.9999, xtol=1e-15, rtol=1e-15)
r = 0.0
dt = 0.5
for _ in range(int(1440.0 / dt)):
    r = r + dt * rhs(r)
    if r < 0:
        r = 0.0
check('envelope_euler_vs_root', abs(r - r_star) / r_star < 1e-3,
      'Euler r_end=%.6f um vs brentq r*=%.6f um' % (r * 1e6, r_star * 1e6))
check('envelope_below_wall', r_star < 0.1 * R,
      'r*=%.6f um << 0.5 mm' % (r_star * 1e6))

# ---- (d) rp_ideal dogrudan ----
ECHG, EPS0 = 1.602176634e-19, 8.8541878128e-12
rp = float(np.sqrt(ECHG * 1.28e5 / (4.0 * np.pi ** 2 * EPS0 * 1.0 * 0.04 * 150e3)))
check('rp_ideal_direct', abs(rp * 1e3 - 0.0988849784) < 1e-9,
      'rp_ideal=%.10f mm' % (rp * 1e3))

print('MINI %d/%d %s' % (sum(PASS), len(PASS), 'PASS' if all(PASS) else 'FAIL'))
