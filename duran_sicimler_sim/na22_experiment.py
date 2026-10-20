"""T37/SS-L1: 22Na + silika-aerojel pozitronyum laboratuvar protokolu.

Amac: arXiv:2605.21289 Denk.(12)'nin YAYIMLANAN (ln'siz) formu ile
DUZELTILMIS 1949 Ore-Powell formunu 1 saatlik bir HPGe olcumuyle
ayirt edilebilir sekilde hesaplamak.

Uc form:
  exact      : duzeltilmis 1949 formu (ln(1-x) carpanlari yerinde)
  published : basimdaki ln'siz form (3. terimde 2m(m-E)/E^2)
  substituted: yalniz E^2 -> (2m-E)^2 ikamesi yapilmis form

Duyarlilik butcesi: 5 uCi 22Na, silika aerojel (f_Ps = 0.55),
katı aci x verim = 0.02, 3600 s, HPGe sigma = 0.85 keV.
"""
import numpy as np
from scipy.integrate import quad

M_E = 511.0          # keV
NS = 511.0           # aerojel yogunlugu (keV) -> 511 keV fotonlarinin
                     # aerojelde durmasi (tesaduf bir parametre degil, veri)
BANDS = [(10.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 350.0),
         (350.0, 490.0), (490.0, 511.0)]
A_SRC = 1.85e5        # Bq (5 uCi)
F_BETA = 0.903        # beta+ dallilma orani
F_PS = 0.55           # o-Ps olusum orani (silika aerojel)
BR_3G = 0.75          # o-Ps icinde 3-foton dallilma (1/4) -> 3/4
EPS = 0.02            # katı aci x verim
T_ACQ = 3600.0        # s
SIG_HPGE = 0.85       # keV


def _om_true(x):
    """Gercek 1949 Omega(x) (x<1e-2 Taylor, ustunde kapali form)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x < 1e-2
    xl = x[lo]
    acc = np.zeros_like(xl)
    for cf in reversed((5.0 / 3.0, 1.0 / 3.0, -2.0 / 15.0, -1.0 / 5.0,
                       -29.0 / 210.0, -13.0 / 210.0, -2.0 / 315.0)):
        acc = acc * xl + cf
    out[lo] = acc * xl
    xh = x[~lo]
    with np.errstate(divide='ignore', invalid='ignore'):
        out[~lo] = (2.0 * (2.0 - xh) / xh
                    + 2.0 * (1.0 - xh) * xh / (2.0 - xh) ** 2
                    + 4.0 * ((1.0 - xh) / xh ** 2
                             - (1.0 - xh) ** 2 / (2.0 - xh) ** 3)
                    * np.log(1.0 - xh))
    return out


def _norm():
    return M_E * (np.pi ** 2 - 9.0)


def o_exact(E):
    E = np.asarray(E, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = _om_true(E / M_E) / _norm()
    return np.where((E > 0) & (E < M_E), v, 0.0)


def o_published(E):
    """arXiv:2605.21289 Denk.(12) BASIMDAKI form (3. terimde E^2 paydasi)."""
    E = np.asarray(E, dtype=float)
    m = M_E
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (2.0 / (m * (np.pi ** 2 - 9.0))) * (
            E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
            + 2 * m * (m - E) / E ** 2
            - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3)
    return np.where((E > 0) & (E < m), v, 0.0)


def o_substituted(E):
    """Yalniz E^2 -> (2m-E)^2 ikamesi yapilmis varyant."""
    E = np.asarray(E, dtype=float)
    m = M_E
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (2.0 / (m * (np.pi ** 2 - 9.0))) * (
            E * (m - E) / (2 * m - E) ** 2 + (2 * m - E) / E
            + 2 * m * (m - E) / (2 * m - E) ** 2
            - 2 * m * (m - E) ** 2 / (2 * m - E) ** 3)
    return np.where((E > 0) & (E < m), v, 0.0)


FORMS = {'exact': o_exact, 'published': o_published, 'substituted': o_substituted}


def band_ints(fn):
    """Her bant icin normalize edilmis spektrum integrali."""
    out = []
    for lo, hi in BANDS:
        v, _ = quad(lambda E: float(fn(E)), lo, hi, limit=400)
        out.append(float(v))
    return out


def detected_counts(fn):
    """1 saatlik olcumde bant basina dedekte edilen 3-foton sayisi."""
    frac = band_ints(fn)
    n3g = A_SRC * F_BETA * F_PS * BR_3G * EPS * T_ACQ
    return [n3g * f for f in frac], n3g


def discriminant(fn):
    """R = N[10,100] / N[350,490] ve Poisson belirsizligi."""
    counts, _ = detected_counts(fn)
    lo = counts[0] + counts[1]
    hi = counts[4]
    R = lo / hi
    sig_R = R * np.sqrt(1.0 / max(lo, 1e-30) + 1.0 / max(hi, 1e-30))
    return dict(N_lo=lo, N_hi=hi, R=float(R), sig_R=float(sig_R),
                counts=counts)


def separation():
    """Uc form arasindaki ayirt edilebilirlik (sigma)."""
    d = {k: discriminant(v) for k, v in FORMS.items()}
    ex, pu, su = d['exact'], d['published'], d['substituted']
    # Yayimlanan form icin "tahmin edilen" R degeri fiziksel olarak
    # reddedilir; ayirt edicilik, olculecek R ile yayimlanan R arasindaki fark.
    out = {}
    for name, ref in (('published', pu), ('substituted', su)):
        out['exact_vs_' + name] = float(
            abs(ref['R'] - ex['R']) / ex['sig_R'])
    # Dusuk-enerji bandi sayim farki (5670x mertebesi) Poisson ile
    out['N_lo_ratio_published'] = float(
        (pu['N_lo']) / max(ex['N_lo'], 1e-30))
    out['N_lo_sigma_published'] = float(
        abs(pu['N_lo'] - ex['N_lo']) / np.sqrt(max(ex['N_lo'], 1.0)))
    return dict(per_form=d, separation=out)


def hpge_resolution_note():
    return ('HPGe sigma = %.2f keV -> 511 keV satirinda FWHM ~ %.2f keV; '
            '3-foton kontinumu 0-511 keV arasi genis, cozunurluk bandi '
            'sayimlari degil bant oranini belirler.'
            % (SIG_HPGE, 2.3548 * SIG_HPGE))


if __name__ == '__main__':
    d = separation()
    for k, v in d['per_form'].items():
        print('%-12s N_lo=%12.1f  N_hi=%12.1f  R=%.6g  +-%.3g'
              % (k, v['N_lo'], v['N_hi'], v['R'], v['sig_R']))
    print()
    for k, v in d['separation'].items():
        print('  %-28s %s' % (k, v))
    print()
    print(hpge_resolution_note())
    for k, v in FORMS.items():
        frac = band_ints(v)
        print('%-12s bant integralleri: %s' %
              (k, ' '.join('%.6g' % f for f in frac)))
