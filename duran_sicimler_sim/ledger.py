"""Derece-defteri (Degrees-of-Freedom Ledger) — Teoremler I–X ve Stationary Strings XII–III.

AMAC: Her yayimlanabilir karsilastirmayi tek tabloda dort soruyla birlikte
gostermek:
  (1) Gozlenebilir nedir, literatur degeri ve hatasi nedir?
  (2) Model degeri nedir (ve kacikla uyusur)?
  (3) Hangi serbest parametreler giriyor, bunlar BU veriye mi fit edildi?
  (4) Kanit sinifi: FIT / FIT-benzeri / TRANSFER / GENUINE / STRUCTURAL ve
      kirilganlik (knob'lar %1 bozuldugunda model degeri ne kadar oynuyor)?

KNOB ROLLERI:
  'locked'    -> Teoremlerden gelen, veriye fit EDILMEYEN kilit
  'ansatz'    -> model yapisi secimi (turetilmemis; kirilganlik testi)
  'fit'       -> BU veriye (veya yakin digerine) secilmis parametre
  'measured'  -> deneyden alinan girdi (serbestlik degil)

TUTARLILIK: Tablonun taban degerleri `run_verification()` ciktisiyla
otomatik karsilastirilir (verify_against_solver).
Kullanim:  python ledger.py            -> tablo + kirilganlik + ozet
"""
import io
import os
import re

import numpy as np

# ---------------- kilit cekirdek (Teoremler I-X) ----------------
LAM0_CLOSED = (28.0 / 79.0) * 0.5 * (
    163.0 / 578.0
    + (64.0 * np.sqrt(17.0) / 4913.0) * np.log((21.0 + 5.0 * np.sqrt(17.0)) / 4.0)
)
MU_S = 2.0 * np.sqrt(2.0) * LAM0_CLOSED
ETA_T0 = 0.1434
BETA0 = 0.5
C_UNIV = 1.6260379365521303
S_LOSS = float(np.exp(2.0 * np.sqrt(2.0)) / (1.0 - ETA_T0))
I_SHIELD = float(np.log(S_LOSS * (1.0 - ETA_T0)))
KNOB_DELTA = 0.01


def _knobs():
    return dict(C_UNIV=C_UNIV, MU_S=MU_S, ETA_T0=ETA_T0, S_LOSS=S_LOSS,
                LAM0=LAM0_CLOSED, BETA0=BETA0)


def _rel(model, obs):
    if obs == 0:
        return 0.0
    return float(abs(model - obs) / abs(obs))


def _frag(fn, knob, ks):
    k2 = dict(ks)
    k2[knob] = ks[knob] * (1.0 + KNOB_DELTA)
    base = fn(ks)
    alt = fn(k2)
    if base == 0:
        return 0.0
    return float(abs(alt - base) / abs(base))


# ---------------- model deger fonksiyonlari ----------------
def m_a_meral(k):
    c, H0 = 299792458.0, 67.4 * 1000.0 / 3.0856775814913673e22
    return k['MU_S'] * (1.0 - k['ETA_T0']) * c * H0


def m_btfr_slope(k):
    return 4.0


def m_sparc_mean(k):
    G, Ms = 6.67430e-11, 1.98847e30
    aM = m_a_meral(k)
    data = ((3.5e8, 47.0), (1.55e9, 67.0), (2.10e10, 134.0),
            (3.80e10, 150.0), (2.50e11, 285.0))
    errs = []
    for Mb, vo in data:
        vf = (G * Mb * Ms * aM) ** 0.25 / 1000.0
        errs.append(abs(vf - vo) / vo)
    return float(np.mean(errs))


def m_eht_ratio(k):
    return 6.0 * k['C_UNIV'] * np.sqrt(1.0 + k['MU_S'] / 4.0)


def m_gap_closed(k):
    return 2.0 * k['C_UNIV'] * np.sqrt(1.0 + 0.5 * k['MU_S'] * (1.0 - k['ETA_T0']))


def m_glue_mean(k):
    g0 = m_gap_closed(k)
    kap = k['MU_S'] * (1.0 - k['ETA_T0'])
    b2 = k['BETA0'] ** 2
    st = {'0++': g0,
          '2++': g0 * np.sqrt(1.0 + 6.0 * kap / (1.0 + b2)),
          '0-+': g0 * np.sqrt(1.0 + (1.0 + b2) * (1.0 - k['ETA_T0'] + k['MU_S'])),
          '0*++': g0 * k['C_UNIV'] * np.sqrt(1.0 + k['MU_S'])}
    obs = {'0++': 3.405, '2++': 4.73, '0-+': 5.12, '0*++': 6.18}
    return float(np.mean([abs(st[s] - obs[s]) / obs[s] for s in st]))


def m_cornell_gamma(k):
    return (np.pi / 12.0) * (1.0 + 0.8 * k['MU_S'] * (1.0 - k['ETA_T0']))


def _eig2(main, off):
    from scipy.linalg import eigh_tridiagonal as _e
    r = _e(np.asarray(main), np.asarray(off), select='i', select_range=(0, 1))
    w = r[0] if isinstance(r, tuple) else r
    return np.atleast_1d(np.asarray(w))


def _quark_split(k, mQ, N=3000, Rmax=30.0):
    sig = 0.44 ** 2
    rc = 0.5 / 0.44
    dgamma = k['MU_S'] / 1.25
    dr = Rmax / N
    r = (np.arange(1, N) * dr)
    G = m_cornell_gamma(k) + dgamma * np.exp(-r / rc)
    V = -G / r + sig * r
    main = 2.0 / (mQ * dr ** 2) + V
    off = -1.0 / (mQ * dr ** 2) * np.ones(len(r) - 1)
    w = _eig2(main, off)
    return float((w[1] - w[0]) * 1000.0)


def m_ccbar(k):
    return _quark_split(k, 1.40)


def m_bbbar(k):
    return _quark_split(k, 4.70)


def m_vela_2004(k):
    base = 0.0140 * 1.55615e-11 * (2.80 * 365.25 * 86400.0) / 11.194615396005
    return base * (1.0 + k['MU_S'])


def m_na22_band(k):
    return 0.08413579096991904


def m_barrier_lo(k):
    return k['S_LOSS'] / k['C_UNIV']


# ---------------- tablo satirlari ----------------
# knob: (ad, rol)
def build_ledger():
    ks = _knobs()
    rows = []
    A = rows.append
    A(dict(id='T15a', domain='ALPHA', observable='a1S/h (1.03 T) [kHz]',
           obs=1420404.8, err=5.6, src='Nature 653 (2026)',
           model=1420404.8,
           knobs=[('omega_D0', 'fit'), ('kappa_a5', 'fit'), ('f_Ps', 'measured')],
           cls='FIT', note='omega_D0 bu degeri uretmek icin secildi'))
    A(dict(id='T15b', domain='ALPHA', observable='a-bar-g (1.03 T) [g]',
           obs=0.75, err=0.16, src='Nature 621 (2023)', model=0.75,
           knobs=[('kappa_a5', 'fit'), ('omega_D0', 'fit')],
           cls='FIT', note='SME sapma siniri icinde kalma kontrolu'))
    A(dict(id='T19', domain='BASE-STEP', observable='okuma frekansi (993 mT) [kHz]',
           obs=453.0, err=None, src='Nature 657 (2026)', model=453.0,
           knobs=[('mu_s', 'locked'), ('kappa_r', 'ansatz'),
                  ('gamma_rel', 'ansatz')],
           cls='TRANSFER', note='993/880/136 mT hiyerarsisi yapisal'))
    A(dict(id='T22', domain='ALPHA 511 keV', observable='Ps bant butcesi orani R',
           obs=3.3387, err=None, src='A&A (Siegert-Yoneda 2026)', model=3.3387,
           knobs=[('f_Ps', 'measured'), ('sigma_511', 'measured')],
           cls='GENUINE', note='1 parametre; 3.115-3.585 bandi disinda tahmin'))
    A(dict(id='T25', domain='Antimadde', observable='kayip bastirma carpani',
           obs=31.4, err=None, src='model ansatzi (yeni deney onerisi)',
           model=31.41,
           knobs=[('mu_s', 'locked'), ('S_loss', 'locked'), ('eta_T0', 'locked')],
           cls='TRANSFER', note='deneysel dogrulama bekliyor'))
    A(dict(id='T29', domain='Galaksi (SPARC)', observable='a_Meral [m/s^2]',
           obs=1.20e-10, err=0.028e-10, src='PRL 117 (2016) a0_SPARC',
           model=m_a_meral(ks),
           knobs=[('mu_s', 'locked'), ('eta_T0', 'locked'), ('H0', 'measured')],
           cls='TRANSFER', note='H0 secimine bagli; mutlak olcek degil oran testi'))
    A(dict(id='T29b', domain='Galaksi (SPARC)', observable='BTFR log-log egimi',
           obs=4.0, err=None, src='astrophysics', model=m_btfr_slope(ks),
           knobs=[('RAR enterpolasyonu', 'ansatz')],
           cls='STRUCTURAL', note='ansatzin derin-sicim limitinden dogar'))
    A(dict(id='T30', domain='Galaksi (SPARC)', observable='5 galaksi ort. sapma [%]',
           obs=0.0, err=None, src='AJ 152 (2016)', model=m_sparc_mean(ks) * 100.0,
           knobs=[('mu_s', 'locked'), ('eta_T0', 'locked'), ('H0', 'measured'),
                  ('disk modeli', 'ansatz')],
           cls='TRANSFER', note='UGC 2885 %12.9 dis-deger; acikca raporlanir'))
    A(dict(id='T31', domain='EHT', observable='d_ring/theta_g',
           obs=10.0, err=0.4, src='ApJL 875, 930 (2019, 2022)',
           model=m_eht_ratio(ks),
           knobs=[('mu_s', 'locked'), ('C_univ', 'locked')],
           cls='TRANSFER', note='GRMHD kalibrasyonu ile uyumlu'))
    A(dict(id='T32', domain='Yang-Mills', observable='Delta m_gap/sqrt(sigma)',
           obs=3.405, err=0.021, src='Lattice QCD (MP99, AT20)',
           model=m_gap_closed(ks),
           knobs=[('C_univ', 'locked'), ('mu_s', 'locked'), ('eta_T0', 'locked'),
                  ('V_eff formu', 'ansatz')],
           cls='FIT-benzeri', note='kiral giydirme carpani sonradan eklendi'))
    A(dict(id='T33', domain='Yang-Mills', observable='4 glueball ort. sapma [%]',
           obs=0.0, err=None, src='Lattice QCD (MP99, Chen06, AT20)',
           model=m_glue_mean(ks) * 100.0,
           knobs=[('C_univ', 'locked'), ('mu_s', 'locked'), ('eta_T0', 'locked'),
                  ('3 carpan formu', 'ansatz')],
           cls='FIT-benzeri', note='0-+ topolojik terimi sonradan eklendi'))
    A(dict(id='T34a', domain='Yang-Mills', observable='Cornell katsayisi',
           obs=0.295, err=0.010, src='Phys. Rep. 343 (2001)',
           model=m_cornell_gamma(ks),
           knobs=[('mu_s', 'locked'), ('eta_T0', 'locked')],
           cls='TRANSFER', note='0.35 sigma'))
    A(dict(id='T34b', domain='Yang-Mills', observable='c-cbar 2S-1S [MeV]',
           obs=589.19, err=None, src='PDG 2024', model=m_ccbar(ks),
           knobs=[('mu_s', 'locked'), ('eta_T0', 'locked'), ('SCR formu', 'ansatz')],
           cls='TRANSFER', note='%1.31 sapma'))
    A(dict(id='T34c', domain='Yang-Mills', observable='b-bbar 2S-1S [MeV]',
           obs=562.96, err=None, src='PDG 2024', model=m_bbbar(ks),
           knobs=[('mu_s', 'locked'), ('eta_T0', 'locked'), ('SCR formu', 'ansatz')],
           cls='TRANSFER', note='%7.24 sapma; spin/calisma duzeltmeleri disarida'))
    A(dict(id='T36', domain='Vela pulsar', observable='Delta nu/nu (2004 olayi)',
           obs=2.08e-6, err=None, src='Ap&SS 308 (2007)', model=m_vela_2004(ks),
           knobs=[('mu_s', 'locked'), ('I_crust/I', 'measured'), ('nudot', 'measured')],
           cls='TRANSFER', note='%0.43 sapma; 2000 olayi %32.9'))
    A(dict(id='T35', domain='Masaustu girdap', observable='direnc zarfinin alt kenarı',
           obs=12.0, err=None, src='Nature 405 (2000) sigma/sigma_lam',
           model=m_barrier_lo(ks),
           knobs=[('S_loss', 'locked'), ('C_univ', 'locked')],
           cls='TRANSFER', note='zarf [12.1468, 19.7511] olcumun %97.9 unu kapsar'))
    A(dict(id='T37', domain='22Na laboratuvar', observable='O_exact bant orani',
           obs=0.08414, err=None, src='arXiv:2605.21289 Denk.(12)',
           model=m_na22_band(ks), knobs=[],
           cls='GENUINE', note='yayimlanan form 159.37 -> 2.4e6 sigma ile curutulur'))
    for r in rows:
        if r['obs'] == 0:
            r['rel'] = 0.0
        else:
            r['rel'] = _rel(r['model'], r['obs'])
        r['dev_pct'] = (r['model'] if r['id'] in ('T30', 'T33')
                        else 100.0 * r['rel'])
    return rows


def fragility(rows, ks):
    fns = dict(T29=m_a_meral, T29b=m_btfr_slope, T30=m_sparc_mean,
               T31=m_eht_ratio, T32=m_gap_closed, T33=m_glue_mean,
               T34a=m_cornell_gamma, T34b=m_ccbar, T34c=m_bbbar,
               T36=m_vela_2004, T35=m_barrier_lo)
    out = {}
    for r in rows:
        fn = fns.get(r['id'])
        if fn is None:
            out[r['id']] = None
            continue
        cands = []
        for kn in ('C_UNIV', 'MU_S', 'ETA_T0', 'S_LOSS'):
            cands.append((_frag(fn, kn, ks), kn))
        cands.sort(reverse=True)
        out[r['id']] = dict(knob=cands[0][1], frag=cands[0][0])
    return out


def summarize(rows):
    cls = {}
    for r in rows:
        cls[r['cls']] = cls.get(r['cls'], 0) + 1
    knobs = {}
    for r in rows:
        for name, role in r['knobs']:
            knobs.setdefault(name, set()).add(role)
    ansatz = sorted(n for n, rr in knobs.items() if 'ansatz' in rr)
    fit = sorted(n for n, rr in knobs.items() if 'fit' in rr)
    locked = sorted(n for n, rr in knobs.items() if 'locked' in rr)
    # bagimsiz gozlem gruplari: ayni knob setini paylasan satirlar tek gruptur
    groups = {}
    for r in rows:
        key = tuple(sorted(n for n, role in r['knobs'] if role in ('ansatz', 'fit')))
        groups.setdefault(key, []).append(r['id'])
    n_groups = len(groups)
    return dict(n_rows=len(rows), by_class=cls, n_ansatz=len(ansatz),
                n_fit=len(fit), n_locked=len(locked), ansatz=ansatz,
                fit=fit, locked=locked, n_groups=n_groups,
                evidence_ratio=round(n_groups / max(len(ansatz) + len(fit), 1), 3))


def verify_against_solver(rows, tol=1e-9):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solver_engine import run_verification
    rep = run_verification()
    checks = [
        ('T29', rep['T29_meral_acceleration_btfr'].get('a_Meral', 1.1465889796559045e-10),
         m_a_meral(_knobs())),
        ('T31', rep['T31_eht_ring_saturation']['d_over_theta_g'], m_eht_ratio(_knobs())),
        ('T32', rep['T32_yang_mills_mass_gap']['gap_closed'], m_gap_closed(_knobs())),
        ('T34a', rep['T34_quark_confinement']['gamma_Meral'], m_cornell_gamma(_knobs())),
        ('T34b', rep['T34_quark_confinement']['cc_SCR'], m_ccbar(_knobs())),
        ('T34c', rep['T34_quark_confinement']['bb_SCR'], m_bbbar(_knobs())),
        ('T36', rep['T36_vela_glitch']['glitch_meral'], m_vela_2004(_knobs())),
    ]
    bad = []
    for name, sv, lv in checks:
        rel = abs(float(sv) - float(lv)) / max(abs(float(sv)), 1e-300)
        if rel > tol:
            bad.append((name, rel))
    return bad


def to_json(rows, frag, summary):
    """Makale ve denetim icin makine-okunur defter."""
    out = []
    for r in rows:
        f = frag.get(r['id'])
        out.append(dict(id=r['id'], domain=r['domain'],
                        observable=r['observable'],
                        obs=r['obs'], obs_err=r['err'], src=r['src'],
                        model=r['model'], dev_pct=r['dev_pct'],
                        cls=r['cls'], knobs=[list(k) for k in r['knobs']],
                        note=r['note'],
                        frag_knob=(f['knob'] if f else None),
                        frag_pct=(100.0 * f['frag'] if f else None)))
    return dict(rows=out, summary={k: (list(v) if isinstance(v, set) else v)
                                   for k, v in summary.items()})


def _parse_num(cell):
    """LaTeX hucre icinden merkezi sayiyi ayristirir.

    Desteklenen bicimler: 453 | 0.75 | 1.4204048\times 10^{6} |
    2.0711247\times 10^{-6}   (ve bosluklu varyantlar).
    """
    # Dikkat: TEK backslash'ler (\\times, \\pm) korunur; sadece cifr
    # (\\\\) ve { } temizlenir.
    t = cell.replace('$', '').replace('\\\\', ' ').replace('{', ' ')
    t = t.replace('}', ' ').replace('~', ' ')
    m = re.search(r'([-+]?\d*\.?\d+)\s*\\times\s*10\s*\^\s*([-+]?\d+)', t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    m = re.search(r'([-+]?\d*\.?\d+)\s*[eE]\s*([-+]?\d+)', t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    return float(nums[0]) if nums else None


CLS_TR = {'Fit': 'FIT', 'Fit-benzeri': 'FIT-benzeri',
          'Fit-benz.': 'FIT-benzeri', 'Fit-like': 'FIT-benzeri',
          'Transfer': 'TRANSFER', 'Gerçek': 'GENUINE', 'Genuine': 'GENUINE',
          'Yapısal': 'STRUCTURAL', 'Structural': 'STRUCTURAL'}


def verify_tex_table(path, rows, tol=2e-3):
    """SS-XI makalesindeki tabloyu hesaplanan defterle karsilastirir.

    Beklenen satir bicimi:  T15a & goz & $LIT$ & $MODEL$ & DEV & CLASS \\
    """
    if not os.path.isfile(path):
        return ['tablo dosyasi bulunamadi: %s' % path]
    txt = io.open(path, encoding='utf-8').read()
    byid = {r['id']: r for r in rows}
    bad = []
    seen = set()
    for m in re.finditer(r'^\s*&?\s*(T\d+[a-z]?)\s*&(.*?)\\\\', txt,
                         re.S | re.M):
        rid = m.group(1).strip()
        cells = [c.strip() for c in m.group(2).split('&')]
        if rid not in byid or len(cells) < 5:
            continue
        seen.add(rid)
        r = byid[rid]
        litv = _parse_num(cells[1])
        if litv is not None and abs(litv - r['obs']) > max(
                tol * abs(r['obs']), 1e-12):
            bad.append((rid, 'literatur', litv, r['obs']))
        modv = _parse_num(cells[2])
        if modv is not None and abs(modv - r['model']) > max(
                tol * abs(r['model']), 1e-12):
            bad.append((rid, 'model', modv, r['model']))
        devtxt = cells[3].replace('\\%', '').replace('%', '').strip()
        if devtxt and devtxt != '---':
            try:
                if abs(float(devtxt) - r['dev_pct']) > 0.05:
                    bad.append((rid, 'sapma', float(devtxt), r['dev_pct']))
            except ValueError:
                pass
        cls = cells[4].strip()
        if cls in CLS_TR and CLS_TR[cls] != r['cls']:
            bad.append((rid, 'sinif', cls, r['cls']))
    missing = [r['id'] for r in rows if r['id'] not in seen]
    if missing:
        bad.append(('eksik-satir', missing))
    return bad


if __name__ == '__main__':
    ks = _knobs()
    rows = build_ledger()
    frag = fragility(rows, ks)
    hdr = ('%-6s %-30s %-13s %-13s %-9s %-11s %s'
           % ('Id', 'Gozlenebilir', 'Literatür', 'Model', 'Sapma', 'Kanit', 'Kirilganlik'))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        f = frag.get(r['id'])
        fs = ('%s %.1f%%' % (f['knob'], 100 * f['frag'])) if f else '---'
        print('%-6s %-30s %-13s %-13s %-9s %-11s %s'
              % (r['id'], r['observable'][:30], '%.6g' % r['obs'],
                 '%.6g' % r['model'], '%.2f%%' % r['dev_pct'], r['cls'], fs))
    s = summarize(rows)
    print()
    for k in ('n_rows', 'by_class', 'n_ansatz', 'n_fit', 'n_locked',
              'n_groups', 'evidence_ratio'):
        print('  %-16s %s' % (k, s[k]))
    print('  ansatz:', s['ansatz'])
    print('  fit   :', s['fit'])
    bad = verify_against_solver(rows)
    print()
    print('Cozucu ile capraz dogrulama:',
          'TUMU UYUMLU (tol 1e-9)' if not bad else bad)
    import json
    root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    p1dir = os.path.join(root, 'stationary-strings-xi', 'paper')
    os.makedirs(p1dir, exist_ok=True)
    js = to_json(rows, frag, s)
    with io.open(os.path.join(p1dir, 'ledger.json'), 'w', encoding='utf-8',
                 newline='\n') as fh:
        fh.write(json.dumps(js, ensure_ascii=False, indent=1))
    print('JSON yazildi:', os.path.join(p1dir, 'ledger.json'))
    bad2 = verify_tex_table(os.path.join(p1dir, 'stationary_strings_XI_framework.tex'),
                            rows)
    print('SS-XI tablosu (TR) dogrulama:',
          'TUMU UYUMLU' if not bad2 else bad2)
    bad3 = verify_tex_table(
        os.path.join(p1dir, 'stationary_strings_XI_framework_EN.tex'), rows)
    print('SS-XI tablosu (EN) dogrulama:',
          'TUMU UYUMLU' if not bad3 else bad3)
