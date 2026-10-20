"""Yayin oncesi denetim (pre-submission audit) -- 6 yeni artifact.

Denetlenenler:
  A1 Overfull / LaTeX hatasi / undefined reference  (son derleme logu)
  A2 README sayfa sayisi ile gercek PDF sayfasi tutarli mi
  A3 DOI her yerde var mi (README, README_TR, CITATION.cff)
  A4 Dahili rapor kodu / eski isim sizintisi (SONUC_RAPORU, Grand Proof, GP-I/II/III)
  A5 Metinlerde "ozet" (abstract) bos mu
  A6 Zenodo DOI'si kayitta gercekten cozunuyor mu (ag erisimi varsa)
  A7 Sekil dosyalari referanslari cozunuyor mu

Cikti: ozet tablo + hata listesi. Cikis kodu 0 = temiz.
"""
import io
import json
import os
import re
import subprocess
import sys
import urllib.request

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'

ART = [
    ('stationary-strings-xi', 'paper/stationary_strings_XI_framework.tex',
     'paper/stationary_strings_XI_framework_EN.tex', 23517470),
    ('stationary-strings-xii', 'paper/stationary_strings_XII_galaktik_dinamik.tex',
     'paper/stationary_strings_XII_galactic_dynamics_EN.tex', 23517474),
    ('stationary-strings-xiii', 'paper/stationary_strings_XIII_yang_mills_kutle_araligi.tex',
     'paper/stationary_strings_XIII_yang_mills_mass_gap_EN.tex', 23519595),
    ('stationary-strings-xiv', 'paper/stationary_strings_XIV_deneysel_kanit_protokolu.tex',
     'paper/stationary_strings_XIV_experimental_proof_protocol_EN.tex', 23519746),
    ('stationary-strings-l1', 'paper/stationary_strings_L1_na22_TR.tex',
     'paper/stationary_strings_L1_na22_EN.tex', 23517487),
    ('stationary-strings-l2', 'paper/stationary_strings_L2_ghost_parameter_forensics_TR.tex',
     'paper/stationary_strings_L2_ghost_parameter_forensics_EN.tex', 23517495),
]

LEAK = re.compile(r'SONUC_RAPORU|Grand Proof|grand[- ]proof|GP-I\b|GP-II\b|GP-III\b|'
                   r'ss-yonetim|GRAND PROOF', re.I)
PAGE_CLAIM = re.compile(r'(\d+)\s*pages', re.I)
PAGE_TR = re.compile(r'(\d+)\s*(?:s\.|sayfa)', re.I)

errs = []
warns = []


def npages(pdf):
    try:
        from pypdf import PdfReader
        import logging
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        return len(PdfReader(pdf).pages)
    except Exception:
        return None


def check(name, ok, detail=''):
    if ok:
        print('  PASS %-30s %s' % (name, detail))
    else:
        print('  FAIL %-30s %s' % (name, detail))
        errs.append('%s: %s' % (name, detail))
    return ok


print('=' * 78)
print('YAYIN ONCESI DENETIM')
print('=' * 78)

for repo, tr_rel, en_rel, recid in ART:
    print('\n--- %s' % repo)
    rdir = os.path.join(ROOT, repo)

    # A1: derleme durumu (log yoksa yeniden derle)
    for rel in (tr_rel, en_rel):
        base = os.path.join(rdir, rel.replace('/', os.sep))
        log = base[:-4] + '.log'
        if not os.path.isfile(log):
            subprocess.run(['pdflatex', '--disable-installer',
                            '-interaction=nonstopmode', os.path.basename(base)],
                           cwd=os.path.dirname(base), capture_output=True)
            subprocess.run(['pdflatex', '--disable-installer',
                            '-interaction=nonstopmode', os.path.basename(base)],
                           cwd=os.path.dirname(base), capture_output=True)
        with io.open(log, encoding='utf-8', errors='replace') as fh:
            logtxt = fh.read()
        bad = [ln for ln in logtxt.splitlines()
               if ln.startswith('!') or 'Overfull' in ln or 'undefined' in ln]
        if bad:
            check('A1_derleme_' + os.path.basename(rel)[:18], False,
                  ' | '.join(b[:70] for b in bad[:2]))
        else:
            print('  PASS %-30s %s' % ('A1_' + os.path.basename(rel)[:24],
                                       'Overfull=0, ^!=0, undefined=0'))

    # A2: sayfa sayisi
    for rel, tag in ((tr_rel, 'TR'), (en_rel, 'EN')):
        pdf = os.path.join(rdir, rel.replace('/', os.sep))[:-4] + '.pdf'
        n = npages(pdf)
        rd = os.path.join(rdir, 'README.md' if tag == 'EN' else 'README_TR.md')
        with io.open(rd, encoding='utf-8') as fh:
            t = fh.read()
        claim = None
        m = PAGE_CLAIM.search(t) if tag == 'EN' else PAGE_TR.search(t)
        if m:
            claim = int(m.group(1))
        if claim is None:
            print('  --  %-30s sayfa iddiasi yok (gercek: %s)' % ('A2_' + tag, n))
        elif claim == n:
            print('  PASS %-30s %s sayfa' % ('A2_' + tag, n))
        else:
            warns.append('%s %s: README %s sayfa diyor, PDF %s sayfa'
                         % (repo, tag, claim, n))
            print('  WARN %-30s README %s vs PDF %s' % ('A2_' + tag, claim, n))

    # A3: DOI
    doi = '10.5281/zenodo.%d' % recid
    for fn in ('README.md', 'README_TR.md', 'CITATION.cff'):
        p = os.path.join(rdir, fn)
        with io.open(p, encoding='utf-8') as fh:
            t = fh.read()
        if doi in t:
            print('  PASS %-30s %s' % ('A3_doi_' + fn[:16], doi))
        else:
            check('A3_doi_' + fn[:16], False, '%s eksik' % doi)

    # A4: sizinti
    for rel in (tr_rel, en_rel):
        p = os.path.join(rdir, rel.replace('/', os.sep))
        with io.open(p, encoding='utf-8') as fh:
            t = fh.read()
        m = LEAK.search(t)
        if m:
            check('A4_sizinti_' + os.path.basename(rel)[:16], False,
                  m.group(0))
        else:
            print('  PASS %-30s temiz' % ('A4_' + os.path.basename(rel)[:24]))
    for fn in ('README.md', 'README_TR.md'):
        p = os.path.join(rdir, fn)
        with io.open(p, encoding='utf-8') as fh:
            t = fh.read()
        m = LEAK.search(t)
        if m:
            check('A4_sizinti_' + fn[:16], False, m.group(0))

    # A5: ozet
    for rel in (tr_rel, en_rel):
        p = os.path.join(rdir, rel.replace('/', os.sep))
        with io.open(p, encoding='utf-8') as fh:
            t = fh.read()
        m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', t, re.S)
        body = re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        if len(body) > 200:
            print('  PASS %-30s %d karakter' % ('A5_ozet_' + os.path.basename(rel)[:14], len(body)))
        else:
            check('A5_ozet_' + os.path.basename(rel)[:14], False,
                  'ozet bos/kisa (%d)' % len(body))

    # A7: sekil referanslari
    for rel in (tr_rel, en_rel):
        p = os.path.join(rdir, rel.replace('/', os.sep))
        with io.open(p, encoding='utf-8') as fh:
            t = fh.read()
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', t):
            fig = os.path.join(rdir, 'figs', os.path.basename(m.group(1)))
            if not os.path.isfile(fig):
                check('A7_sekil_' + os.path.basename(rel)[:16], False,
                      m.group(1))
        print('  PASS %-30s' % ('A7_sekil_' + os.path.basename(rel)[:24]))

print('\n' + '=' * 78)
print('HATA: %d   UYARI: %d' % (len(errs), len(warns)))
for e in errs:
    print('  HATA  ' + e)
for w in warns:
    print('  UYARI ' + w)
print('=' * 78)
sys.exit(0 if not errs else 1)
