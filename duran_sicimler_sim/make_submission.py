"""Gonderim klasorunu hazirlar: hangi dosya, neden, SHA-256 ile.

Bir dergiye gonderirken "dogru dosyalari mi gonderiyorum" sorusu en sik
yapilan hatadir. Bu arac, gonderilecek dosyalari tek klasorde toplar ve
her biri icin SHA-256 ozetini yazar; boylece gonderimden sonra
"gonderdigim dosya ile buradaki dosya ayni mi" sorusu tek komutla
yanitlanir.

Kullanim:
  python -W ignore make_submission.py --journal aa --paper l1
"""
import argparse
import hashlib
import io
import os
import shutil
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
DIST = os.path.join(ROOT, 'dist', 'submissions')

# journal -> (etiket, klasor oneki, ana dosya secimi)
# 'dergi_ozel': dergiye gore ana metin dosyasi (PRD icin revtex surumu)
JOURNALS = {
    'aa': ('Astronomy & Astrophysics', 'aa', {}),
    'qss': ('Quantitative Science Studies', 'qss', {}),
    'prd': ('Physical Review D', 'prd',
            {'stationary-strings-xiii':
             'paper/stationary_strings_XIII_yang_mills_mass_gap_REVTEX.tex'}),
    'epjp': ('European Physical Journal Plus', 'epjp', {}),
}

# paper -> (depo, EN tex, TR tex, A&A/dergi paket dosyasi)
PAPERS = {
    'l1': ('stationary-strings-l1',
           'paper/stationary_strings_L1_na22_AA_EN.tex',
           'paper/stationary_strings_L1_na22_AA_TR.tex',
           'stationary-strings-previews/journal_kit/submission_aa_L1.md'),
    'l2': ('stationary-strings-l2',
           'paper/stationary_strings_L2_ghost_parameter_forensics_EN.tex',
           'paper/stationary_strings_L2_ghost_parameter_forensics_TR.tex',
           'stationary-strings-previews/journal_kit/cover_letter_L2.md'),
    'xiii': ('stationary-strings-xiii',
             'paper/stationary_strings_XIII_yang_mills_mass_gap_EN.tex',
             'paper/stationary_strings_XIII_yang_mills_kutle_araligi.tex',
             'stationary-strings-previews/journal_kit/cover_letter_SS-XIII.md'),
    'xi': ('stationary-strings-xi',
           'paper/stationary_strings_XI_framework_EN.tex',
           'paper/stationary_strings_XI_framework.tex',
           'stationary-strings-previews/journal_kit/cover_letter_SS-XI.md'),
    'xiv': ('stationary-strings-xiv',
            'paper/stationary_strings_XIV_experimental_proof_protocol_EN.tex',
            'paper/stationary_strings_XIV_deneysel_kanit_protokolu.tex',
            'stationary-strings-previews/journal_kit/cover_letter_SS-XIV.md'),
    'xii': ('stationary-strings-xii',
            'paper/stationary_strings_XII_galactic_dynamics_EN.tex',
            'paper/stationary_strings_XII_galaktik_dinamik.tex',
            'stationary-strings-previews/journal_kit/HOLD_SS-XII.md'),
}


# Dergi sinfi / stil dosyalari: kaynakla birlikte verilmeli
EXTRA = {
    'l1': ['paper/aa.cls', 'paper/aa.bst', 'paper/linenoaa.sty',
           'paper/lineno.sty', 'paper/cuted.sty'],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--journal', required=True, choices=sorted(JOURNALS))
    ap.add_argument('--paper', required=True, choices=sorted(PAPERS))
    args = ap.parse_args(argv)

    jname, jpref, override = JOURNALS[args.journal]
    repo, en_rel, tr_rel, kit_rel = PAPERS[args.paper]
    en_rel = override.get(repo, en_rel)
    rdir = os.path.join(ROOT, repo)
    out = os.path.join(DIST, '%s_%s' % (jpref, args.paper))
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    items = []
    for rel, why in ((en_rel, 'ana metin (EN)'),
                     (tr_rel, 'TR esligi (yalnizca A&A icin)'),
                     ('paper/refs.bib', 'makine-okunur kaynakca')):
        src = os.path.join(rdir, rel.replace('/', os.sep))
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out, os.path.basename(src))
        shutil.copy2(src, dst)
        items.append((os.path.basename(src), why, dst))
    # derlenmis PDF (asil yuklenen dosya)
    en_pdf = os.path.join(rdir, en_rel.replace('/', os.sep))[:-4] + '.pdf'
    if os.path.isfile(en_pdf):
        dst = os.path.join(out, os.path.basename(en_pdf))
        shutil.copy2(en_pdf, dst)
        items.insert(0, (os.path.basename(en_pdf), 'derlenmis PDF (ana yukleme)', dst))
    # TR esliginin PDF'i (istege bagli; pakette hazir bulunsun)
    tr_pdf = os.path.join(rdir, tr_rel.replace('/', os.sep))[:-4] + '.pdf'
    if os.path.isfile(tr_pdf):
        dst = os.path.join(out, os.path.basename(tr_pdf))
        shutil.copy2(tr_pdf, dst)
        items.append((os.path.basename(tr_pdf), 'TR esligi PDF (istege bagli)', dst))
    # dergi sinfi dosyalari (kaynakla birlikte)
    for rel in EXTRA.get(args.paper, []):
        src = os.path.join(rdir, rel.replace('/', os.sep))
        if not os.path.isfile(src):
            print('  ! eksik: %s' % rel)
            continue
        dst = os.path.join(out, os.path.basename(src))
        shutil.copy2(src, dst)
        items.append((os.path.basename(src), 'dergi sinfi dosyasi', dst))
    # sekiller (dergi ayrica istiyorsa)
    figdir = os.path.join(rdir, 'figs')
    if os.path.isdir(figdir):
        for fn in sorted(os.listdir(figdir)):
            if fn.endswith('.pdf'):
                src = os.path.join(figdir, fn)
                dst = os.path.join(out, fn)
                shutil.copy2(src, dst)
                items.append((fn, 'sekil (PDF)', dst))
    kit = os.path.join(ROOT, kit_rel.replace('/', os.sep))
    if os.path.isfile(kit):
        dst = os.path.join(out, 'SUBMISSION_NOTES.md')
        shutil.copy2(kit, dst)
        items.append(('SUBMISSION_NOTES.md', 'gonderim talimati + kapak mektubu', dst))

    lines = ['GONDERIM MANIFESTI',
             '=' * 70,
             'Dergi      : %s' % jname,
             'Makale     : %s' % repo,
             'Klasor     : %s' % out,
             '',
             '%-46s %-34s %s' % ('DOSYA', 'NEDEN', 'SHA-256 (ilk 16)'),
             '-' * 70]
    for name, why, path in items:
        lines.append('%-46s %-34s %s'
                     % (name, why[:33], sha256(path)[:16]))
    lines += ['', 'Kontrol (bu klasoru yeniden uretmek icin):',
              '  python -W ignore duran_sicimler_sim/make_submission.py '
              '--journal %s --paper %s' % (args.journal, args.paper),
              '',
              'Gonderimden SONRA: dosyalari GitHub deposunda commit et;',
              'Zenodo v2 kaydi (yeni surum) acilisip DOI guncellenir.']
    man = os.path.join(out, 'MANIFEST.txt')
    with io.open(man, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('\nKlasor: %s' % out)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
