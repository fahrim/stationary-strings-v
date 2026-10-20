"""arXiv'e gonderim icin hazir paketler uretir (bundle).

arXiv kurallari (info.arxiv.org/help/submit/index.html):
  - TeX kaynagi tercih edilir; duz (flat) dizin, `../` YASAK
  - mutlak yol YASAK
  - dosya adi karakterleri: a-z A-Z 0-9 _ + - . , =
  - tum sekiller gonderilmelidir (harici linke referans YASAK)
  - sozlesmeler arXiv'in standart TeXLive'unda vardir (bizimkiler: inputenc,
    fontenc, babel, lmodern, microtype, amsmath, geometry, setspace, url,
    xurl, booktabs, tabularx, graphicx, pgfplots -> hepsi standart)

Bu betik:
  1. EN .tex dosyasini + sekilleri duz bir dizine kopyalar
  2. `../figs/X` yollarini `X` yapar
  3. yasak kaliplari denetler (mutlak yol, `../`, harici include, dosya adi)
  4. paketi pdflatex ile DERLEYerek dogrular (arXiv de derleyecek)
  5. arXiv formunun metadata alanlarini hazirlar (baslik, ozet, yorum, konu)
"""
import argparse
import io
import os
import re
import shutil
import subprocess
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
DIST = os.path.join(ROOT, 'dist', 'arxiv')

# repo -> (EN tex dosyasi, arXiv birincil konu, arXiv capraz liste, not)
PAPERS = [
    ('stationary-strings-l1', 'paper/stationary_strings_L1_na22_EN.tex',
     'astro-ph.HE', 'hep-ex; physics.ins-det',
     '{pages} pages, 1 table. Corrects Eq. (12) of arXiv:2605.21289. '
     'Measurement proposed, not performed. Code: '
     'https://github.com/fahrim/stationary-strings-v'),
    ('stationary-strings-l2',
     'paper/stationary_strings_L2_ghost_parameter_forensics_EN.tex',
     'physics.soc-ph', 'cs.DL; quant-ph',
     '{pages} pages, 1 table, {figs} figure. Forensic method for literature '
     'values; '
     'the error analysed is our own. Code: '
     'https://github.com/fahrim/stationary-strings-v'),
    ('stationary-strings-xi', 'paper/stationary_strings_XI_framework_EN.tex',
     'hep-th', 'quant-ph',
     'Methods and self-assessment paper with a machine-readable '
     'degrees-of-freedom ledger (17 comparisons, evidence ratio 0.889). '
     'Code: https://github.com/fahrim/stationary-strings-v'),
    ('stationary-strings-xii',
     'paper/stationary_strings_XII_galactic_dynamics_EN.tex',
     'astro-ph.GA', 'hep-ph',
     'Application paper (T29-T31). Contains an explicit H0-dependence '
     'caveat. Code: https://github.com/fahrim/stationary-strings-v'),
    ('stationary-strings-xiii',
     'paper/stationary_strings_XIII_yang_mills_mass_gap_EN.tex',
     'hep-th', 'hep-lat',
     'Application paper (T32-T34). NOT a solution of the Yang-Mills mass '
     'gap; the operator is a model ansatz and the glueball agreement is '
     'not robust. Code: https://github.com/fahrim/stationary-strings-v'),
    ('stationary-strings-xiv',
     'paper/stationary_strings_XIV_experimental_proof_protocol_EN.tex',
     'hep-ex', 'physics.acc-phys; astro-ph.HE',
     'Application paper (T35-T37) with a proposed benchtop protocol. '
     'Code: https://github.com/fahrim/stationary-strings-v'),
]

BAD_PATH = re.compile(r'(\.\./|[A-Za-z]:\\\\|/(?:home|Users|mnt|tmp)/)')
INCLUDE = re.compile(r'\\(?:input|include)\{([^}]*)\}')
GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}')
ALLOWED_NAME = re.compile(r'^[A-Za-z0-9_+.,=-]+$')
ABS_RE = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.S)
TITLE_RE = re.compile(r'\\title\{(.*?)\}\s*\n', re.S)
AUTHOR_RE = re.compile(r'\\author\{(.*?)\}\s*\n', re.S)


def strip_tex(s):
    """Kaba LaTeX -> duz metin (metadata formu icin).

    Matematik KORUNUR: arXiv metadata formunda $...$ LaTeX olarak kabul
    edilir ve render edilir. Yalnizca metin bicimlendirme komutlari
    duzlestirilir.
    """
    s = re.sub(r'\\textbf\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\emph\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\textit\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\textsc\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\texttt\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\mathrm\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\\ref\{[^}]*\}', '', s)
    s = re.sub(r'\\(?:footnote|thanks)\{.*?\}', '', s, flags=re.S)
    # kalan komutlari bosalt ama icerigi koru
    s = re.sub(r'\\(?:small|large|Large|huge|bf|it|tt|emph|mbox)\b', '', s)
    s = s.replace('\\&', '&').replace('\\%', '%').replace('\\_', '_')
    s = s.replace('~', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def build(repo, tex_rel, subject, cross, comment, compile_pdf=True):
    src_repo = os.path.join(ROOT, repo)
    tex_path = os.path.join(src_repo, tex_rel.replace('/', os.sep))
    out = os.path.join(DIST, repo)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    with io.open(tex_path, encoding='utf-8') as fh:
        tex = fh.read()

    problems = []

    # 1) sekilleri duzlastir
    figdir = os.path.join(src_repo, 'figs')
    for m in GRAPHIC.finditer(tex):
        target = m.group(1)
        base = os.path.basename(target)
        src_fig = os.path.join(figdir, base)
        if not os.path.isfile(src_fig):
            problems.append('SEKIL BULUNAMADI: %s' % target)
            continue
        shutil.copy2(src_fig, os.path.join(out, base))
        tex = tex.replace(target, base)

    # 2) dis include'lar
    for m in INCLUDE.finditer(tex):
        f = os.path.basename(m.group(1))
        cand = os.path.join(src_repo, f)
        if os.path.isfile(cand):
            shutil.copy2(cand, os.path.join(out, f))
            tex = tex.replace(m.group(1), f)
        else:
            problems.append('DAHILI DOSYA BULUNAMADI: %s' % m.group(1))

    # 3) yas kaliplar
    if BAD_PATH.search(tex):
        problems.append('YASAK YOL DESENI: %s'
                        % ', '.join(sorted(set(BAD_PATH.findall(tex)))))
    for f in os.listdir(out):
        if not ALLOWED_NAME.match(f):
            problems.append('YASAK DOSYA ADI KARAKTERI: %s' % f)

    # 4) yaz
    with io.open(os.path.join(out, 'main.tex'), 'w', encoding='utf-8',
                 newline='\n') as fh:
        fh.write(tex)

    # 5) derle (arXiv de derleyecek)
    n_pages = None
    if compile_pdf and not problems:
        for _ in range(2):
            subprocess.run(['pdflatex', '--disable-installer',
                            '-interaction=nonstopmode', 'main.tex'],
                           cwd=out, capture_output=True)
        log = os.path.join(out, 'main.log')
        if os.path.isfile(log):
            with io.open(log, encoding='utf-8', errors='replace') as fh:
                logtxt = fh.read()
            bad = [ln for ln in logtxt.splitlines()
                   if ln.startswith('!') or 'Overfull' in ln]
            if bad:
                problems.append('DERLEME: %s' % ' | '.join(bad[:3]))
        pdf = os.path.join(out, 'main.pdf')
        if os.path.isfile(pdf):
            try:
                from pypdf import PdfReader
                import logging
                logging.getLogger('pypdf').setLevel(logging.CRITICAL)
                n_pages = len(PdfReader(pdf).pages)
            except Exception:
                n_pages = None
        for junk in ('main.aux', 'main.log', 'main.out', 'main.synctex.gz'):
            jp = os.path.join(out, junk)
            if os.path.isfile(jp):
                os.remove(jp)

    # 6) metadata
    nfig = len([f for f in os.listdir(out)
                if f.endswith(('.pdf', '.png', '.jpg')) and f != 'main.pdf'])
    abs_txt = strip_tex(ABS_RE.search(tex).group(1)) if ABS_RE.search(tex) else ''
    ttl = strip_tex(TITLE_RE.search(tex).group(1)) if TITLE_RE.search(tex) else ''
    au = strip_tex(AUTHOR_RE.search(tex).group(1)) if AUTHOR_RE.search(tex) else ''
    nfig_txt = str(nfig)
    comment_txt = comment.replace('{pages}', str(n_pages or '?')).replace(
        '{figs}', nfig_txt)
    meta = [
        'ARXIV METADATA -- %s' % repo,
        '=' * 74, '',
        'Title       : %s' % ttl, '',
        'Authors     : %s' % au, '',
        'Subjects    : %s   (cross-list: %s)' % (subject, cross), '',
        'Comments    : %s' % comment_txt, '',
        'License     : CC BY 4.0 (veya arXiv varsayilan kalici lisans)', '',
        'Report DOI  : (yayin sonrasi arXiv raporunda eklenir)', '',
        'Abstract (arXiv formuna yapistirilacak):', '',
        abs_txt, '',
    ]
    with io.open(os.path.join(out, 'arXiv_metadata.txt'), 'w',
                 encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(meta))

    return out, n_pages, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-compile', dest='compile', action='store_false',
                    default=True)
    args = ap.parse_args(argv)
    os.makedirs(DIST, exist_ok=True)
    print('=' * 74)
    print('arXiv PAKETLERI')
    print('=' * 74)
    allok = True
    for repo, tex, subj, cross, comment in PAPERS:
        out, pages, problems = build(repo, tex, subj, cross, comment,
                                     compile_pdf=args.compile)
        files = sorted(os.listdir(out))
        nfig = len([f for f in files if f.endswith(('.pdf', '.png', '.jpg'))
                    and f != 'main.pdf'])
        status = 'TAMAM' if not problems else 'SORUN'
        if problems:
            allok = False
        print('%-26s %-8s %2s s.  %d sekil  -> %s'
              % (repo, status, pages or '?', nfig, os.path.basename(out)))
        for p in problems:
            print('     ! %s' % p)
    print('=' * 74)
    print('TUM PAKETLER HAZIR: %s' % ('evet' if allok else 'HAYIR (sorunlara bak)'))
    print('Dizin: %s' % DIST)
    return 0 if allok else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
