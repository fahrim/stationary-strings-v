"""arXiv form alanlarini hazirlar ve dogrular.

arXiv kurallari (info.arxiv.org/help/prep.html, 21.10.2026):
  * metadata alanlari SADECE ASCII kabul eder; unicode -> LaTeX
  * ozet 1920 karakteri gecmemelidir
  * baslik: tum buyuk harf yok, unicode yok, MathJax serbest
  * yazar: "Firstname Lastname", baglantida parantez icinde
  * Comments: sayfa + sekil sayisi + DOI (Zenodo DOI'si BURAYA gider)
  * DOI alani: yalnizca makalenin baska bir surumune cozumlenen DOI
    (arXiv kendi DOI'sini atar; Zenodo DOI'si DOI alanina GIRMEZ)
  * Journal-ref: yalnizca yayimlanmissa
"""
import argparse
import io
import os
import re
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
OUT = os.path.join(ROOT, 'dist', 'arxiv_meta')

# Unicode -> ASCII/LaTeX (arXiv alanlari ASCII-only)
UNI = [
    ('\u2014', '---'), ('\u2013', '--'), ('\u2012', '--'),
    ('\u2018', "`"), ('\u2019', "'"), ('\u201c', '``'), ('\u201d', "''"),
    ('\u2264', r'$\leq$'), ('\u2265', r'$\geq$'), ('\u2248', r'$\approx$'),
    ('\u2260', r'$\neq$'), ('\u00d7', r'$\times$'), ('\u00b7', r'$\cdot$'),
    ('\u00b1', r'$\pm$'), ('\u2212', '-'), ('\u00b5', r'$\mu$'),
    ('\u03bc', r'$\mu$'), ('\u03c9', r'$\omega$'), ('\u03bb', r'$\lambda$'),
    ('\u03b1', r'$\alpha$'), ('\u03b3', r'$\gamma$'), ('\u03b4', r'$\delta$'),
    ('\u03c3', r'$\sigma$'), ('\u03c4', r'$\tau$'), ('\u03c6', r'$\phi$'),
    ('\u03b5', r'$\epsilon$'), ('\u03b8', r'$\theta$'), ('\u03c0', r'$\pi$'),
    ('\u03c1', r'$\rho$'), ('\u0394', r'$\Delta$'), ('\u03a3', r'$\Sigma$'),
    ('\u222b', r'$\int$'), ('\u2211', r'$\sum$'), ('\u2202', r'$\partial$'),
    ('\u221a', r'$\sqrt$'), ('\u2207', r'$\nabla$'), ('\u2192', r'$\to$'),
    ('\u221e', r'$\infty$'), ('\u2261', r'$\equiv$'), ('\u2208', r'$\in$'),
    ('\u2205', r'$\emptyset$'), ('\u2295', r'$\oplus$'),
    ('\u00b0', ' deg'), ('\u2032', "'"), ('\u2033', "''"),
    ('\u2026', '...'), ('\u00a0', ' '),
]
# Ust simgeler (kuadratik: 22 -> ^{22})
SUP = {'\u00b2': '^{2}', '\u00b3': '^{3}', '\u2074': '^{4}', '\u2075': '^{5}',
       '\u2076': '^{6}', '\u2077': '^{7}', '\u2078': '^{8}', '\u2079': '^{9}',
       '\u2070': '^{0}', '\u00b9': '^{1}'}
# Alt simgeler
SUB = {'\u2080': '_{0}', '\u2081': '_{1}', '\u2082': '_{2}', '\u2083': '_{3}',
       '\u2084': '_{4}', '\u2085': '_{5}', '\u2086': '_{6}', '\u2087': '_{7}',
       '\u2088': '_{8}', '\u2089': '_{9}'}


def to_arxiv_ascii(s):
    """arXiv metadata alanlari icin ASCII'ye indirger."""
    out = []
    for ch in s:
        if ch in SUP:
            out.append(SUP[ch])
        elif ch in SUB:
            out.append(SUB[ch])
        else:
            out.append(ch)
    s = ''.join(out)
    for a, b in UNI:
        s = s.replace(a, b)
    # kalan unicode: kaldir ve uyar
    s = ''.join(c if ord(c) < 128 else '?' for c in s)
    # LaTeX komutlarini sadelestir (arXiv "TeX-isms" istemez)
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    s = re.sub(r'\\ref\{[^}]*\}', '', s)
    s = re.sub(r'\\cite\{[^}]*\}', '', s)
    s = re.sub(r'\\citep?\*?\{[^}]*\}', '', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textit\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textsc\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[,;:!]', ' ', s)
    s = re.sub(r'\\(?:small|large|Large|huge|bf|it|tt|rm)\b', '', s)
    s = s.replace('~', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{2,}', '\n', s)
    return s.strip()


def extract(tex):
    m = re.search(r'\\title\s*\{', tex)
    title = ''
    if m:
        i = tex.find('{', m.end() - 1)
        depth, j = 0, i
        while j < len(tex):
            if tex[j] == '{':
                depth += 1
            elif tex[j] == '}':
                depth -= 1
                if depth == 0:
                    title = tex[i + 1:j]
                    break
            j += 1
    m2 = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.S)
    abstract = m2.group(1) if m2 else ''
    m3 = re.search(r'\\author\{([^}]*)\}', tex)
    author = m3.group(1) if m3 else 'Fahri Meral'
    return title, abstract, author


def pages_figures(repo):
    """Sayfa ve sekil sayisi (derlenmis PDF'ten)."""
    import glob
    from pypdf import PdfReader
    import logging
    logging.getLogger('pypdf').setLevel(logging.CRITICAL)
    pdir = os.path.join(ROOT, repo, 'paper')
    npages = nfig = 0
    for pdf in glob.glob(os.path.join(pdir, '*_EN.pdf')):
        try:
            npages = max(npages, len(PdfReader(pdf).pages))
        except Exception:
            pass
    for fig in glob.glob(os.path.join(pdir, 'figs', '*.pdf')):
        nfig += 1
    return npages, nfig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-chars', type=int, default=1920)
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    # (etiket, depo, EN kaynak, kendi Zenodo DOI'si)
    papers = {
        'xi': ('stationary-strings-xi',
               'paper/stationary_strings_XI_framework_EN.tex', 23517470),
        'xii': ('stationary-strings-xii',
                'paper/stationary_strings_XII_galactic_dynamics_EN.tex', 23517474),
        'xiii': ('stationary-strings-xiii',
                 'paper/stationary_strings_XIII_yang_mills_mass_gap_EN.tex',
                 23519595),
        'xiv': ('stationary-strings-xiv',
                'paper/stationary_strings_XIV_experimental_proof_protocol_EN.tex',
                23519746),
        'l1': ('stationary-strings-l1',
               'paper/stationary_strings_L1_na22_EN.tex', 23517487),
        'l2': ('stationary-strings-l2',
               'paper/stationary_strings_L2_ghost_parameter_forensics_EN.tex',
               23517495),
    }
    CODE_DOI = '10.5281/zenodo.23408253'
    # Birincil kategori + capraz liste (onay alanina gore secilir)
    SUBJ = {
        'xi': ('hep-th', 'quant-ph'),
        'xii': ('astro-ph.GA', 'hep-ph'),
        'xiii': ('hep-lat', 'hep-th'),
        'xiv': ('hep-ex', 'physics.acc-phys, astro-ph.HE'),
        'l1': ('astro-ph.HE', 'hep-ex, physics.ins-det'),
        'l2': ('physics.soc-ph', 'cs.DL, quant-ph'),
    }
    for tag, (repo, en_rel, recdoi) in papers.items():
        path = os.path.join(ROOT, repo, en_rel.replace('/', os.sep))
        with io.open(path, encoding='utf-8') as fh:
            tex = fh.read()
        title, abstract, author = extract(tex)
        t_a = to_arxiv_ascii(title)
        # arXiv formu icin kisaltilmis ozet (varsa): makale metni DEGISMEZ,
        # sadece form alanina yapistirilacak surum degisir.
        ov = os.path.join(OUT, 'overrides', '%s_abstract.txt' % tag)
        if os.path.isfile(ov):
            with io.open(ov, encoding='utf-8') as fh:
                abstract = fh.read()
        a_a = to_arxiv_ascii(abstract)
        npages, nfig = pages_figures(repo)
        block = [
            'Title    : %s' % t_a,
            'Authors  : Fahri Meral (Independent Researcher)',
            'Comments : %d pages, %d figure(s). Preprint DOI: 10.5281/zenodo.%d. '
            'Code and verification suite: %s '
            '(https://github.com/fahrim/stationary-strings-v)'
            % (npages, nfig, recdoi, CODE_DOI),
            'Subjects : %s (primary); cross-list: %s' % SUBJ[tag],
            '',
            'Abstract:',
            a_a,
        ]
        outp = os.path.join(OUT, '%s.txt' % tag)
        with io.open(outp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(block) + '\n')
        n = len(a_a)
        flag = 'OK' if n <= args.max_chars else 'ASIMDA (+%d)' % (n - args.max_chars)
        print('%-6s ozet=%4d karakter  %-14s  baslik=%d karakter'
              % (tag, n, flag, len(t_a)))
        if n > args.max_chars:
            print('        -> dist/arxiv_meta/%s.txt (kisaltma gerek)' % tag)
    print('\nKlasor: %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
