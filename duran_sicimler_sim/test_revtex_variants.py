"""revtex4-2 onay blok varyantlarini dener (hata sayisi ile).

Kullanim:
  python -W ignore test_revtex_variants.py <calisma dizini>
"""
import io
import os
import re
import subprocess
import sys

D = sys.argv[1] if len(sys.argv) > 1 else (
    r'G:\my_scientific_work\stationary-strings-theorem\stationary-strings-xiii'
    r'\paper')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TITLE = (r'\title{Stationary Strings XIII: From Navier--Stokes Vortex Saturation '
         r'to the 4D SU(3) Yang--Mills Mass-Gap Bridge and the Lattice QCD '
         r'Glueball Spectrum}')
AUTHOR = (r'\author{Fahri Meral}' + '\n'
          + r'\affiliation{Independent Researcher}' + '\n'
          + r'\date{October 19, 2026}')

HEAD = (r'\documentclass[reprint,aps,pre,twocolumn,notitlepage]{revtex4-2}'
        '\n' + r'\usepackage[utf8]{inputenc}' + '\n'
        + r'\usepackage[T1]{fontenc}' + '\n'
        + r'\usepackage{lmodern}' + '\n'
        + r'\usepackage{microtype}' + '\n'
        + r'\usepackage{amsmath, amssymb, amsfonts}' + '\n'
        + r'\usepackage{booktabs}' + '\n'
        + r'\usepackage{tabularx}' + '\n'
        + r'\usepackage{graphicx}' + '\n'
        + r'\providecommand{\url}[1]{\texttt{#1}}' + '\n\n')

ABS = (r'\begin{abstract}' + '\n' + 'Short abstract for the test.' + '\n'
       + r'\end{abstract}' + '\n')
TAIL = (r'\maketitle' + '\n' + 'Body text.\n'
        + r'\bibliographystyle{apsrev4-2}' + '\n' + r'\bibliography{refs}' + '\n'
        + r'\end{document}' + '\n')

VARIANTS = {
    'a_author_before_doc': HEAD + TITLE + '\n' + AUTHOR + '\n'
                             + r'\begin{document}' + '\n' + ABS + TAIL,
    'b_author_after_doc': HEAD + r'\begin{document}' + '\n' + TITLE + '\n'
                           + AUTHOR + '\n' + ABS + TAIL,
    'c_author_after_abs': HEAD + r'\begin{document}' + '\n' + ABS + TITLE + '\n'
                           + AUTHOR + '\n' + TAIL,
    'd_no_aps_option': HEAD.replace('[reprint,aps,pre,twocolumn,notitlepage]',
                                   '[reprint,pre,twocolumn,notitlepage]')
                         + TITLE + '\n' + AUTHOR + '\n'
                         + r'\begin{document}' + '\n' + ABS + TAIL,
    'e_no_affiliation': HEAD + TITLE + '\n' + r'\author{Fahri Meral}' + '\n'
                         + r'\begin{document}' + '\n' + ABS + TAIL,
    'f_author_with_email': HEAD + TITLE + '\n' + AUTHOR + '\n'
                            + r'\email{fahrimeral@outlook.com}' + '\n'
                            + r'\begin{document}' + '\n' + ABS + TAIL,
}

for name, src in sorted(VARIANTS.items()):
    fn = 'v_%s.tex' % name
    with io.open(os.path.join(D, fn), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(src)
    r = subprocess.run(['pdflatex', '--disable-installer', '-interaction=nonstopmode',
                        fn], cwd=D, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    errs = len(re.findall(r'^!', r.stdout, re.M))
    print('%-24s hata=%d' % (name, errs))
    if errs:
        m = re.search(r'^!.*$', r.stdout, re.M)
        if m:
            print('      ilk: %s' % m.group(0)[:90])
