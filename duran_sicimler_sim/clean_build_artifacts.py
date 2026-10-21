"""Derleme artiklarini temizler (paper/ klasorleri).

Varsayilan davranis: YALNIZCA RAPOR eder (hicbir sey silmez).
Silme icin --apply kullanilir.

Silinenler: LaTeX/BibTeX derleme ciktilari (.aux .log .out .toc .bbl .blg
*Notes.bib) ve deneme dosyalari (loc.*, mini2.*). Kaynak (.tex), .bib,
sekiller ve DERLENMIS PDF'ler korunur -- bir kez silinen PDF, arXiv/dergi
gonderimi icin geri getirilemez.
"""
import argparse
import io
import os
import re
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
REPOS = ['stationary-strings-xi', 'stationary-strings-xii',
         'stationary-strings-xiii', 'stationary-strings-xiv',
         'stationary-strings-l1', 'stationary-strings-l2']

ARTEFACT = re.compile(
    r'^(loc|mini2|bibtest|r|l1|l2|e1|e2|t1|t2|p1|p2|q1|q2|en6|tr6)\.'
    r'(aux|log|out|toc|bbl|blg|pdf|tex)$'
    r'|\.aux$|\.log$|\.out$|\.toc$|\.bbl$|\.blg$|Notes\.bib$'
    r'|\.synctex\.gz$|\.fls$|\.fdb_latexmk$',
    re.I)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='varsayilan: yalnizca rapor. --apply ile sil.')
    args = ap.parse_args(argv)

    total = 0
    for repo in REPOS:
        pdir = os.path.join(ROOT, repo, 'paper')
        if not os.path.isdir(pdir):
            continue
        removed = []
        for fn in sorted(os.listdir(pdir)):
            if not ARTEFACT.search(fn):
                continue
            p = os.path.join(pdir, fn)
            if not os.path.isfile(p):
                continue
            if args.apply:
                os.remove(p)
            removed.append(fn)
        if removed:
            tag = 'SILINDI' if args.apply else 'silinecek'
            print('%-26s %s (%d): %s' % (repo, tag, len(removed),
                                         ', '.join(removed)))
            total += len(removed)
        else:
            print('%-26s temiz' % repo)
    print('\nTOPLAM: %d  (%s)' % (total, 'silindi' if args.apply else 'rapor only'))
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
