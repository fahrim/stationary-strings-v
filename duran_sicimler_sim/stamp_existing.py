"""Yayinlanmis (yoksa) DOI'leri depolara isle -- YENI DEPOSITION OLUSTURMAZ.

Zenodo'da zaten yayinlanmis kayitlarin DOI'sini README/README_TR (rozet),
bibtex ve CITATION.cff icine yazar; git commit + push yapar.

Kullanim:
  python -W ignore stamp_existing.py            # varsayilan 6 depo
  python -W ignore stamp_existing.py --no-push
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import zenodo_deposit as ZD
from deposit_all import ROOT, REPOS

# Zenodo kayit id -> depo
RECORDS = {
    'stationary-strings-xi': 23517470,
    'stationary-strings-xii': 23517474,
    'stationary-strings-xiii': 23517479,
    'stationary-strings-xiv': 23517485,
    'stationary-strings-l1': 23517487,
    'stationary-strings-l2': 23517495,
}


def git(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args),
                          capture_output=True, text=True,
                          encoding='utf-8', errors='replace')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-push', dest='push', action='store_false', default=True)
    args = ap.parse_args(argv)

    print('=' * 74)
    print('DOI ISLEME (yayinlanmis kayitlar)')
    print('=' * 74)
    for name in REPOS:
        recid = RECORDS.get(name)
        if not recid:
            print('ATLANDI: %s' % name)
            continue
        repo = os.path.join(ROOT, name)
        doi = '10.5281/zenodo.%d' % recid
        changed = ZD.stamp_doi(repo, doi, '', name)
        git(repo, 'add', '-A')
        r = git(repo, 'commit', '-m', 'doi: %s (%s)' % (doi, name))
        pushed = False
        if args.push:
            p = git(repo, 'push', 'origin', 'main')
            pushed = (p.returncode == 0)
        print('%-26s %-28s yazildi=%-38s push=%s'
              % (name, doi, ','.join(changed) or '-',
                 'ok' if pushed else 'BASARISIZ'))
        if r.returncode != 0 and 'nothing to commit' not in (r.stdout + r.stderr):
            print('    commit: %s' % (r.stderr or r.stdout)[:160].strip())
    return 0


if __name__ == '__main__':
    sys.exit(main())
