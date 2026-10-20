"""Tum yeni artifact'lar icin Zenodo DOI yükleme (TEK komut, TEK token).

Kullanim (kendi terminalinde, token ile):
  python -W ignore deposit_all.py --token <ZENODO_TOKEN> [--no-push]

Her depo icin:
  1. zip kurar (dist/zenodo/<repo>-v1.zip)
  2. Zenodo deposition olusturur, dosyayi yukler, YAYINLAR
  3. DOI'yi README.md / README_TR.md (rozet) + bibtex + CITATION.cff icine yazar
  4. git add + commit (+ istenirse push)

Token verilmezse yalnizca ZIP'ler uretilir (web form yolu icin hazirlik).
"""
import argparse
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import zenodo_deposit as ZD

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
REPOS = ['stationary-strings-xi', 'stationary-strings-xii',
         'stationary-strings-xiii', 'stationary-strings-xiv',
         'stationary-strings-l1', 'stationary-strings-l2']
DIST = os.path.join(ROOT, 'dist', 'zenodo')


def git(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args),
                          capture_output=True, text=True,
                          encoding='utf-8', errors='replace')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--token', default=None)
    ap.add_argument('--repos', nargs='*', default=REPOS)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--no-push', dest='push', action='store_false', default=True)
    ap.add_argument('--zip-only', action='store_true',
                    help='yalnizca zip uret (web form yolu)')
    args = ap.parse_args(argv)

    os.makedirs(DIST, exist_ok=True)
    print('=' * 74)
    print('ZENODO DOI YUKLEME -- %d depo' % len(args.repos))
    print('=' * 74)

    results = []
    for name in args.repos:
        repo = os.path.join(ROOT, name)
        if not os.path.isdir(repo):
            print('ATLANDI (yok): %s' % name)
            continue
        metadata, _ = ZD.read_metadata(repo)
        zip_path = os.path.join(DIST, '%s-%s.zip' % (name, args.version))
        _, nfiles, zsize = ZD.build_zip(repo, zip_path)
        print('\n[%s] %d dosya, %.1f kB -> %s'
              % (name, nfiles, zsize / 1024.0, os.path.basename(zip_path)))

        if args.zip_only or not args.token:
            print('   ZIP hazir (yayinlanmadi).')
            results.append((name, None, 'zip-only'))
            continue

        try:
            doi, dep_id, html = ZD.deposit(args.token, name, args.version,
                                           zip_path, metadata, publish=True)
        except SystemExit as exc:
            print('   HATA: %s' % exc)
            results.append((name, None, 'hata'))
            continue
        if not doi:
            print('   TASLAK ID: %s' % dep_id)
            results.append((name, None, 'taslak'))
            continue

        changed = ZD.stamp_doi(repo, doi, metadata['title'], name)
        msg = 'doi: %s (%s)' % (doi, name)
        git(repo, 'add', '-A')
        git(repo, 'commit', '-m', msg)
        pushed = False
        if args.push:
            r = git(repo, 'push', 'origin', 'main')
            pushed = (r.returncode == 0)
        print('   DOI : %s' % doi)
        print('   URL : %s' % html)
        print('   yazildi: %s | commit: %s | push: %s'
              % (','.join(changed), 'ok', 'ok' if pushed else 'BASARISIZ'))
        results.append((name, doi, 'yayinlandi' if pushed else 'yayinlandi(push yok)'))

    print('\n' + '=' * 74)
    print('%-26s %-26s %s' % ('DEPO', 'DOI', 'DURUM'))
    print('-' * 74)
    for name, doi, status in results:
        print('%-26s %-26s %s' % (name, doi or '-', status))
    print('=' * 74)
    if not args.token or args.zip_only:
        print('\nZiplere: %s' % DIST)
        print('Yayinlama icin token ile tekrar calistirin (veya web formunu kullanin).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
