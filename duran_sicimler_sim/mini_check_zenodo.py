"""Zenodo yukleme aracinin bagimsiz kendi denetimi (ag erisimi YOK).

Kontrol listesi:
  Z1 zip determinizmi  : ayni girdi -> ayni dosya listesi; .git/.aux/.log haric
  Z2 metadata eslemesi : .zenodo.json -> Zenodo govdesi (tur, lisans, anahtar)
  Z3 DOI isleme        : README/README_TR rozeti + bibtex doi + CITATION.cff doi
  Z4 esleme olmayan girdi: DOI_RE yakalama, bos .zenodo.json hata verir
"""
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import zenodo_deposit as ZD

PASS = [0, 0]


def check(name, cond, detail=''):
    PASS[1] += 1
    if cond:
        PASS[0] += 1
        print('PASS %-28s %s' % (name, detail))
    else:
        print('FAIL %-28s %s' % (name, detail))


def make_repo(files):
    tmp = tempfile.mkdtemp(prefix='zen_l2_')
    for rel, content in files:
        path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
    return tmp


def Z1_zip():
    repo = make_repo([
        ('a.py', 'x = 1\n'),
        ('sub/b.md', '# b\n'),
        ('build.aux', 'junk\n'),
        ('run.log', 'junk\n'),
        ('.git/config', 'junk\n'),
    ])
    # Cikti zip'i repo DISINDA olmali (arac da boyle kullaniyor): aksi
    # halde ikinci zip ilk zip'i icine alir ve determinizm bozulur.
    out = tempfile.mkdtemp(prefix='zen_zip_')
    z1, n1, _ = ZD.build_zip(repo, os.path.join(out, 'o1.zip'))
    z2, n2, _ = ZD.build_zip(repo, os.path.join(out, 'o2.zip'))
    import zipfile
    names1 = sorted(zipfile.ZipFile(z1).namelist())
    names2 = sorted(zipfile.ZipFile(z2).namelist())
    check('Z1_zip_determinism',
          names1 == names2 and 'a.py' in names1 and 'sub/b.md' in names1
          and not any(n.endswith(('.aux', '.log')) for n in names1)
          and not any(n.startswith('.git') for n in names1),
          'dosyalar=%s' % names1)

    # Cikti zip'i repo icindeyse kendini icine almamali
    inside = os.path.join(repo, 'inside.zip')
    z3, n3, _ = ZD.build_zip(repo, inside)
    names3 = sorted(zipfile.ZipFile(z3).namelist())
    check('Z1b_zip_self_exclusion', 'inside.zip' not in names3,
          'dosyalar=%s' % names3)


def Z2_metadata():
    repo = make_repo([
        ('.zenodo.json', json.dumps({
            'title': 'Test Title',
            'description': 'desc',
            'creators': [{'name': 'Meral, Fahri'}],
            'license': {'id': 'CC-BY-4.0'},
            'keywords': ['a', 'b'],
        }, ensure_ascii=False)),
    ])
    body, _ = ZD.read_metadata(repo)
    ok = (body['title'] == 'Test Title'
          and body['upload_type'] == 'software'
          and 'resource_type' not in body
          and body['license'] == 'cc-by-4.0'
          and body['creators'] == [{'name': 'Meral, Fahri'}]
          and body['keywords'] == ['a', 'b'])
    check('Z2_metadata_mapping', ok,
          'upload_type=%s lisans=%s' % (body['upload_type'], body['license']))


def Z3_stamp():
    repo = make_repo([
        ('README.md',
         '# T\n\n> **Status:** Unpublished preprint (v1). Not peer-reviewed. '
         '*Zenodo DOI pending deposit.*\n\n```bibtex\n@misc{meral2026x,\n'
         '  author = {Fahri Meral},\n  year   = {2026},\n}\n```\n'),
        ('README_TR.md',
         '# T\n\n> **Durum:** Yayımlanmamış ön-baskı (v1). Hakem denetiminden '
         'geçmemiştir. *Zenodo DOI\'si yüklemeyi bekliyor.*\n'),
        ('CITATION.cff', 'cff-version: 1.2.0\ndate-released: 2026-10-20\n'),
    ])
    changed = ZD.stamp_doi(repo, '10.5281/zenodo.12345678', 'T', 'x')
    en = io.open(os.path.join(repo, 'README.md'), encoding='utf-8').read()
    tr = io.open(os.path.join(repo, 'README_TR.md'), encoding='utf-8').read()
    cff = io.open(os.path.join(repo, 'CITATION.cff'), encoding='utf-8').read()
    ok = (len(changed) == 3
          and 'img.shields.io/badge/DOI' in en and 'img.shields.io/badge/DOI' in tr
          and 'doi    = {10.5281/zenodo.12345678}' in en
          and 'doi: 10.5281/zenodo.12345678' in cff
          and 'pending deposit' not in en)
    check('Z3_doi_stamping', ok, 'islenen=%s' % ','.join(changed))


def Z4_bad_input():
    repo = make_repo([('x.txt', 'x')])
    raised = False
    try:
        ZD.read_metadata(repo)
    except SystemExit:
        raised = True
    check('Z4_missing_zenodo_json', raised,
          '.zenodo.json yoksa hata veriyor')


if __name__ == '__main__':
    print('=' * 68)
    print('ZENODO ARACI MINI DENETIMI (ag erisimi yok)')
    print('=' * 68)
    Z1_zip()
    Z2_metadata()
    Z3_stamp()
    Z4_bad_input()
    print('-' * 68)
    print('MINI %d/%d PASS' % (PASS[0], PASS[1]))
    sys.exit(0 if PASS[0] == PASS[1] else 1)
