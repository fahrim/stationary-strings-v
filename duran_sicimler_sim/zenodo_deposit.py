"""Zenodo yukleme araci: tek komutla DOI al + DOI'yi depoya isle.

Mevcut seri kayitlarinin kaliplarina uyar:
  resource_type : Software
  dosya         : tek zip  ->  <owner>/<repo>-<tag>.zip
  license       : cc-by-4.0

Kullanim:
  python -W ignore zenodo_deposit.py --repo <repo dizini> \
      --token <ZENODO_TOKEN> --name "Stationary Strings XI: ..." \
      --version v1 [--dry-run] [--no-publish]

Token verilmezse yalnizca HAZIRLIK yapilir (zip uretilir, metadata hazirlanir,
DOI isaretleri hazirlanir) ve ag erisimi denenmez.

Dikkat: yazma islemleri icin PowerShell DEGIL, bu arac kullanilir
(AGENTS.md: kodlama bozulmamasi).
"""
import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile

ZENODO_API = 'https://zenodo.org/api'
SKIP_DIRS = {'.git', '__pycache__', '.obsidian', '.ipynb_checkpoints',
              '.pytest_cache', '.mypy_cache'}
SKIP_EXT = ('.aux', '.log', '.out', '.synctex.gz', '.pyc', '.buildlog')
DOI_RE = re.compile(r'10\.5281/zenodo\.\d+')


def _git(repo, *args):
    out = subprocess.run(['git', '-C', repo] + list(args),
                         capture_output=True, text=True,
                         encoding='utf-8', errors='replace')
    return out.stdout.strip()


def build_zip(repo, out_path, exclude=()):
    """Depoyu (geçici dosyalar hariç) zip'ler. Deterministik: sıralı giriş.

    Cikti zip'i repo icinde ise kendini icine almaz.
    """
    out_abs = os.path.abspath(out_path)
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo).replace(os.sep, '/')
            if rel.startswith('.git/'):
                continue
            if rel.endswith(SKIP_EXT):
                continue
            if os.path.abspath(full) == out_abs:
                continue
            if any(rel == e or rel.startswith(e.rstrip('/') + '/')
                   for e in exclude):
                continue
            files.append((full, rel))
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for full, rel in files:
            zf.write(full, rel)
    return out_path, len(files), os.path.getsize(out_path)


def read_metadata(repo):
    """`.zenodo.json` -> Zenodo metadata gövdesi."""
    path = os.path.join(repo, '.zenodo.json')
    if not os.path.isfile(path):
        raise SystemExit('.zenodo.json bulunamadi: %s' % path)
    with io.open(path, encoding='utf-8') as fh:
        meta = json.load(fh)
    body = {
        'title': meta.get('title', ''),
        'description': meta.get('description', ''),
        'creators': [{'name': c.get('name')} for c in meta.get('creators', [])],
        'license': (meta.get('license') or {}).get('id', 'cc-by-4.0').lower(),
        'keywords': meta.get('keywords', []),
        # Modern Zenodo deposit API: "upload_type" secimli koten ile
        # "resource_type" TURETILIR; elle verilmesi HTTP 400 verir.
        'upload_type': 'software',
        'publication_date': meta.get('publication_date', '2026-10-20'),
        'version': meta.get('version', 'v1'),
    }
    repo_url = meta.get('related_identifiers')
    if repo_url is None:
        body['related_identifiers'] = [{
            'identifier': 'https://github.com/fahrim/%s'
                           % os.path.basename(os.path.abspath(repo)),
            'relation': 'isSupplementTo',
            'scheme': 'url',
        }]
    return body, meta


# ------------------------------------------------------------------ ag islemleri

def _request(url, token, method='GET', data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header('Authorization', 'Bearer %s' % token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart_body(field, filename, content, mime='application/zip'):
    boundary = '----opencode%s' % uuid.uuid4().hex
    parts = []
    parts.append(('--%s\r\n' % boundary).encode())
    parts.append(('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                  % (field, filename)).encode())
    parts.append(('Content-Type: %s\r\n\r\n' % mime).encode())
    parts.append(content)
    parts.append(('\r\n--%s--\r\n' % boundary).encode())
    return b''.join(parts), 'multipart/form-data; boundary=%s' % boundary


def deposit(token, name, version, zip_path, metadata, publish=True):
    """Taslak olustur -> dosya yukle -> (varsa) yayinla. DOI dondurur."""
    filename = 'fahrim/%s-%s.zip' % (name, version)
    payload = {'metadata': metadata,
               'files': [{'key': filename, 'size': os.path.getsize(zip_path)}]}
    status, body = _request(
        ZENODO_API + '/deposit/depositions', token, method='POST',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    if status not in (200, 201):
        raise SystemExit('deposition olusturulamadi (HTTP %s): %s'
                         % (status, body[:500].decode('utf-8', 'replace')))
    dep = json.loads(body)
    dep_id = dep['id']

    with open(zip_path, 'rb') as fh:
        content = fh.read()
    mbody, ctype = multipart_body('file', filename, content)
    status, body = _request(
        '%s/deposit/depositions/%s/files' % (ZENODO_API, dep_id), token,
        method='POST', data=mbody, headers={'Content-Type': ctype})
    if status not in (200, 201):
        raise SystemExit('dosya yuklenemedi (HTTP %s): %s'
                         % (status, body[:500].decode('utf-8', 'replace')))

    if not publish:
        return None, dep_id, dep.get('links', {}).get('html')

    status, body = _request(
        '%s/deposit/depositions/%s/actions/publish' % (ZENODO_API, dep_id),
        token, method='POST', data=b'{}',
        headers={'Content-Type': 'application/json'})
    # Zenodo yayinlama islemini ASENKRON yurutur: HTTP 202 (Accepted) +
    # govdede DOI doner. 200/201 de kabul edilir.
    if status not in (200, 201, 202):
        raise SystemExit('yayinlanamadi (HTTP %s): %s'
                         % (status, body[:500].decode('utf-8', 'replace')))
    rec = json.loads(body)
    doi = rec.get('doi')
    if not doi:
        # 202'de govde tam kayit donmeyebilir; kaydi sorgula
        status2, body2 = _request('%s/records/%s' % (ZENODO_API, dep_id), token)
        if status2 == 200:
            doi = json.loads(body2).get('doi')
    if not doi:
        raise SystemExit('yayinlandi ama DOI alinamadi (deposition id=%s)' % dep_id)
    return doi, dep_id, rec.get('links', {}).get('html') or \
        'https://zenodo.org/records/%s' % dep_id


# ------------------------------------------------------------------ DOI isleme

DOI_NOTE_EN = ('**Status:** Unpublished preprint (v1). Not peer-reviewed. '
               '*Zenodo DOI pending deposit.*')
DOI_NOTE_TR = ('**Durum:** Yayımlanmamış ön-baskı (v1). Hakem denetiminden '
               'geçmemiştir. *Zenodo DOI\'si yüklemeyi bekliyor.*')
BADGE = ('[![DOI](https://img.shields.io/badge/DOI-10.5281%%2Fzenodo.%s-'
         'blue.svg)](https://doi.org/10.5281/zenodo.%s)')
CITE_EN = ('```bibtex\n@misc{meral%s,\n  author = {Fahri Meral},\n'
           '  title  = {%s},\n  year   = {2026},\n'
           '  doi    = {10.5281/zenodo.%s},\n'
           '  url    = {https://doi.org/10.5281/zenodo.%s}\n}\n```')
CITE_TR = ('```bibtex\n@misc{meral%s,\n  author = {Fahri Meral},\n'
           '  title  = {%s},\n  year   = {2026},\n'
           '  doi    = {10.5281/zenodo.%s},\n'
           '  url    = {https://doi.org/10.5281/zenodo.%s}\n}\n```')


def stamp_doi(repo, doi, title, key):
    """DOI'yi README/README_TR/CITATION.cff icine yazar."""
    num = doi.rsplit('.', 1)[-1]
    changed = []

    for name, note, cite in (('README.md', DOI_NOTE_EN, CITE_EN),
                             ('README_TR.md', DOI_NOTE_TR, CITE_TR)):
        path = os.path.join(repo, name)
        if not os.path.isfile(path):
            continue
        with io.open(path, encoding='utf-8') as fh:
            t = fh.read()
        orig = t
        t = t.replace(note, note.replace('*Zenodo DOI pending deposit.*',
                                         '*%s*' % (BADGE % (num, num)))
                      .replace("*Zenodo DOI'si yüklemeyi bekliyor.*",
                               '*%s*' % (BADGE % (num, num))))
        if 'img.shields.io/badge/DOI' not in t:
            t = t.replace('**L2:', BADGE % (num, num) + '\n\n**L2:', 1)
            t = t.replace('**PV-1', BADGE % (num, num) + '\n\n**PV-1', 1)
            t = t.replace('**PV-2', BADGE % (num, num) + '\n\n**PV-2', 1)
        # bibtex govdesine doi ekle
        t = re.sub(r'(```bibtex\n@misc\{meral[^\n]*\n(?:.*\n)*?)(\}\n```)',
                   lambda m: m.group(1)
                   + '  doi    = {%s},\n' % doi + m.group(2), t, count=1)
        if t != orig:
            with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(t)
            changed.append(name)

    cff = os.path.join(repo, 'CITATION.cff')
    if os.path.isfile(cff):
        with io.open(cff, encoding='utf-8') as fh:
            t = fh.read()
        orig = t
        if '\ndoi:' not in t:
            t = t.replace('date-released:', 'doi: %s\ndate-released:' % doi, 1)
        if t != orig:
            with io.open(cff, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(t)
            changed.append('CITATION.cff')
    return changed


# ------------------------------------------------------------------ ana akis

def main(argv=None):
    ap = argparse.ArgumentParser(description='Zenodo DOI yukleme araci')
    ap.add_argument('--repo', required=True, help='depo dizini')
    ap.add_argument('--token', default=None, help='Zenodo API tokeni')
    ap.add_argument('--name', default=None, help='kayit adi (or. stationary-strings-xi)')
    ap.add_argument('--version', default='v1')
    ap.add_argument('--title', default=None, help='kayit basligi')
    ap.add_argument('--publish', action='store_true', default=True)
    ap.add_argument('--no-publish', dest='publish', action='store_false')
    ap.add_argument('--dry-run', action='store_true',
                    help='yalnizca hazirlik (ag erisimi yok)')
    ap.add_argument('--key', default=None, help='bibtex anahtar sozcugu')
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    name = args.name or os.path.basename(repo)
    metadata, raw_meta = read_metadata(repo)
    if args.title:
        metadata['title'] = args.title
    metadata['version'] = args.version

    tmpdir = tempfile.mkdtemp(prefix='zenodo_')
    zip_path = os.path.join(tmpdir, '%s-%s.zip' % (name, args.version))
    zip_path, nfiles, zsize = build_zip(repo, zip_path)
    print('hazirlik : %s' % zip_path)
    print('  dosya   : %d  (%d bayt)' % (nfiles, zsize))
    print('  baslik  : %s' % metadata['title'][:90])
    print('  tur     : %s' % metadata['upload_type'])

    if args.dry_run or not args.token:
        print('\n[HAZIRLIK] ag erisimi denenmedi (token yok veya --dry-run).')
        print('  Yayinlamak icin: --token <TOKEN> ile tekrar calistirin.')
        print('  Zip korundu: %s' % zip_path)
        return 0

    doi, dep_id, html = deposit(args.token, name, args.version, zip_path,
                                metadata, publish=args.publish)
    if doi:
        print('\nDOI      : %s' % doi)
        print('Kayit    : %s' % html)
        changed = stamp_doi(repo, doi, metadata['title'], args.key)
        print('Islenen  : %s' % (', '.join(changed) if changed else '(yok)'))
        print('\nSONRA    : git add -A && git commit -m "doi: %s" && git push' % doi)
    else:
        print('\nTASLAK ID: %s  (yayinlanmadi)' % dep_id)
        print('  Yayinlamak icin: --token ile --no-publish olmadan calistirin.')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
