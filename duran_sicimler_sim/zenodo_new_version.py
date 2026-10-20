"""Yayimlanmis bir Zenodo kaydini yeni surum (v2) olarak guncelle.

Akis (dogrulanmis 21.10.2026):
  1. POST /api/deposit/depositions/<recid>/actions/newversion  -> v2 taslagi
     (AYNI concept DOI, surum DOI'si ...v2 olur; eski DOI bozulmaz)
  2. PUT  /api/deposit/depositions/<newid>                      -> metadata
  3. eski dosyayi sil, yenisini yukle (arsiv guncellendiyse)
  4. POST /api/deposit/depositions/<newid>/actions/publish     -> yayinla

Kullanim:
  python -W ignore zenodo_new_version.py --recid 23517479 \
      --repo <depo dizini> --token <TOKEN> [--dry-run]
"""
import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import zenodo_deposit as ZD

API = 'https://zenodo.org/api'


def req(url, token, method='GET', data=None, headers=None, raw=False):
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header('Authorization', 'Bearer %s' % token)
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            body = resp.read()
            return resp.status, (body if raw else json.loads(body or b'{}'))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode('utf-8', 'replace')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--recid', required=True, help='mevcut kayit id')
    ap.add_argument('--repo', required=True, help='depo dizini')
    ap.add_argument('--token', required=True)
    ap.add_argument('--version', default='v2')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    name = os.path.basename(repo)
    metadata, _ = ZD.read_metadata(repo)

    status, rec = req('%s/records/%s' % (API, args.recid), args.token)
    if status != 200:
        raise SystemExit('kayit okunamadi (HTTP %s)' % status)
    print('MEVCUT KAYIT %s' % args.recid)
    print('  baslik : %s' % rec.get('metadata', {}).get('title'))
    print('  doi    : %s' % rec.get('doi'))
    print('  yeni baslik: %s' % metadata['title'])
    if args.dry_run:
        print('[HAZIRLIK] ag erisimi yapilmadi.')
        return 0

    # 1) yeni surum taslagi
    #    Onceden acilmis bir v2 taslagi varsa onu kullan: Zenodo ayni kayit
    #    icin ikinci bir newversion'a izin vermiyor ("remove all files first")
    #    ve eski taslagi bosaltmak yeni surum olusturmaktan temizdir.
    cur_title = rec.get('metadata', {}).get('title', '')
    st, lst = req('%s/deposit/depositions?size=50' % API, args.token)
    newid = None
    if st == 200 and isinstance(lst, list):
        for d in lst:
            if d.get('state') in ('done', 'published'):
                continue
            if d.get('metadata', {}).get('title', '') == cur_title:
                newid = d['id']
                print('\n1) mevcut v2 taslagi bulundu: %s' % newid)
                break
    if newid is None:
        st, nv = req('%s/deposit/depositions/%s/actions/newversion'
                     % (API, args.recid), args.token, method='POST',
                     data=b'{}', headers={'Content-Type': 'application/json'})
        if st not in (200, 201):
            raise SystemExit('newversion basarisiz (HTTP %s): %s'
                             % (st, str(nv)[:300]))
        newid = nv['id']
        print('\n1) yeni surum taslagi: %s' % newid)

    # 1b) taslaktaki dosyalari temizle (newversion bunu istiyor)
    st, files = req('%s/deposit/depositions/%s/files' % (API, newid), args.token)
    for f in (files if isinstance(files, list) else []):
        fid = f.get('id') or (f.get('links', {}).get('self', '').rsplit('/', 1)[-1])
        if fid:
            req('%s/deposit/depositions/%s/files/%s' % (API, newid, fid),
                args.token, method='DELETE')
    print('   eski dosyalar temizlendi')

    # 2) metadata  (PUT govdesi {"metadata": {...}} seklinde olmali)
    md = dict(metadata)
    md.setdefault('access_right', 'open')
    md['version'] = args.version
    st, body = req('%s/deposit/depositions/%s' % (API, newid), args.token,
                   method='PUT',
                   data=json.dumps({'metadata': md}).encode('utf-8'),
                   headers={'Content-Type': 'application/json'})
    if st not in (200, 201):
        raise SystemExit('metadata guncellenemedi (HTTP %s): %s'
                         % (st, str(body)[:400]))
    print('2) metadata guncellendi (baslik + aciklama + lisans + anahtar)')

    # 3) arsivi yenile
    zip_path = os.path.join(HERE, '_tmp_%s.zip' % name)
    _, nfiles, zsize = ZD.build_zip(repo, zip_path)
    with open(zip_path, 'rb') as fh:
        content = fh.read()
    mbody, ctype = ZD.multipart_body('file', 'fahrim/%s-%s.zip' % (name, args.version),
                                     content)
    st, body = req('%s/deposit/depositions/%s/files' % (API, newid), args.token,
                   method='POST', data=mbody, headers={'Content-Type': ctype})
    if st not in (200, 201):
        raise SystemExit('dosya yuklenemedi (HTTP %s): %s'
                         % (st, str(body)[:300]))
    os.remove(zip_path)
    print('3) arsiv yenilendi (%d dosya, %.1f kB)' % (nfiles, zsize / 1024.0))

    # 4) yayinla
    st, body = req('%s/deposit/depositions/%s/actions/publish' % (API, newid),
                   args.token, method='POST', data=b'{}',
                   headers={'Content-Type': 'application/json'})
    if st not in (200, 201, 202):
        raise SystemExit('yayinlanamadi (HTTP %s): %s' % (st, str(body)[:300]))
    out = body if isinstance(body, dict) else (json.loads(body) if body else {})
    print('4) YAYINLANDI')
    print('   doi    : %s' % out.get('doi'))
    print('   surum  : %s' % out.get('metadata', {}).get('version'))
    print('   konsept: %s' % out.get('conceptdoi'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
