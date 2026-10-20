"""Zenodo web-form yolu icin yapistirilabilir metadata blogu uretir.

Web formunda her depo icin su alanlari doldurman gerekir; bu dosya
degerleri hazir halde verir. Dosya: dist/zenodo/METADATA.txt
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import zenodo_deposit as ZD
from deposit_all import ROOT, REPOS, DIST

out = []
out.append('ZENODO YAYINLAMA -- YAPISTIRILABILIR METADATA')
out.append('=' * 78)
out.append('')
out.append('Adim 1) https://zenodo.org/upload  -> "New upload"')
out.append('Adim 2) "Upload" bolumune dosyayi surukleyin:')
for r in REPOS:
    out.append('         dist/zenodo/%s-v1.zip' % r)
out.append('Adim 3) Asagidaki degerleri form alanlarina yazin.')
out.append('Adim 4) "Publish" -> DOI olusur -> DOI\'yi bana verin,')
out.append('         README/CITATION dosyalarina islemek icin.')
out.append('')
out.append('Alanlar: Resource type = "Software";  Upload type = "Software";')
out.append('        License = "Creative Commons Attribution 4.0 International";')
out.append('        Publish = "Publish" (yoksa "Save draft" + DOI sonra).')
out.append('')

for r in REPOS:
    repo = os.path.join(ROOT, r)
    if not os.path.isdir(repo):
        continue
    body, raw = ZD.read_metadata(repo)
    out.append('=' * 78)
    out.append('DEPO: %s' % r)
    out.append('=' * 78)
    out.append('Title       : %s' % body['title'])
    out.append('Creators    : %s'
               % '; '.join(c['name'] for c in body['creators']))
    out.append('Description :')
    out.append(body['description'])
    out.append('Keywords    : %s' % ', '.join(body['keywords']))
    out.append('License     : %s' % body['license'])
    out.append('Version     : %s' % body['version'])
    out.append('Related URL : %s'
               % body['related_identifiers'][0]['identifier'])
    out.append('File        : dist/zenodo/%s-v1.zip' % r)
    out.append('')

path = os.path.join(DIST, 'METADATA.txt')
os.makedirs(DIST, exist_ok=True)
with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
    fh.write('\n'.join(out))
print('yazildi: %s (%d satir)' % (path, len(out)))
