"""DOI alanini bibtex bloklarina DOGRU virgullarla yeniden yazar.

Mevcut hata: `url = {...}` satirinda virgul yokken `doi` satiri eklendigi icin
blok gecersiz BibTeX oluyor. Bu betik blogu normalize eder:
  - mevcut `doi` satiri silinir
  - `doi` satiri kapanis braces'ten once, dogru virgulle eklenir
  - son giris satirinda virgul garanti edilir
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from deposit_all import ROOT, REPOS
from stamp_existing import RECORDS

BIB_RE = re.compile(r'```bibtex\n(.*?)```', re.S)
DOI_LINE = re.compile(r'^\s*doi\s*=')


def fix_block(block, doi):
    lines = block.rstrip('\n').split('\n')
    # mevcut doi satirini at
    lines = [ln for ln in lines if not DOI_LINE.match(ln)]
    # bos bir son satir varsa kaldir
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return block
    # kapanis satirini ayir
    close = lines[-1]
    body = lines[:-1]
    # son giris satirinda virgul garantisi
    for i in range(len(body) - 1, -1, -1):
        if body[i].strip():
            if not body[i].rstrip().endswith(','):
                body[i] = body[i].rstrip() + ','
            break
    body.append('  doi    = {%s},' % doi)
    return '\n'.join(body + [close]) + '\n'


def main():
    for name in REPOS:
        recid = RECORDS.get(name)
        if not recid:
            continue
        doi = '10.5281/zenodo.%d' % recid
        repo = os.path.join(ROOT, name)
        for fn in ('README.md', 'README_TR.md'):
            path = os.path.join(repo, fn)
            if not os.path.isfile(path):
                continue
            with io.open(path, encoding='utf-8') as fh:
                t = fh.read()
            new = BIB_RE.sub(lambda m: '```bibtex\n' + fix_block(m.group(1), doi) + '```', t)
            if new != t:
                with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
                    fh.write(new)
                print('duzeltildi: %s/%s' % (name, fn))
    return 0


if __name__ == '__main__':
    sys.exit(main())
