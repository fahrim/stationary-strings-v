"""Dergi formu icin alan denetimi (abstract kelime siniri vb.).

A&A Research Note / Letter: ozet 150 kelimeyi gecmemelidir.
Bu arac ozeti metinden sayar, LaTeX kacislarini temizler ve Sozluk
denetimlerini yapar (ozet icinde yazilmis olmamasi gereken seyler).

Kullanim:
  python -W ignore abstract_check.py --file <dosya> --max-words 150
"""
import argparse
import io
import os
import re
import sys

# Ozette olmamasi gerekenler (dergi formu ayri alanlari var)
FORBIDDEN = [
    (r'\\citep?\b', 'LaTeX atif komutu'),
    (r'\\ref\b', 'LaTeX referans komutu'),
    (r'\\label\b', 'LaTeX etiket komutu'),
    (r'\\footnote\b', 'dipnot'),
    (r'\\begin\{(figure|table|equation)', 'ortam'),
    (r'\\[a-zA-Z]+', 'LaTeX komutu'),
    (r'\$[^$]*\$', 'math (ozette matematik kullanmayin)'),
    (r'https?://', 'URL (ayri alana yazin)'),
]


def strip_tex(s):
    s = re.sub(r'\\textit\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\mathcal\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = s.replace('{', '').replace('}', '').replace('~', ' ')
    s = s.replace('--', '-').replace('&', ' and ')
    s = re.sub(r'[^\w\s\u00b0\u00b5\u2013\u2014\.\,\;\:\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def count_words(text):
    # A&A sayimi: bolme isaretleri, tireli kelimeler, sayilar birer "kelime"
    t = strip_tex(text)
    toks = [x for x in re.split(r'[\s]+', t) if x]
    # Bilesik ifadeleri (orn. 511 keV) tek say: boslukla ayrilmis her parca
    # sayilir; A&A'nin sayimi da kelime sayar, oyle kabul ediyoruz.
    return len(toks)


def check(path, max_words, label=''):
    with io.open(path, encoding='utf-8') as fh:
        raw = fh.read()
    # ozet: dosyanin icindeki "## Abstract" veya "## Ozet" bolumunden
    m = re.search(r'##+\s*(?:Abstract|Özet|Ozet)\s*\n(.*?)(?=\n##|\Z)',
                  raw, re.S)
    if m:
        text = m.group(1)
        where = 'dosyadaki ozet bolumu'
    else:
        text = raw
        where = 'dosyanin tamami'
    n = count_words(text)
    print('=' * 70)
    print('OZET DENETIMI %s' % (label or os.path.basename(path)))
    print('  kaynak        : %s' % where)
    print('  kelime sayisi : %d  (sinir: %d)' % (n, max_words))
    ok = n <= max_words
    print('  sonuc         : %s' % ('GECTI' if ok else 'ASIMDA'))
    bad = []
    for pat, name in FORBIDDEN:
        for mm in re.finditer(pat, raw if where.endswith('tamami') else text):
            bad.append((name, mm.group(0)[:40]))
    if bad:
        print('  uyarilar:')
        for nm, sn in bad[:8]:
            print('     - %s: %r' % (nm, sn))
    else:
        print('  format uyarisi: yok')
    print('=' * 70)
    return ok, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    ap.add_argument('--max-words', type=int, default=150)
    ap.add_argument('--label', default='')
    args = ap.parse_args(argv)
    ok, n = check(args.file, args.max_words, args.label)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
