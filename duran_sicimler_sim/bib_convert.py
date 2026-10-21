"""thebibliography -> .bib donusturucu + otomatik DOGRULAMA.

Her makalenin elle yazilmis `\\begin{thebibliography}` blokunu BibTeX'e
cevirir ve **dogrular**: .bib ile yeniden render edilen kaynakca metni,
orijinal kaynacka metniyle karsilastirilir (normalize edilmis karakter
dizisi uzerinden en uzun ortak alt dizi / toplam uzunluk orani).

Cikti:
  <repo>/paper/refs.bib
  her repo icin bir dogrulama raporu
"""
import argparse
import io
import os
import re
import subprocess
import sys
import tempfile

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'

PAIRS = [
    ('stationary-strings-xi', 'paper/stationary_strings_XI_framework_EN.tex'),
    ('stationary-strings-xii', 'paper/stationary_strings_XII_galactic_dynamics_EN.tex'),
    ('stationary-strings-xiii', 'paper/stationary_strings_XIII_yang_mills_mass_gap_EN.tex'),
    ('stationary-strings-xiv', 'paper/stationary_strings_XIV_experimental_proof_protocol_EN.tex'),
    ('stationary-strings-l1', 'paper/stationary_strings_L1_na22_EN.tex'),
    ('stationary-strings-l2', 'paper/stationary_strings_L2_ghost_parameter_forensics_EN.tex'),
]

BIB_RE = re.compile(r'\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}', re.S)
ITEM_RE = re.compile(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\Z)', re.S)


def _clean(s):
    s = s.replace('~', ' ').replace('\\&', '&').replace('\\%', '%')
    s = re.sub(r'\\textit\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\emph\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+\s*', ' ', s)
    s = s.replace('{', '').replace('}', '').replace('~', ' ')
    s = re.sub(r'--', '-', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _balanced(text, start):
    """text[start] icinde '{' ile baslayan grubu dondurur: (icerik, bitis_idx)."""
    i = text.find('{', start)
    if i < 0:
        return '', -1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    return '', -1


def _cmd(text, name):
    """\name{...} -> (icerik, son_indeks) ya da ('', -1)."""
    m = re.search(r'\\' + name + r'\s*\{', text)
    if not m:
        return '', -1
    inner, end = _balanced(text, m.end() - 1)
    return inner, end


def to_bibtex_authors(raw):
    """thebibliography yazar metnini BibTeX 'author' alanina cevirir.

    BibTeX kurallari:
      * yazarlar ' and ' ile ayrilir (kaynakta LaTeX '&' kullaniliyor)
      * 'et al.' -> 'others'
      * "Last, F." biciminde iki yazar virgulle ayrilirsa BibTeX bunu
        "Too many commas" hatasina dusurur; araya 'and' girer.
    """
    s = raw
    s = re.sub(r'\\textit\{\s*et al\.?\s*\}', 'et al.', s)
    s = re.sub(r'\bet\s+al\.?', 'others', s, flags=re.I)
    s = s.replace('&', ' and ')
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s*,\s*others', ' and others', s)
    # "A, B., C." -> "A, B. and C."   (virgulle ayrilmis Last-Initial)
    s = re.sub(r'([A-Z][A-Za-z\'\-]+,\s*(?:[A-Z]\.\s*)+),\s*([A-Z][A-Za-z\'\-]+,)',
               r'\1 and \2', s)
    s = re.sub(r',\s*,', ',', s)
    s = re.sub(r'\s+', ' ', s).strip().rstrip(',')
    return s


def parse_entry(key, raw):
    """thebibliography girdisini BibTeX alanlarina cevirir.

    Kaynakca metninde iki bicim var:
      (a) ic ice:  \\textit{Nature \\textbf{405}}, 53--61
      (b) ayri:     \\textit{Nature} \\textbf{405}, 53--61
    Ikisi de desteklenir. Baslik, ilk "venue" isaretinden (\\textit, \\url,
    arXiv:, https://) onceki metindir.
    """
    txt = raw
    fields = {}

    m = re.search(r'arXiv:([A-Za-z0-9./-]+)', txt)
    if m:
        fields['eprint'] = m.group(1).rstrip('.')
        fields['archivePrefix'] = 'arXiv'
    m = re.search(r'doi[:\s]*\s*(10\.\d{4,9}/[^\s,}]+)', txt)
    if m:
        fields['doi'] = m.group(1).rstrip('.')

    m = re.search(r'\((\d{4}[a-z]?)\)', txt)
    if m:
        fields['year'] = m.group(1)
        head, body = txt[:m.start()], txt[m.end():]
    else:
        head, body = txt, txt

    fields['author'] = to_bibtex_authors(_clean(head).rstrip(',. ').strip())

    # basligi kesilecek nokta
    cut = len(body)
    for pat in (r'\\textit\s*\{', r'\\url\s*\{', r'arXiv:', r'https?://'):
        mm = re.search(pat, body)
        if mm and mm.start() < cut:
            cut = mm.start()
    title = _clean(body[:cut]).strip(' .,')
    rest = body[cut:]
    # baslik bos kaldiysa ve govde \textit ile basliyorsa, baslik odur
    if not title:
        inner, end = _cmd(body, 'textit')
        if inner:
            title = _clean(inner).strip(' .,')
            rest = body[end:]

    # journal (\textit) + volume (\textbf) + pages
    j, jend = _cmd(rest, 'textit')
    if j:
        jclean = _clean(j)
        # ic ice bicim: "Nature 405" -> journal + volume
        mvol = re.match(r'^(.*?[A-Za-z\.\)])\s+(\d+[A-Za-z]?)$', jclean)
        if mvol:
            fields['journal'] = mvol.group(1).strip()
            fields['volume'] = mvol.group(2)
        else:
            fields['journal'] = jclean
        after = rest[jend:]
        v, vend = _cmd(after, 'textbf')
        if v and re.match(r'^\s*\d', _clean(v)):
            fields['volume'] = _clean(v)
            after = after[vend:]
        mm = re.match(r'\s*,\s*([0-9]+(?:--[0-9]+)?)', after)
        if mm:
            fields['pages'] = mm.group(1)
    # not: kaynak katalogunda verilen URL / paket bilgisi
    mnote = re.search(r'https?://[^\s,}]+|arXiv:[A-Za-z0-9./-]+', rest)
    if mnote and 'note' not in fields:
        fields['note'] = mnote.group(0).rstrip('.')
    if title:
        fields['title'] = title
    return fields


def to_bib(items):
    out = []
    for key, f in items:
        out.append('@article{%s,' % key)
        order = ['author', 'title', 'journal', 'volume', 'pages', 'year',
                 'eprint', 'archivePrefix', 'doi']
        for k in order:
            v = f.get(k)
            if not v:
                continue
            out.append('  %-13s = {%s},' % (k, v))
        out.append('}\n')
    return '\n'.join(out)


def render_bib(bib_path, workdir):
    """refs.bib -> PDF (natbib + plainnat) ve metin cikar."""
    tex = os.path.join(workdir, 'bibtest.tex')
    with io.open(tex, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\\documentclass{article}\n'
                 '\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n'
                 '\\usepackage{natbib}\n'
                 '\\begin{document}\n\\nocite{*}\n'
                 '\\bibliographystyle{plainnat}\n'
                 '\\bibliography{%s}\n\\end{document}'
                 % os.path.splitext(os.path.basename(bib_path))[0])
    for _ in range(2):
        subprocess.run(['pdflatex', '--disable-installer',
                        '-interaction=nonstopmode', 'bibtest.tex'],
                       cwd=workdir, capture_output=True)
    pdf = os.path.join(workdir, 'bibtest.pdf')
    if not os.path.isfile(pdf):
        return ''
    try:
        from pypdf import PdfReader
        import logging
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        return '\n'.join((p.extract_text() or '') for p in PdfReader(pdf).pages)
    except Exception:
        return ''


def normalize(s):
    s = re.sub(r'[^A-Za-z0-9]+', ' ', s).lower()
    return ' '.join(s.split())


def similarity(a, b):
    """En uzun ortak alt dizi / max(len) -> 0..1 arasi benzerlik."""
    A, B = normalize(a), normalize(b)
    if not A or not B:
        return 0.0
    prev = [0] * (len(B) + 1)
    best = 0
    for i in range(1, len(A) + 1):
        cur = [0] * (len(B) + 1)
        for j in range(1, len(B) + 1):
            if A[i - 1] == B[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best / max(len(A), len(B))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='.bib dosyalarini yaz')
    args = ap.parse_args(argv)

    total_ok = 0
    total = 0
    for repo, rel in PAIRS:
        path = os.path.join(ROOT, repo, rel.replace('/', os.sep))
        with io.open(path, encoding='utf-8') as fh:
            tex = fh.read()
        m = BIB_RE.search(tex)
        if not m:
            print('%-26s kaynakca bulunamadi' % repo)
            continue
        items = [(k, parse_entry(k, body)) for k, body in ITEM_RE.findall(m.group(1))]
        bib = to_bib(items)
        outdir = os.path.join(ROOT, repo, 'paper')
        bibpath = os.path.join(outdir, 'refs.bib')
        if args.write:
            with io.open(bibpath, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(bib)

        with tempfile.TemporaryDirectory() as td:
            tmpbib = os.path.join(td, 'refs.bib')
            with io.open(tmpbib, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(bib)
            rendered = render_bib(tmpbib, td)
        orig = _clean(m.group(1))
        orig = re.sub(r'\\bibitem\{[^}]*\}', ' ', orig)
        sim = similarity(orig, rendered)
        ok = sim >= 0.80
        total_ok += 1 if ok else 0
        total += 1
        print('%-26s %d girdi  benzerlik=%.3f  %s'
              % (repo, len(items), sim, 'OK' if ok else 'KONTROL GEREKLI'))
        if not ok:
            print('    ORJ : %s' % normalize(orig)[:220])
            print('    BIB : %s' % normalize(rendered)[:220])
    print('\n%d/%d makale dogrulandi' % (total_ok, total))
    return 0 if total_ok == total else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
