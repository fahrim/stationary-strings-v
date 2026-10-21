import argparse
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import bib_convert as B

STOP = set(('the and for with from that this are was were has have had not but '
            'its their there here they them then than into out under over '
            'between using used use more most also only other such same each '
            'per via one two three').split())


def sig_words(text):
    w = re.findall(r'[A-Za-z]{3,}', text or '')
    return [x.lower() for x in w if x.lower() not in STOP]


LIGATURES = {
    '\x0b': 'ff', '\x0c': 'fi', '\x0d': 'fl', '\x0e': 'ffi', '\x0f': 'ffl',
    '\x1c': 'fi', '\x1d': 'fl', '\x1e': 'ff', '\x1f': 'ffi',
    '\x13': "''", '\x14': "'", '\x15': '-', '\x16': '--', '\x1a': '--',
}


def fix_ligatures(text):
    """PDF metin cikariminda ligaturlar kontrol karakteri olarak gelir.

    'hyper\\x1cne' -> 'hyperfine'.  Bu olmadan kelime bazli karsilastirma
    tireleme sanilir ve her seferinde yanlis negatif uretir.
    """
    if not text:
        return ''
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return text


def squash(text):
    """Harf/rakam disi her sey silinir.

    LaTeX tirelemesi PDF metin cikariminda 'hyper-fine' gibi bölünmeler
    uretir; bosluksuzlastirma bu yapay bölmeleri ortadan kaldirir.
    """
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def render_bib(bib_path, workdir):
    """refs.bib -> BibTeX (plain stili) -> PDF -> metin.

    natbib bu makinede kurulu degil; klasik `plain` stili her MiKTeX/TexLive
    kurulumunda vardir ve dogrulama icin yeterlidir.
    """
    tex = os.path.join(workdir, 'bibtest.tex')
    src = (r'\documentclass{article}' + '\n'
           r'\usepackage[utf8]{inputenc}' + '\n'
           r'\usepackage[T1]{fontenc}' + '\n'
           r'\begin{document}' + '\n'
           r'\nocite{*}' + '\n'
           r'\bibliographystyle{plain}' + '\n'
           r'\bibliography{' + os.path.splitext(os.path.basename(bib_path))[0]
           + '}' + '\n' + r'\end{document}' + '\n')
    with io.open(tex, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(src)
    for _ in range(2):
        subprocess.run(['pdflatex', '--disable-installer',
                        '-interaction=nonstopmode', 'bibtest.tex'],
                       cwd=workdir, capture_output=True)
    r = subprocess.run(['bibtex', 'bibtest'], cwd=workdir, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    for _ in range(2):
        subprocess.run(['pdflatex', '--disable-installer',
                        '-interaction=nonstopmode', 'bibtest.tex'],
                       cwd=workdir, capture_output=True)
    pdf = os.path.join(workdir, 'bibtest.pdf')
    if not os.path.isfile(pdf):
        return '', (r.stdout or '') + (r.stderr or '')
    try:
        from pypdf import PdfReader
        import logging
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        txt = '\n'.join((p.extract_text() or '') for p in PdfReader(pdf).pages)
        return txt, (r.stdout or '') + (r.stderr or '')
    except Exception as exc:
        return '', str(exc)


def entries_of(repo, rel):
    path = os.path.join(B.ROOT, repo, rel.replace('/', os.sep))
    with io.open(path, encoding='utf-8') as fh:
        tex = fh.read()
    m = B.BIB_RE.search(tex)
    if not m:
        return None, None
    items = [(k, B.parse_entry(k, body))
             for k, body in B.ITEM_RE.findall(m.group(1))]
    return items, B.to_bib(items)


def verify(repo, rel):
    items, bib = entries_of(repo, rel)
    if items is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        with io.open(os.path.join(td, 'refs.bib'), 'w', encoding='utf-8',
                     newline='\n') as fh:
            fh.write(bib)
        rendered, err = render_bib(os.path.join(td, 'refs.bib'), td)
    if not rendered:
        return dict(repo=repo, ok=False, n=len(items), missing=[],
                    error=('render basarisiz: ' + (err or '')[:200]))
    rend = squash(fix_ligatures(rendered))
    missing = []
    for key, f in items:
        title = f.get('title', '')
        if not title:
            continue
        tw = [w.lower() for w in re.findall(r'[A-Za-z]{4,}', title)]
        if not tw:
            continue
        hit = sum(1 for w in tw if w in rend)
        if hit < 0.85 * len(tw):
            missing.append((key, '%d/%d kelime bulundu' % (hit, len(tw))))
    return dict(repo=repo, ok=not missing, n=len(items), missing=missing,
                error='', rendered_words=len(rend))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='refs.bib dosyalarini yaz')
    args = ap.parse_args(argv)
    ok = 0
    for repo, rel in B.PAIRS:
        res = verify(repo, rel)
        if res is None:
            print('%-26s kaynakca yok' % repo)
            continue
        if args.write:
            items, bib = entries_of(repo, rel)
            path = os.path.join(B.ROOT, repo, 'paper', 'refs.bib')
            with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(bib)
        if res['ok']:
            ok += 1
            print('%-26s %2d girdi  TAMAM (tum basliklar render edildi)'
                  % (repo, res['n']))
        else:
            print('%-26s %2d girdi  SORUN: %s %s'
                  % (repo, res['n'], res.get('missing'), res.get('error', '')[:120]))
    print('\n%d/%d makale dogrulandi' % (ok, len(B.PAIRS)))
    return 0 if ok == len(B.PAIRS) else 1


if __name__ == '__main__':
    sys.exit(main())
