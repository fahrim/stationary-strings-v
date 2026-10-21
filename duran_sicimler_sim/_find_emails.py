import gzip
import io
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# SS-XIII (hep-lat) onaycisi adaylari: bizim kendi kaynak listemizden
# (stationary-strings-xiii/paper/refs.bib) -> arXiv kaynaklari
CANDIDATES = [
    ('2411.16507', 'Teper (RHUL) - Towards an Effective String Theory for the flux tube'),
    ('2312.03855', 'Athenodorou - Confining Strings and Glueballs in ZN Gauge Theories'),
    ('2411.03435', 'Athenodorou/Teper - Confining Strings and the Worldsheet Axion'),
    ('2411.16507', 'Athenodorou+Teper (flop e-posta icin)'),
]

MAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

for aid, who in CANDIDATES:
    try:
        req = urllib.request.Request('https://arxiv.org/src/' + aid,
                                     headers={'User-Agent': 'Mozilla/5.0'})
        raw = urllib.request.urlopen(req, timeout=60).read()
        try:
            data = gzip.decompress(raw)
        except OSError:
            data = raw
        txt = data.decode('utf-8', 'replace')
        mails = sorted(set(MAIL.findall(txt)))
        authors = []
        for m in re.finditer(r'\\author\{', txt):
            i = m.end()
            depth = 1
            while i < len(txt) and depth:
                if txt[i] == '{':
                    depth += 1
                elif txt[i] == '}':
                    depth -= 1
                i += 1
            authors.append(' '.join(txt[m.end():i - 1].split())[:70])
        print('=== %s  (%s)' % (aid, who))
        print('   yazarlar : %s' % ' | '.join(authors[:4]))
        print('   e-postalar: %s' % (', '.join(mails) if mails else '(yok)'))
    except Exception as exc:
        print('=== %s  HATA: %s' % (aid, exc))
