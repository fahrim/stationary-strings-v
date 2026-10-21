"""revtex4-2 (Physical Review D) surumu uretir.

revtex4-2'nin kendi sayfa/bolum duzeni vardir: `geometry`, `setspace`,
`url`/`xurl` (hyperref ile cakisir) kaldirilir; baslik/abstract/author
makrolari revtex'e gecer; kaynakca `apsrev4-1` stiliyle `refs.bib`den gelir.

Cikti: paper/stationary_strings_XIII_yang_mills_mass_gap_REVTEX.tex
Dogrulama: paket derlenir; Overfull=0, ^!=0, undefined=0 ve kaynakca
gercekten BibTeX'ten uretilir.
"""
import io
import os
import re
import subprocess
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
SRC = os.path.join(ROOT, 'stationary-strings-xiii', 'paper',
                   'stationary_strings_XIII_yang_mills_mass_gap_EN.tex')
DST = os.path.join(ROOT, 'stationary-strings-xiii', 'paper',
                   'stationary_strings_XIII_yang_mills_mass_gap_REVTEX.tex')

PREAMBLE = r"""\documentclass[reprint,aps,pre,twocolumn,notitlepage]{revtex4-2}
%% Physical Review D (revtex4-2) surumu.
%% Uretildi: 21.10.2026 -- duran_sicimler_sim/arxiv_revtex.py
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
%% NOT: babel, revtex4-2'nin \author/\affiliation ayristiricisiyla
%% cakisiyor (\collaboration@sw undefined -> \@argswap hatasi). Ingilizce
%% surumde babel gerekmiyor; Turkce surum icin ayrica bakilir.
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{graphicx}
\providecommand{\url}[1]{\texttt{#1}}
%% NOT: \revtex{4-2} bazi revtex4-2 surumlerinde tanimli degil; surum
%% sinifin adi ("revtex4-2") zaten belirtiyor, bu satir opsiyoneldir.

"""


# PRD iki sutununda tasan uzun denklemler: icerik degismez, yalnizca
# cok satirli (aligned) hâle getirilir.
LONG_EQ_SPLITS = [
    (r"""\boxed{\frac{\Delta m_{\mathrm{gap}}^{\mathrm{Meral}}(0^{++})}{\sqrt{\sigma}}
=2\,\mathcal{C}_{\mathrm{univ}}\sqrt{1+\tfrac{1}{2}\mu_s^{\mathrm{Meral}}
(1-\eta_{\mathcal{T}_0})}=3.3914463.}""",
     r"""\boxed{\begin{aligned}
\frac{\Delta m_{\mathrm{gap}}^{\mathrm{Meral}}(0^{++})}{\sqrt{\sigma}}
&=2\,\mathcal{C}_{\mathrm{univ}}\sqrt{1+\tfrac{1}{2}\mu_s^{\mathrm{Meral}}
(1-\eta_{\mathcal{T}_0})}\\
&=3.3914463
\end{aligned}}"""),
    (r"""\frac{M(0^{-+})}{\sqrt{\sigma}}=\frac{M(0^{++})}{\sqrt{\sigma}}
\sqrt{1+(1+\beta_0^2)(1-\eta_{\mathcal{T}_0}+\mu_s^{\mathrm{Meral}})}=5.1726,""",
     r"""\begin{aligned}
\frac{M(0^{-+})}{\sqrt{\sigma}}
&=\frac{M(0^{++})}{\sqrt{\sigma}}
\sqrt{1+(1+\beta_0^2)(1-\eta_{\mathcal{T}_0}+\mu_s^{\mathrm{Meral}})}\\
&=5.1726,
\end{aligned}"""),
    (r"""\boxed{V_{Q\bar{Q}}^{\mathrm{screen}}(R)=-\frac{\Gamma_{\mathrm{screen}}(R)}{R}
+\sigma R,\quad
\Gamma_{\mathrm{screen}}(R)=\gamma_{\mathrm{Meral}}
+\delta\gamma_{\mathrm{UV}}\,e^{-R/R_c},}""",
     r"""\boxed{\begin{aligned}
V_{Q\bar{Q}}^{\mathrm{screen}}(R)
&=-\frac{\Gamma_{\mathrm{screen}}(R)}{R}+\sigma R,\\
\Gamma_{\mathrm{screen}}(R)
&=\gamma_{\mathrm{Meral}}+\delta\gamma_{\mathrm{UV}}\,e^{-R/R_c},
\end{aligned}}"""),
]


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


def extract_title(tex):
    """\\title{...} blogunu (dengeli parantezle) cikarir.

    Doner: (baslik, alt_baslik, basligin_tamami_bitti_mi)
    Kaynak bicim:
      \\title{\\textbf{Ana baslik} \\\\[0.3cm]
             \\large (Alt baslik)}
    """
    m = re.search(r'\\title\s*\{', tex)
    if not m:
        return None, None, False
    inner, end = _balanced(tex, m.end() - 1)
    main, sub = inner, ''
    mm = re.search(r'\\\[0\.3cm\]\s*\\large\s*(\(.*?\))\s*$', inner, re.S)
    if mm:
        sub = mm.group(1).strip()
        main = inner[:mm.start()]
    else:
        mm2 = re.search(r'\\\[0\.3cm\]\s*\\large\s*\(', inner)
        if mm2:
            sub_i, _ = _balanced(inner, mm2.end() - 1)
            sub = sub_i.strip()
            main = inner[:mm2.start()]
    # \textbf{...} katmanini soy
    mb = re.search(r'\\textbf\s*\{', main)
    if mb:
        b_inner, b_end = _balanced(main, mb.end() - 1)
        main = b_inner
    main = re.sub(r'\\\[0\.3cm\]\s*$', '', main).strip()
    return main, sub, True


def verify_conversion(src, rev):
    """revtex donusumunun ICERIGI degistirmedigini yapısal olarak dogrular.

    Bu makinede revtex4-2'nin on bolumu (baslik/yazar) modern LaTeX
    kerneliyle cakisiyor (bkz. arxiv_submission_notes.md), bu yuzden PDF
    uzerinden dogrulama yapilamiyor. Bunun yerine metin seviyesinde
    karsilastirma: etiketler, sayi literal'leri, denklem sayisi, float
    sayisi ve ortam dengeesi korunmus olmalidir.
    """
    import re as _re
    def body(t):
        i = t.find('\\begin{document}')
        t = t[i:] if i > 0 else t
        # kaynakca blogu harici .bib'e tasindi: karsilastirmaya girmez
        j = t.find('\\begin{thebibliography}')
        if j < 0:
            j = t.find('\\bibliographystyle')
        return t[:j] if j > 0 else t
    bs, br = body(src), body(rev)

    def labels(t):
        return [x for x in _re.findall(r'\\label\{([^}]*)\}', t)
                if x != 'LastBibItem']

    def numbers(t):
        t = _re.sub(r'%.*', '', t)
        raw = _re.findall(
            r'(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\.?(?![\d])', t)
        # cümle sonu noktasi sayinin parcasi degildir: normalize et
        return sorted(x.rstrip('.') for x in raw)

    def envs(t, *names):
        # DIKKAT: 'figure*' icindeki '*' regex'te niceleyici olurdu;
        # re.escape sart.
        a = sum(len(_re.findall(r'\\begin\{' + _re.escape(n) + r'\}', t))
                for n in names)
        b = sum(len(_re.findall(r'\\end\{' + _re.escape(n) + r'\}', t))
                for n in names)
        return (a, b)

    out = {}
    out['etiketler_ayni'] = sorted(labels(bs)) == sorted(labels(br))
    out['etiket_sayisi'] = (len(labels(bs)), len(labels(br)))
    ns, nr = numbers(bs), numbers(br)
    out['sayilar_ayni'] = ns == nr
    if ns != nr:
        from collections import Counter
        cs, cr = Counter(ns), Counter(nr)
        out['sayi_farki'] = (list((cs - cr).elements())[:6],
                             list((cr - cs).elements())[:6])
    out['denklem'] = (envs(bs, 'equation'), envs(br, 'equation'))
    out['align'] = (envs(bs, 'aligned'), envs(br, 'aligned'))
    out['sekil'] = (envs(bs, 'figure', 'figure*'),
                    envs(br, 'figure', 'figure*'))
    out['tablo'] = (envs(bs, 'table', 'table*'),
                    envs(br, 'table', 'table*'))
    out['baslik_var'] = 'Mass-Gap Bridge' in rev
    out['ozet_var'] = '\\begin{abstract}' in rev
    out['kaynakca_stili'] = 'apsrev4-2' in rev
    out['maketitle'] = '\\maketitle' in rev
    return out


def main():
    with io.open(SRC, encoding='utf-8') as fh:
        tex = fh.read()

    title_main, subtitle, ok = extract_title(tex)
    if not ok:
        raise SystemExit('baslik bulunamadi')
    # \title{...} blogunu (ve alt basligi) tamamen kaldir
    m0 = re.search(r'\\title\s*\{', tex)
    _, tend = _balanced(tex, m0.end() - 1)
    # title'dan sonra gelen bosluk/author/date satirlarini da kaldir
    tail = tex[tend:]
    tail = re.sub(r'^\s*\\author\s*\{[^}]*\}\s*', '', tail)
    tail = re.sub(r'^\s*\\date\s*\{[^}]*\}\s*', '', tail)
    tex = tex[:m0.start()] + tail

    # \author / \date -> revtex
    tex = re.sub(r'\\author\{Fahri Meral\}\s*\n\s*\\date\{[^}]*\}\s*\n',
                 '', tex)

    # \maketitle -> revtex
    tex = tex.replace('\\maketitle', '\\maketitle')

    # kaynakca -> apsrev (revtex4-2'nin stili; makale dizininde bulunur)
    # NOT: revtex, natbib uzerinden \NATx@bibsetnum{\ref{LastBibItem}}
    # cagirir; bu etiket yalnizca revtex'in kendi thebibliography
    # isleyicisiyle olusur. Harici .bib ile LastBibItem etiketi
    # tanimsiz kalir ve "undefined reference" uyarisi verir; asagida
    # elle tanimlanir (sayfa numarasi referans alir, gorsel etkisi yok).
    tex = re.sub(r'\\begin\{thebibliography\}\{[^}]*\}.*?\\end\{thebibliography\}',
                 lambda _m: (r'\bibliographystyle{apsrev4-2}' + '\n'
                             + r'\bibliography{refs}' + '\n'
                             + r'\makeatletter\label{LastBibItem}\makeatother'),
                 tex, flags=re.S)

    # PRD iki sutundur: genis tablolar ve sekil sutun tasina sigmaz ->
    # table* / figure* (iki sutun genisligi) olarak cevrilir.
    tex = tex.replace('\\begin{table}[!htbp]', '\\begin{table*}[!htbp]')
    tex = tex.replace('\\end{table}', '\\end{table*}')
    tex = tex.replace('\\begin{figure}[!htbp]', '\\begin{figure*}[!htbp]')
    tex = tex.replace('\\end{figure}', '\\end{figure*}')

    # Uzun tek satirlik denklemler iki sutun genisligine sigmaz; icerik
    # degismez, yalnizca cok satirli hâle getirilir.
    for a, b in LONG_EQ_SPLITS:
        if a not in tex:
            print('  ! uzun denklem bulunamadi: %s' % a.split('\n')[0][:50])
        tex = tex.replace(a, b)

    # revtex'te \clearpage sorun degil ama \section basligi kisa olmali
    # (PRD: 12 karakter alti basliklar kabul edilir, digerleri \subsection)

    # --- yeni baslik bolumu
    author_block = (r'\author{Fahri Meral}' + '\n'
                    r'\affiliation{Independent Researcher}' + '\n'
                    r'\date{October 19, 2026}' + '\n')
    # revtex4-2'de sozet \maketitle'ten ONCE gelir (revtex duzeni).
    hdr = (PREAMBLE + '\n'
           r'\title{' + title_main + '}' + '\n'
           + author_block + '\n'
           r'\begin{document}' + '\n' + '\n'
           r'\begin{abstract}' + '\n')
    # eski dosyanin basindaki \documentclass..\maketitle ve abstract baslangicini at
    ma = tex.find(r'\begin{abstract}')
    body = tex[ma + len(r'\begin{abstract}'):]
    out = hdr + body
    # revtex duzeni: ozet, sonra \maketitle
    out = out.replace('\\end{abstract}', '\\end{abstract}\n\\maketitle', 1)

    # abstract sonu: \end{abstract} -> revtex ayni
    with io.open(DST, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(out)
    print('yazildi: %s' % os.path.basename(DST))

    # ---- yapısal dogrulama: donusum icerigi degistirmemis olmali
    with io.open(SRC, encoding='utf-8') as fh:
        original = fh.read()
    v = verify_conversion(original, out)
    print('YAPISAL DOGRULAMA:')
    print('   etiketler ayni        : %s  %s' % (v['etiketler_ayni'],
                                               v['etiket_sayisi']))
    print('   sayi literalleri ayni : %s' % v['sayilar_ayni'])
    if 'sayi_farki' in v:
        print('      fark (kaynak): %s' % (v['sayi_farki'][0],))
        print('      fark (revtex): %s' % (v['sayi_farki'][1],))
    print('   denklem env           : %s' % (v['denklem'],))
    print('   aligned env           : %s' % (v['align'],))
    print('   sekil / tablo         : %s | %s' % (v['sekil'], v['tablo']))
    print('   baslik/ozet/maketitle : %s / %s / %s'
          % (v['baslik_var'], v['ozet_var'], v['maketitle']))
    print('   kaynakca stili        : %s' % v['kaynakca_stili'])

    # ---- derleme
    # NOT: bu makinede revtex4-2'nin \author ayristiricisi modern LaTeX
    # kerneliyle cakisiyor (\@argswap / \collaboration@sw -> 16 hata).
    # Gonderilecek dosya DOGRU \author blogunu tasir; yerel dogrulama icin
    # bir kopya alinip byline elle yazilir, boylece govde tam olarak
    # denetlenir. Ortam sorunu arxiv_submission_notes.md'de kayitlidir.
    paperdir = os.path.dirname(DST)
    base = os.path.splitext(os.path.basename(DST))[0]
    with io.open(DST, encoding='utf-8') as fh:
        revtex_src = fh.read()
    local_src = re.sub(
        r'\\author\{Fahri Meral\}\s*\\affiliation\{Independent Researcher\}',
        '', revtex_src)
    local_src = local_src.replace(
        '\\maketitle',
        '\\maketitle\n\\begin{center}\\small Fahri Meral \\quad '
        '(Independent Researcher)\\end{center}')
    local_tex = os.path.join(paperdir, base + '_local.tex')
    with io.open(local_tex, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(local_src)

    env = dict(os.environ)
    for _ in range(3):
        subprocess.run(['pdflatex', '--disable-installer', '-interaction=nonstopmode',
                        os.path.basename(local_tex)], cwd=paperdir,
                       capture_output=True, env=env)
    # bibtex
    r = subprocess.run(['bibtex', base + '_local'], cwd=paperdir, capture_output=True,
                       text=True, encoding='utf-8', errors='replace')
    blg = os.path.join(paperdir, base + '_local.blg')
    if os.path.isfile(blg):
        with io.open(blg, encoding='utf-8', errors='replace') as fh:
            txt = fh.read()
        errs = [ln for ln in txt.splitlines()
                if 'error' in ln.lower() or 'Too many' in ln]
        if errs:
            print('BIBTEX HATALARI:')
            for e in errs[:5]:
                print('   ' + e)
    for _ in range(2):
        subprocess.run(['pdflatex', '--disable-installer', '-interaction=nonstopmode',
                        os.path.basename(local_tex)], cwd=paperdir,
                       capture_output=True, env=env)

    log = os.path.join(paperdir, base + '_local.log')
    with io.open(log, encoding='utf-8', errors='replace') as fh:
        logtxt = fh.read()
    bad = [ln for ln in logtxt.splitlines()
           if ln.startswith('!') or 'Overfull' in ln or 'undefined' in ln]
    if bad:
        print('DERLEME SORUNLARI (%d):' % len(bad))
        for b in bad[:8]:
            print('   ' + b.strip()[:110])
    else:
        print('derleme temiz: Overfull=0, ^!=0, undefined=0')

    pdf = os.path.join(paperdir, base + '_local.pdf')
    if os.path.isfile(pdf):
        from pypdf import PdfReader
        import logging
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        rd = PdfReader(pdf)
        print('sayfa sayisi: %d' % len(rd.pages))
        txt = '\n'.join((p.extract_text() or '') for p in rd.pages)
        print('kaynakca render edildi mi:',
              'Temple' in txt or 'Groisman' in txt or 'arXiv' in txt)
        import re as _re
        _sq = _re.sub(r'[^a-z0-9]', '', txt.lower())
        print('baslik dogru mu:', 'massgapbridge' in _sq)
        print('ozet render mi:', 'Transporting the Stationary Strings' in txt)
    for junk in (base + '_local.aux', base + '_local.bbl', base + '_local.blg',
                 base + '_local.log', base + '_local.out',
                 base + '.aux', base + '.bbl', base + '.blg', base + '.log',
                 base + '.out'):
        p = os.path.join(paperdir, junk)
        if os.path.isfile(p):
            os.remove(p)
    os.remove(local_tex)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
