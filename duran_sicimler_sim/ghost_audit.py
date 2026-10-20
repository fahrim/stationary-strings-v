"""L2: Hayalet parametre adli analizi (ghost-parameter forensics).

Iki bagimsiz denetim:

  1) GIT ARKEOLOJISI  -- bir sayinin "ilk gercek gorunusu" bulunur ve
     "referans kisiminda ilk ortaya cikisi" ile karsilastirilir. Eger sayi
     depoya girdigi andan itibaren literatur/referans kisiminda ise
     (onceki hicbir commit'te kaynakli bir kayit yoksa) HAYALET ADAYIDIR.
     Cunku: literatur degeri icin "dogrulama" sorusu "sayi dogru mu?"
     degil "kaynakta VAR MI?" sorusudur.

  2) KAYNAK VARLIK TARAMASI -- bir sayinin kaynak korpusunda (arXiv LaTeX
     kaynaklari + PDF metni) kac kez gectigi sayilir. 0 -> kayit yok.

Tasarim ilkesi: hicbir sey sessizce "0" sayilmaz. PDF metin cikarimi
basarisiz olursa bu acikca raporlanir.

Kullanim:
  python -W ignore ghost_audit.py --repo <git repo> \
      --values 1.55 1.70 0.30 0.35 --sources <kaynak kok> --out rapor.json
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

# Bir satirin "literatur/referans" baglami sayilmasini saglayan isaretler.
LIT_MARKERS = (
    'lit_', 'literat', 'Literature', 'LITERAT', 'band', 'Band', 'BAND',
    'obs_', 'observ', 'measured', 'olcul', 'ref_', 'source_', 'kaynak',
    'published', 'band_', 'range_', 'exp_',
)

# Bir satirin "IDDIA" baglami sayilmasini saglayan isaretler: degerin
# literaturle UYUSTUGU / modelin onu onayladigi yazilmaktadir.
# (Referans kaydi ile iddia ayrimi: degerin kaynakli bir kaydi varsa
#  git arkeolojisi "hayalet" kararini vermemelidir.)
CLAIM_MARKERS = (
    'uyum', 'uyumu', 'agrees', 'agree', 'mukabele', 'coincid', 'tutar',
    'match', 'inside', 'icinde', 'içinde', 'predicted', 'prediction',
    'onay', 'confirm', 'dogrular', 'doğrular', 'lands in', 'falls in',
    'within the band', 'bandin icinde', 'bandın içinde', 'hit the band',
)

# Isaretler kucuk harfe indirgenip karsilastirilir (buyuk/kucuk duyarsiz).
_LIT_LOW = tuple(m.lower() for m in LIT_MARKERS)
_CLAIM_LOW = tuple(m.lower() for m in CLAIM_MARKERS)


def _has_any(text, markers_low):
    low = text.lower()
    return any(m in low for m in markers_low)

# ERE (git grep -E) lookaround desteklemez; bu yuzden token siniri
# karakter sinifi ile kurulur: rakam/nokta/e/E ile komsulanmamali.
# (Alt dize eslesmesi '1.55' -> '1.556152e+05' ile yanlis provenance
#  uretecegi icin bu zorunludur.)
_LEFT = r'(^|[^0-9.eE])'
_RIGHT = r'([^0-9.eE]|$)'


def _boundary_pattern(literal):
    return re.escape(literal)


def standalone_pattern(literal):
    """Token olarak tek basina eslesen regex (alt dize degil).

    '1.55' -> '1.556152e+05' ile ESLESMEMELI (sahte provenance onlemi).
    """
    return _LEFT + _boundary_pattern(literal) + _RIGHT


def _git(repo, args):
    out = subprocess.run(['git', '-C', repo] + args,
                         capture_output=True, text=True,
                         encoding='utf-8', errors='replace')
    return out.stdout


def _commits_chrono(repo, paths=None):
    """Kronolojik sirali commit listesi (hash, tarih, konu)."""
    args = ['log', '--all', '--reverse', '--format=%H|%ad|%s', '--date=short']
    if paths:
        args += ['--'] + list(paths)
    rows = []
    for line in _git(repo, args).splitlines():
        parts = line.split('|', 2)
        if len(parts) == 3:
            rows.append(tuple(parts))
    return rows


def _matched_literals(line, literals):
    """Satirdaki hangi literaller geciyor? (token siniri kurallarina gore)"""
    found = []
    for lit in literals:
        pat = re.compile(_LEFT + _boundary_pattern(lit) + _RIGHT)
        if pat.search(line):
            found.append(lit)
    return found


def _file_lines(repo, commit, fname):
    """Bir commit'teki dosyanin satirlari (cache'li)."""
    key = (commit, fname)
    if key in _FILE_CACHE:
        return _FILE_CACHE[key]
    txt = _git(repo, ['show', '%s:%s' % (commit, fname)])
    lines = txt.splitlines()
    _FILE_CACHE[key] = lines
    return lines


_FILE_CACHE = {}


def _has_marker(repo, commit, fname, lno, window=3, markers=_LIT_LOW):
    """Satirin etrafinda (+/- window satir) verilen isaret var mi?

    Isaret ayni satirda olmak zorunda degildir: "PEO girdap bantlari
    (r: [1.55, 1.70])" ifadesi iki satir arasinda bolunebilir.
    """
    lines = _file_lines(repo, commit, fname)
    if not lines:
        return False
    lo = max(0, lno - 1 - window)
    hi = min(len(lines), lno + window)
    for ln in lines[lo:hi]:
        if _has_any(ln, markers):
            return True
    return False


def archaeology(repo, literals, paths=None, groups=None):
    """Tek gecis: her commit'te tum literalleri ayni anda tara.

    args:
      literals : tekil degerler (orn. '1.55')
      groups   : gruplar; her grup '+' ile ayrilmis uyelerden olusur ve
                 bir "gorunus" = tum uyelerin AYNI DOSYADA bulunmasi
                 (orn. '1.55+1.70' -> bant cifti)

    Doner (sozluk, anahtar = deger veya grup adi):
      first[lit]      = ilk token gorunusu
      lit_first[lit]  = ilk REFERANS baglami gorunusu (degerin kaynakli kaydi)
      claim_first[lit]= ilk IDDIA baglami gorunusu (degerin uyum iddiasi)
      lit_all[lit]    = tum referans baglami gorunusleri
    """
    groups = groups or {}
    lits = list(literals) + [m for g in groups.values() for m in g]
    lits = list(dict.fromkeys(lits))
    if not lits:
        return {}, {}, {}, {}
    alt = '|'.join(_boundary_pattern(x) for x in lits)
    pat = re.compile(_LEFT + '(' + alt + ')' + _RIGHT)
    first, lit_first, claim_first = {}, {}, {}
    lit_all = {x: [] for x in lits}
    grp_first, grp_lit_first, grp_claim_first = {}, {}, {}
    grp_all = {g: [] for g in groups}

    for chash, date, subject in _commits_chrono(repo, paths):
        args = ['grep', '-n', '-I', '-E', pat.pattern, chash]
        if paths:
            args += ['--'] + list(paths)
        out = _git(repo, args)
        per_file = {}
        for raw in out.splitlines():
            if not raw:
                continue
            rest = raw.split(':', 1)[1]
            fname, lno, text = rest.split(':', 2)
            text = text.rstrip('\n')
            hit = _matched_literals(text, lits)
            is_lit = _has_any(text, _LIT_LOW) or any(
                _has_marker(repo, chash, fname, int(lno), markers=_LIT_LOW)
                for _ in hit)
            is_claim = _has_any(text, _CLAIM_LOW) or any(
                _has_marker(repo, chash, fname, int(lno), markers=_CLAIM_LOW)
                for _ in hit)
            for lit in hit:
                if lit not in first:
                    first[lit] = dict(commit=chash, date=date, subject=subject,
                                      file=fname, line=int(lno),
                                      text=text.strip())
                if is_lit:
                    rec = dict(commit=chash, date=date, subject=subject,
                               file=fname, line=int(lno), text=text.strip())
                    lit_all[lit].append(rec)
                    if lit not in lit_first:
                        lit_first[lit] = rec
                if is_claim:
                    rec = dict(commit=chash, date=date, subject=subject,
                               file=fname, line=int(lno), text=text.strip())
                    if lit not in claim_first:
                        claim_first[lit] = rec
            for lit in hit:
                per_file.setdefault(fname, set()).add(lit)
        # gruplar: ayni dosyada tum uyeler
        for gname, members in groups.items():
            for fname, found in per_file.items():
                if all(m in found for m in members):
                    rec = dict(commit=chash, date=date, subject=subject,
                               file=fname, line=0, text=' + '.join(members))
                    if gname not in grp_first:
                        grp_first[gname] = rec
                    grp_all[gname].append(rec)
                    break

    # gruplar icin referans/iddia baglami: dosyanin ilk satirlari
    for gname, members in groups.items():
        if gname not in grp_first:
            continue
        rec = grp_first[gname]
        lines = _git(repo, ['grep', '-n', '-I', '-E', pat.pattern,
                            rec['commit'], '--', rec['file']]).splitlines()
        lit_ln = [ln for ln in lines if _has_any(ln, _LIT_LOW)]
        claim_ln = [ln for ln in lines if _has_any(ln, _CLAIM_LOW)]
        if lit_ln:
            grp_lit_first[gname] = dict(
                rec, text=lit_ln[0].split(':', 2)[-1].strip())
        if claim_ln:
            grp_claim_first[gname] = dict(
                rec, text=claim_ln[0].split(':', 2)[-1].strip())

    for gname, members in groups.items():
        if gname not in first and gname in grp_first:
            first[gname] = grp_first[gname]
        if gname not in lit_first and gname in grp_lit_first:
            lit_first[gname] = grp_lit_first[gname]
        if gname not in claim_first and gname in grp_claim_first:
            claim_first[gname] = grp_claim_first[gname]
        lit_all[gname] = grp_all.get(gname, [])
    return first, lit_first, claim_first, lit_all


def provenance_verdict(lit, first, lit_first, claim_first, lit_all):
    """Karar kurallari:
      ABSENT          : deger depoda hic gorunmedi.
      NOT_A_LIT_VALUE : hicbir yerde referans baglami yok.
      NO_CLAIM        : referans baglami var ama uyum iddiasi yok (zararsiz
                        girdi parametresi gibi).
      GHOST_CANDIDATE : deger, uyum iddiasinin OLDUGU commit'te ilk kez
                        referans baglaminda gorunuyor -> oncesinde kaynakli
                        bir kaydi YOK. "Bagimsiz literatur" denilen deger
                        bastan iddiaya dogmus.
      HAS_PROVENANCE : degerin referans baglaminda ilk kaydi, iddiadan
                        ONCEKI bir commit'te (yani kaynakli bir kayit var).
    """
    if lit not in first:
        return dict(literal=lit, verdict='ABSENT', first=None,
                    first_lit=None, first_claim=None, n_lit_appearances=0)
    if lit not in lit_first:
        verdict = 'NOT_A_LIT_VALUE'
    elif lit not in claim_first:
        verdict = 'NO_CLAIM'
    elif lit_first[lit]['commit'] == claim_first[lit]['commit']:
        verdict = 'GHOST_CANDIDATE'
    else:
        verdict = 'HAS_PROVENANCE'
    return dict(literal=lit, verdict=verdict, first=first.get(lit),
                first_lit=lit_first.get(lit), first_claim=claim_first.get(lit),
                n_lit_appearances=len(lit_all.get(lit, [])))


# ---------------------------------------------------------------- kaynak taramasi

def _pdf_text(path):
    """PDF metnini cikar. Basarisizsa (None, hata mesaji) dondur."""
    try:
        import logging
        logging.getLogger('pypdf').setLevel(logging.CRITICAL)
        from pypdf import PdfReader
    except Exception as exc:
        return None, 'pypdf yok: %s' % exc
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or '')
            except Exception:
                chunks.append('')
        return '\n'.join(chunks), None
    except Exception as exc:
        return None, 'okunamadi: %s' % exc


def scan_sources(root, literal):
    """Kaynak korpusunda tokenin gectigi dosyalar + toplam hit sayisi."""
    pat = re.compile(_LEFT + _boundary_pattern(literal) + _RIGHT)
    hits, errors, n_files = [], [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            low = fn.lower()
            if low.endswith('.tex'):
                try:
                    with io.open(path, encoding='utf-8', errors='replace') as fh:
                        text = fh.read()
                except Exception as exc:
                    errors.append((path, 'okunamadi: %s' % exc))
                    continue
                n_files += 1
                cnt = len(pat.findall(text))
                if cnt:
                    hits.append((path, cnt))
            elif low.endswith('.pdf'):
                text, err = _pdf_text(path)
                n_files += 1
                if err:
                    errors.append((path, err))
                    continue
                cnt = len(pat.findall(text))
                if cnt:
                    hits.append((path, cnt))
    return dict(literal=literal, n_files=n_files, n_hits=sum(c for _, c in hits),
                hits=hits, errors=errors)


def scan_sources_group(root, members):
    """Bir bandin (veya herhangi bir sayi grubunun) kaynak korpusundaki
    YERLESIMI.

    Onemli ayrim:
      n_hits        : her uyenin toplam kac kez gectigi (ayri ayri)
      cooccurrence  : TUM uyelerin AYNI DOSYADA bulundugu dosya sayisi

    '0.30' ve '0.35' tek tek bir 511 keV makalesinde gecabilir; bu, bandin
    o kaynakta OLDUGU anlamina gelmez. Co-occurrence olcumu bunu ayirt
    eder, ancak baglam (semantik) yine de insanca dogrulanmalidir.
    """
    per_file = {}
    errors = []
    n_files = 0
    pats = {m: re.compile(_LEFT + _boundary_pattern(m) + _RIGHT)
            for m in members}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            low = fn.lower()
            if low.endswith('.tex'):
                try:
                    with io.open(path, encoding='utf-8', errors='replace') as fh:
                        text = fh.read()
                except Exception as exc:
                    errors.append((path, 'okunamadi: %s' % exc))
                    continue
                n_files += 1
                counts = {m: len(p.findall(text)) for m, p in pats.items()}
                if any(counts.values()):
                    per_file[path] = counts
            elif low.endswith('.pdf'):
                text, err = _pdf_text(path)
                n_files += 1
                if err:
                    errors.append((path, err))
                    continue
                counts = {m: len(p.findall(text)) for m, p in pats.items()}
                if any(counts.values()):
                    per_file[path] = counts
    cooc = [p for p, c in per_file.items()
            if all(c.get(m, 0) > 0 for m in members)]
    return dict(members=members, n_files=n_files, per_file=per_file,
                cooccurrence=len(cooc), cooccurrence_files=cooc,
                n_hits=sum(sum(c.values()) for c in per_file.values()),
                errors=errors)


def inventory_hits(inventory_path, literal):
    """Kaynak envanter dosyasinda (kaynaklarim.md) tokenin gecisi."""
    if not os.path.isfile(inventory_path):
        return None
    with io.open(inventory_path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    pat = re.compile(_LEFT + _boundary_pattern(literal) + _RIGHT)
    return len(pat.findall(text))


# ---------------------------------------------------------------- ana akis

def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Hayalet parametre adli analizi (L2)')
    ap.add_argument('--repo', required=True,
                    help='denetlenecek git deposu (arkeoloji icin)')
    ap.add_argument('--values', nargs='+', required=True,
                    help='denetlenecek sayi literalleri')
    ap.add_argument('--groups', nargs='*', default=[],
                    help='bant gruplari: "1.55+1.70" gibi, ayni dosyada '
                         'birlikte bulunmalari gereken uyeler')
    ap.add_argument('--paths', nargs='*', default=None,
                    help='arama ile sikistirilacak yollar (varsayilan: tum)')
    ap.add_argument('--sources', default=None,
                    help='kaynak korpusu kok dizini (PDF + .tex)')
    ap.add_argument('--inventory', default=None,
                    help='kaynak envanteri (orn. kaynaklarim.md)')
    ap.add_argument('--out', default=None, help='JSON rapor dosyasi')
    ap.add_argument('--quiet', action='store_true',
                    help='pypdf uyarilarini sustur')
    args = ap.parse_args(argv)

    if args.quiet:
        import logging
        logging.disable(logging.CRITICAL)

    # Konsol kodlamasi (Windows cp1254) Unicode hatalarini yuzeye cikarmasin
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    groups = {}
    for g in args.groups:
        members = [m.strip() for m in g.split('+') if m.strip()]
        if len(members) > 1:
            groups['+'.join(members)] = members
    singles = [v for v in args.values if '+' not in v]

    report = dict(repo=os.path.abspath(args.repo), sources=args.sources,
                  groups=list(groups.keys()), values=[])
    print('=' * 78)
    print('HAYALET PARAMETRE ADLI ANALIZI')
    print('depo: %s' % report['repo'])
    if groups:
        print('gruplar: %s' % ', '.join(groups.keys()))
    print('=' * 78)

    first, lit_first, claim_first, lit_all = archaeology(
        args.repo, singles, args.paths, groups)

    for lit in singles + list(groups.keys()):
        row = provenance_verdict(lit, first, lit_first, claim_first, lit_all)
        row['source_scan'] = None
        row['inventory_hits'] = None
        members = groups.get(lit, [lit])
        if args.sources:
            if lit in groups:
                row['source_scan'] = scan_sources_group(args.sources, members)
            else:
                sc = scan_sources(args.sources, lit)
                row['source_scan'] = dict(sc, members=members, cooccurrence=None)
        if args.inventory:
            row['inventory_hits'] = sum(
                inventory_hits(args.inventory, m) or 0 for m in members)
        report['values'].append(row)

        print('')
        print('--- %s ---' % lit)
        print('  karar            : %s' % row['verdict'])
        if row['first']:
            print('  ilk gorunus      : %s (%s) %s' %
                  (row['first']['commit'][:7], row['first']['date'],
                   row['first']['subject'][:58]))
            print('                     %s:%d  %s' %
                  (row['first']['file'], row['first']['line'],
                   row['first']['text'][:68]))
        if row['first_lit']:
            print('  ilk REFERANS kaydi: %s (%s)' %
                  (row['first_lit']['commit'][:7], row['first_lit']['date']))
            print('                     %s:%d  %s' %
                  (row['first_lit']['file'], row['first_lit']['line'],
                   row['first_lit']['text'][:68]))
        if row['first_claim']:
            print('  ilk UYUM IDDIASI : %s (%s)' %
                  (row['first_claim']['commit'][:7], row['first_claim']['date']))
            print('                     %s:%d  %s' %
                  (row['first_claim']['file'], row['first_claim']['line'],
                   row['first_claim']['text'][:68]))
        print('  referans baglami gorunus sayisi: %d' %
              row['n_lit_appearances'])
        if row['source_scan']:
            sc = row['source_scan']
            print('  kaynak korpusu    : %d dosya tarandi, %d HIT' %
                  (sc['n_files'], sc['n_hits']))
            if sc.get('cooccurrence') is not None:
                print('  ES-SIDD (ayni dosyada tum uyeler): %d dosya' %
                      sc['cooccurrence'])
                for path in sc['cooccurrence_files']:
                    print('      COOC %s' % os.path.basename(path))
            for path, cnt in (sc.get('hits') or []):
                print('      HIT %-66s x%d' %
                      (os.path.basename(path), cnt))
            for path, counts in (sc.get('per_file') or {}).items():
                print('      HIT %-52s %s' % (os.path.basename(path),
                                            ' '.join('%s:%d' % (m, c)
                                                     for m, c in counts.items())))
            for path, err in sc['errors']:
                print('      HATA %-66s %s' % (os.path.basename(path), err))
        if row['inventory_hits'] is not None:
            print('  envanter hit     : %d' % row['inventory_hits'])

    n_ghost = sum(1 for r in report['values']
                  if r['verdict'] == 'GHOST_CANDIDATE')
    n_zero = sum(1 for r in report['values']
                 if r['source_scan'] and r['source_scan']['n_hits'] == 0)
    print('')
    print('=' * 78)
    print('HAYALET ADAYI (referans baglaminda, ilk commit ile ayni): %d' % n_ghost)
    print('KAYNAK KORPUSUNDA 0 HIT (kayit yok): %d' % n_zero)
    print('=' * 78)

    if args.out:
        with io.open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print('JSON rapor yazildi: %s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
