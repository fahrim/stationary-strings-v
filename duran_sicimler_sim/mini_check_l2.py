"""L2: ghost_audit aracinin bagimsiz kendi denetimi.

Kural: bu betik COZUCUYU IMPORT ETMEZ. Yalnizca denetim aracini (ghost_audit)
test eder ve sentetik depo/korpus kurar.

Kontrol listesi:
  L1 token siniri      : '1.55' -> '1.556152e+05' ile ESLESMEMELI
  L2 hayalet adayi     : deger, iddiayla ayni commit'te referans baglaminda
  L3 provenance        : deger, iddiadan onceki commit'te referans baglaminda
  L4 referans disi     : hicbir yerde isaret yok -> NOT_A_LIT_VALUE
  L5 yokluk            : depoda hic yok -> ABSENT
  L6 kaynak taramasi   : gecis sayimi + es-sidd (co-occurrence)
  L7 gercek vaka       : PEO bandi -> GHOST_CANDIDATE ve korpusta 0 HIT
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import ghost_audit as GA

PASS = [0, 0]


def check(name, cond, detail=''):
    PASS[1] += 1
    if cond:
        PASS[0] += 1
        print('PASS %-26s %s' % (name, detail))
    else:
        print('FAIL %-26s %s' % (name, detail))


def _git(repo, *args):
    subprocess.run(['git', '-C', repo] + list(args), check=True,
                   capture_output=True, text=True)


def make_repo(files):
    """files: [(relative_path, content), ...] -> (repo_path, commit_hash list)"""
    tmp = tempfile.mkdtemp(prefix='ghost_l2_')
    _git(tmp, 'init', '-q')
    _git(tmp, 'config', 'user.email', 'l2@test')
    _git(tmp, 'config', 'user.name', 'L2 Test')
    hashes = []
    for rel, content in files:
        path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        _git(tmp, 'add', '-A')
        _git(tmp, 'commit', '-q', '-m', 'commit for %s' % rel)
        h = subprocess.run(['git', '-C', tmp, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True).stdout.strip()
        hashes.append(h)
    return tmp, hashes


# ------------------------------------------------------------------ L1
def L1_token_boundary():
    pat = GA.standalone_pattern('1.55')
    rx = __import__('re').compile(pat)
    check('L1_token_boundary',
          rx.search('1.556152e+05') is None and
          rx.search('x = 1.55,') is not None and
          rx.search('[1.55, 1.70]') is not None,
          'alt dize eslesmesi engellendi')


# ------------------------------------------------------------------ L2
def L2_ghost_candidate():
    tmp, _ = make_repo([
        ('solver.py', '# baseline\nx = 1\n'),
        ('report.md', '2. **Bagimsiz literatur (model disi):** PEO girdap bantlari '
                      '(r: [7.77, 9.99])\n'
                      '   model sonucu 0.83 bant [7.77, 9.99] icinde\n'),
    ])
    first, lf, cf, la = GA.archaeology(tmp, ['7.77'])
    v = GA.provenance_verdict('7.77', first, lf, cf, la)
    check('L2_ghost_candidate', v['verdict'] == 'GHOST_CANDIDATE',
          'karar=%s' % v['verdict'])


# ------------------------------------------------------------------ L3
def L3_has_provenance():
    tmp, _ = make_repo([
        ('src.txt', 'Kaynak: Smith (2020), deney banti 7.77\n'),
        ('report.md', 'model sonucu 0.83 bant [7.77, 9.99] icinde\n'),
    ])
    first, lf, cf, la = GA.archaeology(tmp, ['7.77'])
    v = GA.provenance_verdict('7.77', first, lf, cf, la)
    check('L3_has_provenance', v['verdict'] == 'HAS_PROVENANCE',
          'karar=%s' % v['verdict'])


# ------------------------------------------------------------------ L4
def L4_not_a_lit_value():
    tmp, _ = make_repo([('solver.py', 'tol = 7.77\n')])
    first, lf, cf, la = GA.archaeology(tmp, ['7.77'])
    v = GA.provenance_verdict('7.77', first, lf, cf, la)
    check('L4_not_a_lit_value', v['verdict'] == 'NOT_A_LIT_VALUE',
          'karar=%s' % v['verdict'])


# ------------------------------------------------------------------ L5
def L5_absent():
    tmp, _ = make_repo([('solver.py', 'tol = 1\n')])
    first, lf, cf, la = GA.archaeology(tmp, ['7.77'])
    v = GA.provenance_verdict('7.77', first, lf, cf, la)
    check('L5_absent', v['verdict'] == 'ABSENT', 'karar=%s' % v['verdict'])


# ------------------------------------------------------------------ L6
def L6_source_scan():
    tmp = tempfile.mkdtemp(prefix='ghost_src_')
    os.makedirs(os.path.join(tmp, 'sub'), exist_ok=True)
    with open(os.path.join(tmp, 'a.tex'), 'w', encoding='utf-8') as fh:
        fh.write('band 1.55 ve 0.35\n')
    with open(os.path.join(tmp, 'sub', 'b.tex'), 'w', encoding='utf-8') as fh:
        fh.write('1.55 only\n')
    with open(os.path.join(tmp, 'c.txt'), 'w', encoding='utf-8') as fh:
        fh.write('1.55 (ignored: .txt)\n')
    sc = GA.scan_sources(tmp, '1.55')
    grp = GA.scan_sources_group(tmp, ['1.55', '0.35'])
    check('L6_source_scan',
          sc['n_hits'] == 2 and grp['cooccurrence'] == 1,
          'hits=%d (beklenen 2), cooc=%d (beklenen 1)'
          % (sc['n_hits'], grp['cooccurrence']))


# ------------------------------------------------------------------ L7
def L7_real_case():
    repo = os.path.abspath(os.path.join(HERE, '..'))
    sources = r'G:\my_scientific_work\sources\antimatter'
    if not os.path.isdir(sources):
        print('SKIP %-26s kaynak korpusu yok' % 'L7_real_case')
        return
    first, lf, cf, la = GA.archaeology(repo, ['1.55'], ['duran_sicimler_sim'])
    v = GA.provenance_verdict('1.55', first, lf, cf, la)
    grp = GA.scan_sources_group(sources, ['1.55', '1.70'])
    ok = (v['verdict'] == 'GHOST_CANDIDATE' and grp['n_hits'] == 0
          and grp['cooccurrence'] == 0)
    check('L7_real_case', ok,
          'karar=%s, korpus HIT=%d, es-sidd=%d'
          % (v['verdict'], grp['n_hits'], grp['cooccurrence']))


if __name__ == '__main__':
    print('=' * 68)
    print('L2 MINI DENETIMI (ghost_audit araci)')
    print('=' * 68)
    L1_token_boundary()
    L2_ghost_candidate()
    L3_has_provenance()
    L4_not_a_lit_value()
    L5_absent()
    L6_source_scan()
    L7_real_case()
    print('-' * 68)
    print('MINI %d/%d PASS' % (PASS[0], PASS[1]))
    sys.exit(0 if PASS[0] == PASS[1] else 1)
