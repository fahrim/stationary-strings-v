"""TR cevirilerinin EN kaynaklariyla sayi/identifikator parity'sini dogrular.

Kural: bir TR dosyasinda, EN kaynagindaki HER sayi ve HER kimlik
(DOI, dosya adi, URL, depo adi) bulunmalidir. Ceviride sayi degistirmek
veya bir sayiyi dusmek hatadir.

Kullanim:
  python -W ignore verify_tr_parity.py
"""
import io
import os
import re
import sys

ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
PREV = os.path.join(ROOT, 'stationary-strings-previews')

# (EN dosya, TR dosya)
PAIRS = [
    ('arXiv_endorsement_step_by_step.md', 'arXiv_endorsement_step_by_step_TR.md'),
    ('arXiv_submission_notes.md', 'arXiv_submission_notes_TR.md'),
    ('letter_to_siegert_yoneda_EN.md', 'letter_to_siegert_yoneda_TR.md'),
    ('cover_note_EN.md', 'cover_note_TR.md'),
    ('journal_kit/README.md', 'journal_kit/README_TR.md'),
    ('journal_kit/highlights.md', 'journal_kit/highlights_TR.md'),
    ('journal_kit/declarations.md', 'journal_kit/declarations_TR.md'),
    ('journal_kit/porting_notes.md', 'journal_kit/porting_notes_TR.md'),
    ('journal_kit/submission_aa_L1.md', 'journal_kit/submission_aa_L1_TR.md'),
    ('arxiv_metadata/README.md', 'arxiv_metadata/README_TR.md'),
]

NUM = re.compile(r'(?<![\w.,])\d+(?:[.,]\d+)?(?![\w.])')
IDENT = re.compile(
    r'10\.\d{4,9}/[^\s,)\]}>"]+'
    r'|[A-Za-z0-9_.-]+\.(?:md|py|tex|bib|pdf|png|json|cls|sty|bst|rtx|txt)'
    r'|https?://[^\s,)\]}>"]+'
    r'|stationary-strings-[a-z0-9-]+'
    r'|\b(?:GHOST_CANDIDATE|HAS_PROVENANCE|NOT_A_LIT_VALUE|ABSENT)\b'
    r'|\b(?:astro-ph|hep-th|hep-ex|hep-ph|hep-lat|physics\.soc-ph|q-bio|quant-ph|cs\.DL)\b')


def _norm_num(s):
    """Turkce ondalik ayracini (virgul) noktaya cevirir.

    TR metinlerinde 0,08414 dogru yazimdir; arXiv/journal metadatasinda
    ise 0.08414 olmalidir. Parity kontrolu metin oldugu icin virgulu
    noktaya normalize ediyoruz -- sonra dosya turune gore ayrica bakilir.
    """
    return s.replace(',', '.')


def read(p):
    with io.open(p, encoding='utf-8') as fh:
        return fh.read()


def ctrl(t):
    return [c for c in t if ord(c) < 32 and c not in '\n\t']


def main(argv=None):
    ok_all = True
    for en_rel, tr_rel in PAIRS:
        en_p = os.path.join(PREV, en_rel.replace('/', os.sep))
        tr_p = os.path.join(PREV, tr_rel.replace('/', os.sep))
        if not os.path.isfile(tr_p):
            print('%-42s TR YOK' % tr_rel)
            ok_all = False
            continue
        en, tr = read(en_p), read(tr_p)
        en_nums = sorted(_norm_num(x) for x in NUM.findall(en))
        tr_nums = sorted(_norm_num(x) for x in NUM.findall(tr))
        miss_n = [x for x in set(en_nums) if x not in tr_nums]
        en_id = set(IDENT.findall(en))
        tr_id = set(IDENT.findall(tr))
        miss_i = sorted(x for x in en_id if x not in tr_id)
        c = ctrl(tr)
        bom = open(tr_p, 'rb').read(3) == b'\xef\xbb\xbf'
        status = 'TAMAM' if (not miss_n and not miss_i and not c and not bom) else 'SORUN'
        if status != 'TAMAM':
            ok_all = False
        print('%-42s %-6s sayi=%d ident=%d  eksik_sayi=%s  eksik_ident=%s%s%s'
              % (tr_rel, status, len(en_nums), len(en_id),
                 miss_n[:4] or '-', miss_i[:3] or '-',
                 '  CTRL:%d' % len(c) if c else '',
                 '  BOM' if bom else ''))
    print('\nSONUC: %s' % ('hepsi tamam' if ok_all else 'eksikler var'))
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
