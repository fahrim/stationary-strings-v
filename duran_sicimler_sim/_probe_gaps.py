import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'G:\my_scientific_work\stationary-strings-theorem'
PREV = os.path.join(ROOT, 'stationary-strings-previews')

CASES = [
    ('arXiv_endorsement_step_by_step.md', 'arXiv_endorsement_step_by_step_TR.md',
     ['6.3', '0000', '07', '9.1']),
    ('arXiv_submission_notes.md', 'arXiv_submission_notes_TR.md',
     ['22', '40', '2016', '12']),
    ('cover_note_EN.md', 'cover_note_TR.md', ['7']),
    ('journal_kit/submission_aa_L1.md', 'journal_kit/submission_aa_L1_TR.md',
     ['0', '13']),
]

for en_rel, tr_rel, probes in CASES:
    en = io.open(os.path.join(PREV, en_rel), encoding='utf-8').read()
    tr = io.open(os.path.join(PREV, tr_rel), encoding='utf-8').read()
    print('=== %s' % en_rel)
    for p in probes:
        in_tr = p in tr
        print('  [%s] TR var mi: %s' % (p, in_tr))
        for ln in [x.strip() for x in en.split('\n') if p in x][:2]:
            print('      EN: %s' % ln[:130])
    for m in re.finditer(r'https?://[^\s,)\]}>"]+', en):
        u = m.group(0)
        if u not in tr:
            ctx = en[max(0, m.start() - 95):m.start()].replace('\n', ' ')
            print('  EKSIK URL: %s' % u[:95])
            print('      baglam: ...%s' % ctx[-85:])
    print()
