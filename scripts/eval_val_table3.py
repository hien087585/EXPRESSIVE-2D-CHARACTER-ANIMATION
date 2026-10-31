"""Table 3 (B0, B1, B2, C1-C5) re-computed on the 4 MEAD VALIDATION speakers.

Runs the evaluation code of notebooks/2b_phaseB_emotion_finetune.ipynb (Cells 1, 2, 3 and 7)
unchanged, except: the clips come from SPLIT['val'] instead of SPLIT['test'], and nothing is
written to Drive (phaseB_evaluation.json is not overwritten). S comes from the true-label lookup
table label_offsets_train.npz (train speakers), exactly as in Table 3.

    python scripts/eval_val_table3.py --drive "<...>/Data_student" [--split val|test] [--out results/val_table3.csv]

--split test reproduces the published Table 3 (sanity check).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / 'notebooks' / '2b_phaseB_emotion_finetune.ipynb'


def patch(src, old, new, count=1):
    assert src.count(old) == count, 'notebook changed, cannot find: %r' % old
    return src.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', required=True, help='Data_student folder (read only)')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--out', default=str(REPO / 'results' / 'val_table3.csv'))
    args = ap.parse_args()

    cells = [''.join(c['source']) for c in json.loads(NB.read_text(encoding='utf-8'))['cells']]
    c1, c2, c3, c7 = cells[4], cells[7], cells[9], cells[19]
    assert c1.startswith('import os, sys, json') and 'EVAL_RUNS' in c7

    c1 = patch(c1, 'EVAL_ONLY = False', 'EVAL_ONLY = True')
    c2 = patch(c2, 'OUT_DIR.mkdir(parents=True, exist_ok=True)', 'pass')
    c7 = patch(c7, "OUT_ROOT.mkdir(parents=True, exist_ok=True)", 'pass')
    c7 = patch(c7, "np.savez(OFF_PATH,", "raise RuntimeError('label_offsets_train.npz missing'); np.savez(OFF_PATH,")
    c7 = patch(c7, "(OUT_ROOT / 'phaseB_evaluation.json').write_text(", '(lambda *a, **k: None)(')
    c7 = patch(c7, "SPLIT['test']", "SPLIT[EVAL_SPLIT]", count=3)

    g = dict(__name__='__nb2b__', DRIVE_FOUND=Path(args.drive), EVAL_SPLIT=args.split)
    exec(c1, g)
    g['REPO_DIR'] = REPO
    g['MEAN_STD_PATH'] = REPO / 'src' / 'dataset' / 'utils' / 'MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt'
    exec(c2, g)
    print('Speaker %s: %s' % (args.split, g['SPLIT'][args.split]))
    exec(c3, g)
    exec(c7, g)

    rep, labels = g['report'], g['LABELS']
    short = {'B0 Phase A (khong fine-tune)': 'B0'}
    short.update({v: k.split('_')[0] for k, v in labels.items()})
    cands = g['CANDS']
    rows = []
    for name in cands[1:]:
        ge, gn = rep['groups']['cam xuc'][name], rep['groups']['neutral'][name]
        mot = rep['motion_only'].get('cam xuc', {}).get(name)
        mo = rep['mouth'].get('cam xuc', {}).get(name)
        model = short.get(name, 'Lookup table only' if name == cands[1] else 'Ceiling (true S)')
        rows.append(dict(model=model, run=name, full_face=round(ge['gain_pct'], 1),
                         mouth_motion=round(mot['mouth'], 1) if mot else '',
                         neutral_clips=round(gn['gain_pct'], 1) if mot else '',
                         mouth_corr=round(mo['corr'], 3) if mo else '',
                         amplitude=round(mo['amp'], 2) if mo else ''))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('\n| Model | Full face | Mouth motion | Neutral clips | Mouth corr. | Amplitude |')
    print('| :--- | ---: | ---: | ---: | ---: | ---: |')
    for r in rows:
        print('| %s | %s | %s | %s | %s | %s |' % (r['model'], r['full_face'], r['mouth_motion'],
                                                 r['neutral_clips'], r['mouth_corr'], r['amplitude']))
    print('\nSaved:', out)


if __name__ == '__main__':
    main()
