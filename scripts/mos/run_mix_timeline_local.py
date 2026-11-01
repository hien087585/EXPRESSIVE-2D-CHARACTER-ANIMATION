"""Fig. 9 data, chay LOCAL (GPU neu co): notebook 3 Cell 0 -> 6 (QUICK_UI = True) + colab_mix_timeline_cell.py
-> M037_MIX1_timeline.npz, roi Acc / UAR cua timeline do, tinh DUNG nhu Cell 12 (luoi 0.1 s, sau Viterbi; recall TB 3 lop).
Doc <DRIVE> va MEAD CHI DOC; moi thu ghi vao --out (tren may). Can funasr (emotion2vec+) trong moi truong Python.
  python run_mix_timeline_local.py --mead <...\\MEAD> --out <LOCAL_OUT>\\timeline
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
REPO = S.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from local_paths import local_path  # noqa: E402

RES = 0.1                                                               # = Cell 12


def cell12_metrics(z, key, cls, shift=0.0):
    """Acc / UAR / Acc tho cua 1 mo hinh tren 1 file, cung cong thuc voi Cell 12 (gt, pred_grid).
    shift (giay): dich cac ranh gioi dap an ben trong (kiem tra do bat dinh khi ranh gioi lay tu khoang lang)."""
    grid = np.arange(0, float(z['dur']), RES)
    b = np.r_[z['gt_start'], z['gt_end'][-1]].astype(float); b[1:-1] += shift
    segs = list(zip(b[:-1], b[1:], z['gt_emo']))
    gt = np.array([cls.index(str(next(s[2] for s in segs if s[0] <= g < s[1] + 1e-9))) for g in grid])
    lab = np.zeros(len(grid), int)
    for t0, t1, e in zip(z[key + '_start'], z[key + '_end'], z[key + '_emo']):
        lab[(grid >= t0) & (grid < t1 + 1e-9)] = cls.index(str(e)) if str(e) in cls else 0
    raw = np.stack([np.interp(grid, z[key + '_centers'], z[key + '_prob'][:, j]) for j in range(len(cls))], 1).argmax(1)
    rec = [float((lab[gt == c] == c).mean()) for c in range(len(cls))]
    return dict(acc=float((lab == gt).mean()), uar=float(np.mean(rec)), acc_raw=float((raw == gt).mean()), recall=rec, n=len(grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mead', required=True, help='thu muc MEAD goc (<speaker>/audio.tar), chi doc')
    ap.add_argument('--out', required=True, help='thu muc ghi tren may (npz, thu vien, file tam)')
    ap.add_argument('--nb', default=str(REPO / 'notebooks' / '3_phaseC_demo_render.ipynb'))
    a = ap.parse_args()
    G, OUT = local_path('DRIVE'), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / 'M037_MIX1_timeline.npz'
    if not npz.exists():
        TMP = OUT / '_tmp'; TMP.mkdir(exist_ok=True)
        os.environ['PHASEC_TEST_ROOT'] = str(OUT / '_troot'); (OUT / '_troot').mkdir(exist_ok=True)   # Cell 1: thu vien ghi o day
        nb = json.loads(Path(a.nb).read_text(encoding='utf-8'))
        C = [''.join(c['source']) for c in nb['cells']]
        g = {'__name__': '__main__'}
        exec(C[2], g)                                                   # Cell 0 (khong phai Colab: chi dinh nghia)
        exec(C[6], g)                                                   # Cell 1 (cau hinh mac dinh, QUICK_UI = True)
        assert g['QUICK_UI'], 'notebook 3 Cell 1 phai co QUICK_UI = True'
        # duong dan: doc tu <DRIVE> / MEAD (chi doc), ghi vao OUT (nhu run_nb3.py)
        g.update(REPO_DIR=REPO, DRIVE=G, MEAD_SRC=Path(a.mead), PHASEB=G / 'student_a2l' / 'phaseB',
                 READY=G / 'Data_emotion_mead_task_b' / 'phaseB_ready', OUT_DIR=OUT / 'demo', REDRAW_DIR=OUT / 'demo_mouth_redraw',
                 CHAR_DIRS=[OUT / 'demo' / 'characters', REPO / 'examples_cartoon'], SER_CKPT=G / 'student_a2l' / 'ser_mead' / 'ser_best.pth',
                 F2_PATHS={'phaseG': G / 'student_a2l' / 'phaseF' / 'phaseG_final.pth',
                           'phaseF2': G / 'student_a2l' / 'phaseF' / 'phaseF2_final.pth'})
        g['F2_CKPT'] = next((p for p in g['F2_PATHS'].values() if p.exists()), g['F2_PATHS']['phaseF2'])
        exec(C[4], g)                                                   # Cell 1b
        exec(C[8], g)                                                   # Cell 2
        A = g['A']
        A.load_wav_16k.__defaults__ = (str(TMP),)
        print('device:', g['DEVICE'], flush=True)
        for i in (10, 12, 14, 16):                                      # Cell 3, 4, 5, 6 (QUICK_UI: chi nap mo hinh + dinh nghia ham)
            t0 = time.time(); exec(C[i], g); print('-- cell %d: %.1f s' % (i, time.time() - t0), flush=True)
        g.update(MIX_OUT_NPZ=npz, MIX_TMP=OUT / 'mixsrc')
        t0 = time.time()
        exec(compile((S / 'colab_mix_timeline_cell.py').read_text(encoding='utf-8'), 'colab_mix_timeline_cell.py', 'exec'), g)
        print('-- mix timeline: %.1f s' % (time.time() - t0))
    else:
        print('Da co %s -> chi tinh lai Acc / UAR' % npz)
    z = np.load(npz)
    cls = [str(c) for c in z['classes']]
    unc = int(z['gt_uncertainty_samples']) / 16000 if 'gt_uncertainty_samples' in z else 0.0
    print('\n%s | %.1f s | %d doan that | audio trung file da luu: %s | dap an: %s (bat dinh %.1f ms)'
          % (z['name'], float(z['dur']), len(z['gt_emo']), bool(z['audio_identical']), z['gt_source'] if 'gt_source' in z else 'rebuilt', 1000 * unc))
    print('%-22s %-7s %-7s %-8s %-18s %s' % ('model', 'Acc', 'UAR', 'Acc raw', 'recall ' + '/'.join(cls), 'Acc/UAR khi dich ranh gioi -/+ bat dinh'))
    res = {}
    for key, lab in (('ens', 'phaseG+e2v (ELSA)'), ('spex', 'phaseG (SpEx-KD)'), ('teacher', 'emotion2vec (teacher)')):
        m = res[key] = cell12_metrics(z, key, cls)
        sh = [cell12_metrics(z, key, cls, s) for s in (-unc, unc)]
        m['shifted'] = [dict(shift_s=s, acc=x['acc'], uar=x['uar']) for s, x in zip((-unc, unc), sh)]
        print('%-22s %-7.3f %-7.3f %-8.3f %-18s %s' % (lab, m['acc'], m['uar'], m['acc_raw'], '/'.join('%.2f' % r for r in m['recall']),
                                                 ' '.join('%.3f/%.3f' % (x['acc'], x['uar']) for x in sh)))
    (OUT / 'M037_MIX1_metrics.json').write_text(json.dumps(dict(file=str(z['name']), res_s=RES, classes=cls, models=res), indent=1))
    ev = G / 'student_a2l' / 'phaseC_demo' / 'test_audio' / 'eval_multi_emotion.json'
    if ev.exists():                                                     # ket qua Cell 12 tren Colab (gop moi file ghep), de doi chieu
        E = json.loads(ev.read_text())
        print('\nCell 12 (Colab, %d file ghep, gop): ' % E['n_files'] +
              ' | '.join('%s Acc %.3f UAR %.3f' % (k, v['acc'], v['uar']) for k, v in E['models'].items()))


if __name__ == '__main__':
    main()
