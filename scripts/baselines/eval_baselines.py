"""SadTalker / Wav2Lip baselines on MEAD, evaluated like MakeItTalk in Table 2 (local part).

Three steps (<DRIVE> is only read; outputs go to --out and results/):

  check    MediaPipe check: re-extract 12 real MEAD front clips (4 test speakers) with the code of notebook 0b and
           compare with phaseB_ready lm_res_raw (normalised face units). Stops if mean >= 0.002 or max >= 0.01.
  extract  mp4 of each baseline -> 68 landmarks with the SAME code as the MEAD data:
             notebook 0b: decode_frames (resize 640x360), FaceLandmarker (same .task model), LMK68 478->68,
                          normalize_landmarks (eye centre / eye distance, no rotation);
             notebook 2a: to62 (linear interpolation to 62.5 fps; here from the fps read from the video);
             notebook 5 : similarity_fit (MakeItTalk's Procrustes), ONE fit per clip from the landmarks of the INPUT
                          image to the speaker's ref, applied to every frame; then minus ref.
           -> <out>/baseline_landmarks.npz, to upload to Drive for colab_eval_baselines_cell.py.
  report   reads the CSV files written by colab_eval_baselines_cell.py + params.json / *_timing.csv of the Colab
           run -> results/baselines_158.csv and a markdown table in the format of Table 2.

    python scripts/baselines/eval_baselines.py check   --drive <Data_student> --mead <MEAD>
    python scripts/baselines/eval_baselines.py extract --drive <Data_student> --inputs <baseline_inputs dir> --runs <out dir>
    python scripts/baselines/eval_baselines.py report  --runs <out dir> --colab <dir with baselines_158_*.csv>
--out / --extract_dir default to <LOCAL_OUT>/baselines (scripts/local_paths.py: env ELSA_LOCAL_OUT or local_paths.json).
"""
import argparse
import ast
import csv
import json
import os
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from local_paths import local_path  # noqa: E402
NB = REPO / 'notebooks'
THRESH_MEAN, THRESH_MAX = 0.002, 0.01
FPS62 = 62.5


# ----------------------------------------------------------------------------- code of the old notebooks
def _cells(nb):
    return [''.join(c['source']) for c in json.loads((NB / nb).read_text(encoding='utf-8'))['cells'] if c['cell_type'] == 'code']


def _defs(nb, names, nested=False):
    """Source of top-level (or, with nested=True, any) functions / assignments `names` of a notebook, unchanged."""
    out = {}
    for src in _cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        nodes = ast.walk(tree) if nested else tree.body
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                key = node.name
            elif isinstance(node, ast.Assign) and not nested:
                tg = [t.id for t in node.targets if isinstance(t, ast.Name)] + \
                     [e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
                key = next((t for t in tg if t in names), None)
            else:
                key = None
            if key and key not in out:
                out[key] = textwrap.dedent(' ' * node.col_offset + ast.get_source_segment(src, node))
    miss = [n for n in names if n not in out]
    assert not miss, '%s: not found %s' % (nb, miss)
    return [out[n] for n in names]


def load_0b(work):
    """Notebook 0b (MEAD landmarks): constants + functions, exactly as written there."""
    g = dict(np=np, os=os, json=json, LOCAL_ROOT=str(work))
    consts = ['FRAME_W', 'MAX_FRAMES', 'SAVE_FULL_478', 'SAVE_FRAME_JPG', 'JPEG_QUALITY', 'LMK68', 'EYE_L', 'MODEL_URL',
              'N_BLEND', 'CLIP_SCHEMA_VERSION']
    funcs = ['get_landmarker', 'run_detector', 'euler_xyz', 'normalize_landmarks', 'decode_frames', 'parse_detection',
             '_nan', 'extract_clip']
    for s in _defs('0b_stageB_mead_landmarks.ipynb', consts):
        exec(s, g)
    g['MODEL_PATH'] = os.path.join(str(work), 'face_landmarker.task'); g['_LANDMARKER'] = None
    (Path(work) / 'tmp').mkdir(parents=True, exist_ok=True)
    for s in _defs('0b_stageB_mead_landmarks.ipynb', funcs):
        exec(s, g)
    assert len(g['LMK68']) == 68 and (g['FRAME_W'], g['FRAME_H']) == (640, 360)
    return g


def load_to62():
    """Notebook 2a to62, with the source frame rate as a parameter (2a hard-codes 30 fps; default unchanged)."""
    src = _defs('2a_phaseB_prepare_data.ipynb', ['to62'], nested=True)[0]
    assert src.count('/ 30.0') == 1 and src.startswith('def to62(x, N):'), src
    src = src.replace('def to62(x, N):', 'def to62(x, N, src_fps=30.0):').replace('/ 30.0', '/ src_fps')
    g = dict(np=np, FPS=FPS62)
    exec(src, g)
    return g['to62']


def load_similarity_fit():
    """similarity_fit of MakeItTalkTeacher (notebook 5), used to map MakeItTalk into MEAD space in Table 2."""
    src = next(s for s in _cells('5_phaseE_benchmark.ipynb') if s.startswith('MIT_TEACHER_SOURCE'))
    mod = ast.literal_eval(src.split('\n', 1)[0][len('MIT_TEACHER_SOURCE = '):])
    tree = ast.parse(mod)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'similarity_fit')
    g = dict(np=np)
    exec(ast.get_source_segment(mod, fn), g)
    return g['similarity_fit']


def mediapipe_version():
    import mediapipe as mp
    return mp.__version__


# ----------------------------------------------------------------------------- check
def tar_members(tar_path, wanted):
    """{name: bytes} for the wanted members of a plain tar (headers are read in order, data skipped)."""
    out = {}
    with tarfile.open(tar_path, 'r:') as tf:
        while len(out) < len(wanted):
            m = tf.next()
            if m is None:
                break
            if m.name in wanted:
                out[m.name] = tf.extractfile(m).read()
    return out


def cmd_check(a):
    from concurrent.futures import ThreadPoolExecutor
    g, to62 = load_0b(Path(a.out) / 'mp_work'), load_to62()
    print('mediapipe %s | model %s' % (mediapipe_version(), g['MODEL_URL']))
    ready = Path(a.drive) / 'Data_emotion_mead_task_b' / 'phaseB_ready'
    split = json.loads((ready / 'split.json').read_text())
    want = {sp: ['video/front/angry/level_1/%03d.mp4' % k for k in (1, 2, 3)] for sp in split['test']}
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        vids = dict(zip(want, ex.map(lambda sp: tar_members(Path(a.mead) / sp / 'video.tar', set(want[sp])), want)))
    print('read %d videos from video.tar (%.0f s)' % (sum(len(v) for v in vids.values()), time.time() - t0))
    rows, all_e = [], []
    for sp, members in want.items():
        z = np.load(ready / (sp + '.npz'))
        names, off, ref = [str(n) for n in z['names']], z['offsets'], z['ref'].astype(np.float32)
        for mname in members:
            clip = '%s__front__angry__L1__%s' % (sp, Path(mname).stem)
            if clip not in names or mname not in vids[sp]:
                print('  skip %s (not in phaseB_ready / video.tar)' % clip); continue
            i = names.index(clip)
            gt = z['lm_res_raw'][int(off[i]):int(off[i + 1])].astype(np.float32)
            A, meta = g['extract_clip'](vids[sp][mname], clip, {})
            x = A['landmark_norm'][:, :, :2].astype(np.float32).reshape(-1, 136)
            T = len(x); N = int((T - 1) / 30 * FPS62) + 1
            mine = to62(x, N) - ref                                   # 2a: R.append(to62(x, N) - ref) for lm_res_raw
            n = min(len(mine), len(gt))
            e = np.linalg.norm((mine[:n] - gt[:n]).reshape(n, 68, 2), axis=2)
            all_e.append(e.ravel())
            rows.append(dict(clip=clip, frames_30fps=T, src_fps=round(float(meta['src_fps']), 3), detected=meta['n_detected'],
                             mean=float(np.nanmean(e)), max=float(np.nanmax(e))))
            print('  %-30s %3d frames | mean %.5f | max %.5f' % (clip, T, rows[-1]['mean'], rows[-1]['max']))
    e = np.concatenate(all_e)
    ok = np.nanmean(e) < THRESH_MEAN and np.nanmax(e) < THRESH_MAX
    res = dict(mediapipe=mediapipe_version(), model=g['MODEL_URL'], n_clips=len(rows), mean=float(np.nanmean(e)),
               max=float(np.nanmax(e)), p99=float(np.nanpercentile(e, 99)), threshold=dict(mean=THRESH_MEAN, max=THRESH_MAX),
               passed=bool(ok), clips=rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'mediapipe_check.json').write_text(json.dumps(res, indent=1))
    print('\n%d clips | mean %.5f (< %.3f) | max %.5f (< %.2f) | p99 %.5f -> %s'
          % (len(rows), res['mean'], THRESH_MEAN, res['max'], THRESH_MAX, res['p99'], 'PASS' if ok else 'FAIL - STOP'))
    if not ok:
        sys.exit(2)


# ----------------------------------------------------------------------------- extract
def video_landmarks(g, path):
    """mp4 -> (normalised (T, 68, 2) with NaN where no face, fps read, frame size, n_detected)."""
    A, meta = g['extract_clip'](Path(path).read_bytes(), Path(path).stem, {})
    import cv2
    cap = cv2.VideoCapture(str(path)); size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return A['landmark_norm'][:, :, :2].astype(np.float32), float(meta['src_fps']), size, int(meta['n_detected']), meta


def image_landmarks(g, path):
    """Input image -> normalised (68, 2), through the same resize as decode_frames and the same detector."""
    import cv2
    fr = cv2.resize(cv2.imread(str(path)), (g['FRAME_W'], g['FRAME_H']), interpolation=cv2.INTER_AREA)
    d = g['parse_detection'](g['run_detector'](cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    return None if d is None else d['norm'][:, :2].astype(np.float32)


def cmd_extract(a):
    g, to62, sim = load_0b(Path(a.out) / 'mp_work'), load_to62(), load_similarity_fit()
    ready = Path(a.drive) / 'Data_emotion_mead_task_b' / 'phaseB_ready'
    inp, runs = Path(a.inputs), Path(a.runs)
    clips = list(csv.DictReader(open(inp / 'list.csv', encoding='utf-8')))
    refs = {sp: np.load(ready / (sp + '.npz'))['ref'].astype(np.float32) for sp in {c['speaker'] for c in clips}}
    print('mediapipe %s | %d clips | methods %s' % (mediapipe_version(), len(clips), a.methods))
    # one Procrustes per speaker image (the same image is the input of every clip of that speaker)
    fits = {}
    for sp in sorted(refs):
        lm = image_landmarks(g, inp / 'images' / (sp + '.png'))
        if lm is None:
            print('  !! no face in input image %s' % sp); continue
        s, R, t = sim(lm, refs[sp].reshape(68, 2))
        res = np.linalg.norm(s * lm @ R + t - refs[sp].reshape(68, 2), axis=1)
        fits[sp] = (s, R, t)
        print('  image %s: similarity to ref s=%.4f, rot=%.2f deg, residual mean %.4f'
              % (sp, s, np.degrees(np.arctan2(R[0, 1], R[0, 0])), res.mean()))
    out, meta, fails = {}, [], []
    for m in a.methods:
        for c in clips:
            cid, sp = c['clip_id'], c['speaker']
            f = runs / m / (cid + '.mp4')
            rec = dict(method=m, clip_id=cid, speaker=sp, split=c['split'])
            if not f.exists():
                rec.update(status='missing mp4'); fails.append(rec); meta.append(rec); continue
            try:
                x, fps, size, n_det, _ = video_landmarks(g, f)
            except Exception as e:                                    # unreadable video
                rec.update(status='decode error: %s' % e); fails.append(rec); meta.append(rec); continue
            T = len(x); rec.update(frames=T, fps=round(fps, 3), size='%dx%d' % size, detected=n_det)
            if abs(size[0] / size[1] - 16 / 9) > 0.01:                 # normalize_landmarks assumes the 640x360 aspect
                rec['warning'] = 'aspect %dx%d != 16:9' % size
            if sp not in fits or n_det < T:
                rec.update(status='no face in %d / %d frames' % (T - n_det, T) if sp in fits else 'no face in input image')
                fails.append(rec); meta.append(rec); continue
            s, R, t = fits[sp]
            xy = s * x @ R + t                                        # one fit per clip (from the input image)
            N = int((T - 1) / fps * FPS62) + 1                        # 2a: N = int((T - 1) / 30 * FPS) + 1
            out['%s/%s' % (m, cid)] = (to62(xy.reshape(T, 136), N, src_fps=fps) - refs[sp]).astype(np.float32)
            rec.update(status='ok', frames_62p5=N, audio_s=float(c['audio_s']), video_s=round(T / fps, 3))
            meta.append(rec)
        n_ok = sum(r['status'] == 'ok' for r in meta if r['method'] == m)
        fps_seen = sorted({r['fps'] for r in meta if r['method'] == m and 'fps' in r})
        print('%-9s %d / %d clips ok | fps read: %s' % (m, n_ok, len(clips), fps_seen))
    dst = Path(a.out) / 'baseline_landmarks.npz'
    np.savez_compressed(dst, **out, _meta=json.dumps(dict(
        mediapipe=mediapipe_version(), model=g['MODEL_URL'], fps_out=FPS62,
        pipeline='0b decode_frames/FaceLandmarker/LMK68/normalize_landmarks -> 5 similarity_fit (input image -> ref, '
                 'one per clip) -> 2a to62 (video fps) -> minus ref', clips=meta)))
    print('\n%s | %.1f MB | %d arrays' % (dst, dst.stat().st_size / 1e6, len(out)))
    if fails:
        print('\nClips with failed landmark extraction (%d):' % len(fails))
        for r in fails:
            print('  %-9s %-30s %s' % (r['method'], r['clip_id'], r['status']))


# ----------------------------------------------------------------------------- report
def cmd_report(a):
    runs, col = Path(a.runs), Path(a.colab)
    pick = lambda pat: sorted(col.glob(pat))[-1]                     # newest file of each kind
    summ = list(csv.DictReader(open(pick('baselines_158_summary_*.csv'), encoding='utf-8')))
    wil = list(csv.DictReader(open(pick('baselines_158_wilcoxon_*.csv'), encoding='utf-8')))
    sens = list(csv.DictReader(open(pick('baselines_158_sensitivity_*.csv'), encoding='utf-8')))
    lags = json.loads(pick('baselines_158_lags_*.json').read_text())
    params = json.loads((runs / 'params.json').read_text()) if (runs / 'params.json').exists() else {}
    timing = {}
    for m in ('sadtalker', 'wav2lip'):
        f = runs / ('%s_timing.csv' % m)
        if f.exists():
            x = np.array([float(r['s_per_audio_s']) for r in csv.DictReader(open(f))])
            timing[m] = dict(mean=float(x.mean()), median=float(np.median(x)), n=len(x))
    cols = [('LMD', 'LMD ↓', 4), ('M_LMD_mieng', 'M-LMD ↓', 4), ('Tuong_quan', 'Mouth corr. ↑', 3), ('Bien_do', 'Amplitude (1)', 3),
            ('Khep_khi_im', 'Closure in silence ↓', 4), ('Emo_Acc', 'Emo-Acc ↑', 3)]
    systems = list(dict.fromkeys(r['system'] for r in summ))
    val = {(r['system'], r['metric']): r for r in summ}
    out_rows = []
    for s in systems:
        d = lags.get(s, {})
        main_l = d.get('used_in_main_row')                            # ELSA main row keeps lag 0 as in the paper
        row = dict(system=s, lag=d.get('lag', '') if main_l is None else main_l,
                   lag_shift_frames=d.get('shift', '') if main_l is None else 0,
                   params=val.get((s, 'params'), {}).get('value', ''))
        for k, _, _ in cols:
            r = val.get((s, k))
            row[k] = r['value'] if r else 'n/a'
            row[k + '_ci_lo'] = r['ci_lo'] if r else ''
            row[k + '_ci_hi'] = r['ci_hi'] if r else ''
        out_rows.append(row)
    res = REPO / 'results' / 'baselines_158.csv'
    res.parent.mkdir(exist_ok=True)
    with open(res, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)

    def fmt(r, k, d):
        if r[k] in ('n/a', ''):
            return 'n/a'
        v = '%.*f' % (d, float(r[k]))
        return v + (' [%.*f, %.*f]' % (d, float(r[k + '_ci_lo']), d, float(r[k + '_ci_hi'])) if r[k + '_ci_lo'] not in ('', 'nan') else '')

    print('**158 clips, neutral / angry / happy (MEAD test). 95% CI in brackets (1000 bootstrap resamples over clips, '
          'the same for every system).**\n')
    print('| System | Lag (frames @62.5) | ' + ' | '.join(c[1] for c in cols) + ' |')
    print('| :--- | ---: |' + ' ---: |' * len(cols))
    for r in out_rows:
        print('| %s | %s | %s |' % (r['system'], r['lag_shift_frames'], ' | '.join(fmt(r, k, d) for k, _, d in cols)))
    print('\nLag rule (notebook 5, MakeItTalk): best mouth-opening correlation on the 40 validation clips, '
          'shift -20..+16 frames at 62.5 fps (= MakeItTalk lag -12..24); 0 = no shift.')
    print('\n**Wilcoxon signed-rank, paired by clip, two-sided; Holm over the 3 comparisons per metric**\n')
    print('| Metric | Comparison | n clips | median ELSA | median baseline | p | p (Holm) |')
    print('| :--- | :--- | ---: | ---: | ---: | ---: | ---: |')
    for r in wil:
        print('| %s | ELSA vs %s | %s | %.4g | %.4g | %.3g | %.3g |' % (r['metric'], r['baseline'], r['n'], float(r['median_elsa']),
                                                                  float(r['median_base']), float(r['p']), float(r['p_holm'])))
    print('\n**Sensitivity (separate): per-frame similarity fit on brows / nose / eyes for every system AND the real '
          'landmarks; geometric metrics only, Emo-Acc n/a**\n')
    print('| System | ' + ' | '.join(c[1] for c in cols[:5]) + ' |')
    print('| :--- |' + ' ---: |' * 5)
    for r in sens:
        print('| %s | %s |' % (r['system'], ' | '.join('%.*f' % (d, float(r[k])) if r.get(k) not in (None, '', 'n/a') else 'n/a'
                                                     for k, _, d in cols[:5])))
    print('\n**Parameters (counted on the models loaded from the official checkpoints)**\n')
    for m, d in params.items():
        for k, v in d.items():
            print('- %s / %s: %.2fM' % (m, k, v / 1e6))
        print('- **%s total = sum of the networks above (all used at inference): %.2fM**' % (m, sum(d.values()) / 1e6))
    print('\n**Run time per second of audio (Colab T4, model loading excluded)**\n')
    for m, d in timing.items():
        print('- %s: mean %.3f s, median %.3f s (%d clips)' % (m, d['mean'], d['median'], d['n']))
    ex = Path(a.extract_dir or runs) / 'baseline_landmarks.npz'
    if ex.exists():
        meta = json.loads(str(np.load(ex)['_meta']))
        bad = [r for r in meta['clips'] if r['status'] != 'ok']
        print('\nLandmark extraction: mediapipe %s; %d failed clips%s' % (meta['mediapipe'], len(bad), ':' if bad else ''))
        for r in bad[:3]:
            print('  %s %s: %s' % (r['method'], r['clip_id'], r['status']))
    print('\nSaved:', res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check'); p.add_argument('--drive', required=True); p.add_argument('--mead', required=True)
    p.add_argument('--out', help='default <LOCAL_OUT>/baselines')
    p = sub.add_parser('extract'); p.add_argument('--drive', required=True); p.add_argument('--inputs', required=True)
    p.add_argument('--runs', required=True, help='folder with sadtalker/ and wav2lip/ mp4 (downloaded from Drive)')
    p.add_argument('--methods', nargs='+', default=['sadtalker', 'wav2lip']); p.add_argument('--out', help='default <LOCAL_OUT>/baselines')
    p = sub.add_parser('report'); p.add_argument('--runs', required=True); p.add_argument('--colab', required=True)
    p.add_argument('--extract_dir', help='default <LOCAL_OUT>/baselines')
    a = ap.parse_args()
    for k in ('out', 'extract_dir'):                                  # not given -> <LOCAL_OUT>/baselines
        if k in vars(a) and getattr(a, k) is None:
            setattr(a, k, str(local_path('LOCAL_OUT') / 'baselines'))
    dict(check=cmd_check, extract=cmd_extract, report=cmd_report)[a.cmd](a)


if __name__ == '__main__':
    main()
