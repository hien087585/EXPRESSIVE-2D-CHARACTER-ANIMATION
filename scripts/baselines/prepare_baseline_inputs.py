"""Inputs for the SadTalker / Wav2Lip baselines (Table 2 protocol) -> baseline_inputs.zip.

Clips are chosen with the code of notebooks/5_phaseE_benchmark.ipynb (load_eval, audio_map, load_m4a are
read from the notebook, not rewritten):
  * test: the 480 MakeItTalk clips (rng(SEED).choice, MIT_CLIPS_PER_SPK per test speaker), restricted to clips
    with emotion2vec+ features and to the SpEx-KD classes KEEP3 (neutral / angry / happy) -> 158 clips;
  * val : the MIT_VAL_CLIPS clips used to search the best lag for MakeItTalk (same rng), used for every system.
Audio: the same m4a member and the same ffmpeg call (16 kHz mono) as MakeItTalk got.
Neutral image per speaker: no frame was used as the neutral face (ref is a template from derived.npz), so the
middle frame of the first front / neutral / level_1 clip that is NOT in any evaluated list is used (logged).

<DRIVE> is only read. Output goes to --out (default <LOCAL_OUT>/baselines, see scripts/local_paths.py).

    python scripts/baselines/prepare_baseline_inputs.py --drive "<...>/Data_student" --mead "<...>/MEAD"
"""
import argparse
import ast
import csv
import json
import shutil
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from local_paths import local_path  # noqa: E402
NB5 =REPO / 'notebooks' / '5_phaseE_benchmark.ipynb'


def nb_functions(names):
    """Source of the named top-level functions of notebook 5 (exactly as written there)."""
    out = {}
    for c in json.loads(NB5.read_text(encoding='utf-8'))['cells']:
        src = ''.join(c['source'])
        if c['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in out:
                out[node.name] = ast.get_source_segment(src, node)
    assert set(out) == set(names), 'missing in notebook 5: %s' % (set(names) - set(out))
    return '\n\n\n'.join(out[n] for n in names)


def nb_constant(name):
    for c in json.loads(NB5.read_text(encoding='utf-8'))['cells']:
        for line in ''.join(c['source']).splitlines():
            if line.startswith(name + ' ') and '=' in line:
                return ast.literal_eval(line.split('=', 1)[1].split('#')[0].strip())
    raise KeyError(name)


def scan_video_tar(mead, sp, exclude, tmp):
    """One pass over the headers of video.tar (plain tar, data skipped):
    * names of all front clips, as 0b named them (<sp>__front__<emo>__L<l>__<nnn>) - the clip list that notebook 5
      read from <sp>_landmark_package.zip, which is no longer in stage_b_mead_out (0b: 0 failed clips);
    * middle frame of the first front/neutral/level_1 video whose clip is not evaluated."""
    t0 = time.time()
    cache = tmp.parent / '_scan' / sp                                  # result kept: a rerun does not read the tar again
    if (cache / 'scan.json').exists():
        d = json.loads((cache / 'scan.json').read_text())
        if d['exclude_hash'] == hash_names(exclude):
            return cv2.imread(str(cache / 'frame.png')), d['front'], d['info']
    cache.mkdir(parents=True, exist_ok=True)
    front, cands = [], []
    with tarfile.open(mead / sp / 'video.tar', 'r:') as tf:            # whole tar: some (e.g. M037) are not grouped by angle
        while True:
            m = tf.next()
            if m is None:
                break
            if m.isfile() and m.name.startswith('video/front/') and m.name.endswith('.mp4'):
                q = Path(m.name).parts                                     # video/front/<emo>/level_<l>/<nnn>.mp4
                front.append('%s__front__%s__L%s__%s' % (sp, q[2], q[3].replace('level_', ''), Path(q[4]).stem))
                if m.name.startswith('video/front/neutral/level_1/'):
                    cands.append(m)
        cands.sort(key=lambda m: m.name)
        for m in cands:
            clip = '%s__front__neutral__L1__%s' % (sp, Path(m.name).stem)
            if clip not in exclude:
                break
        else:
            raise RuntimeError('%s: every front neutral L1 clip is evaluated' % sp)
        f = tmp / ('%s.mp4' % clip)
        f.write_bytes(tf.extractfile(m).read())
    cap = cv2.VideoCapture(str(f))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, fr = cap.read(); cap.release()
    assert ok, f
    info = dict(speaker=sp, source_clip=clip, tar_member='%s/video.tar:%s' % (sp, m.name), frame_index=n // 2,
                n_frames=n, fps=round(fps, 3), size=[fr.shape[1], fr.shape[0]], n_front_clips=len(front),
                seconds=round(time.time() - t0, 1))
    cv2.imwrite(str(cache / 'frame.png'), fr)
    (cache / 'scan.json').write_text(json.dumps(dict(exclude_hash=hash_names(exclude), front=front, info=info)))
    return fr, front, info


def hash_names(names):
    return __import__('hashlib').sha256('\n'.join(sorted(names)).encode()).hexdigest()


class FrontClipList:
    """Stands in for zipfile.ZipFile(<sp>_landmark_package.zip) inside notebook 5's audio_map: infolist() gives
    '<sp>/clips/<clip>.npz' for every front clip found in video.tar (audio_map only reads these names)."""
    FRONT = {}

    def __init__(self, path):
        self.sp = Path(path).parent.name

    def infolist(self):
        return [SimpleNamespace(filename='%s/clips/%s.npz' % (self.sp, n)) for n in self.FRONT[self.sp]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', required=True, help='Data_student folder (read only)')
    ap.add_argument('--mead', required=True, help='MEAD folder with <speaker>/audio.tar, video.tar (read only)')
    ap.add_argument('--out', help='default <LOCAL_OUT>/baselines')
    args = ap.parse_args()
    drive, mead = Path(args.drive), Path(args.mead)
    out = Path(args.out) if args.out else local_path('LOCAL_OUT') / 'baselines'
    root = out / 'baseline_inputs'
    tmp = out / '_tmp'
    for d in (root / 'images', root / 'wav', tmp):
        d.mkdir(parents=True, exist_ok=True)

    READY = drive / 'Data_emotion_mead_task_b' / 'phaseB_ready'
    STAGE = drive / 'Data_emotion_mead_task_b' / 'stage_b_mead_out'
    E2V_DIR = drive / 'Data_emotion_mead_task_b' / 'phaseF_e2v'
    SPEX_CKPT = drive / 'student_a2l' / 'phaseF' / 'phaseG_final.pth'
    SEED, PER_SPK, N_VAL = nb_constant('SEED'), nb_constant('MIT_CLIPS_PER_SPK'), nb_constant('MIT_VAL_CLIPS')
    EMO_NAMES = nb_constant('EMO_NAMES')
    LEAD, W = 8, 18                                                    # notebook 5: LEAD, W = 8, 18
    ms = np.loadtxt(REPO / 'src' / 'dataset' / 'utils' / 'MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt').reshape(2, 80)
    g = dict(np=np, READY=READY, STAGE=STAGE, MEAD_SRC=mead, LEAD=LEAD, W=W, Path=Path,
             MEL_MEAN=ms[0].astype(np.float32), MEL_STD=np.maximum(ms[1], 1e-5).astype(np.float32),
             zipfile=SimpleNamespace(ZipFile=FrontClipList), tarfile=tarfile, subprocess=__import__('subprocess'))
    exec(nb_functions(['load_eval', 'audio_map', 'load_m4a']), g)
    SPLIT = json.loads((READY / 'split.json').read_text())
    print('SEED %d | MakeItTalk %d clip / test speaker | %d val clip' % (SEED, PER_SPK, N_VAL))

    DTE, DVA = g['load_eval'](SPLIT['test'], 'test'), g['load_eval'](SPLIT['val'], 'val')
    assert len(DTE['clips']) == 2652 and len(DVA['clips']) == 2592

    # --- same draws as notebook 5 Cell 5 (va_ids first, then te_ids) ---
    va_ids = list(np.random.default_rng(SEED).choice(len(DVA['clips']), min(N_VAL, len(DVA['clips'])), replace=False))
    rng_ = np.random.default_rng(SEED); te_ids = []
    for sp in SPLIT['test']:
        ids = [i for i, c in enumerate(DTE['clips']) if c['sp'] == sp]
        te_ids += sorted(rng_.choice(ids, min(PER_SPK, len(ids)), replace=False).tolist())

    # --- video.tar: front clip names (for audio_map) + neutral image per speaker (not one of the drawn clips) ---
    exclude = {DTE['clips'][ci]['name'] for ci in te_ids} | {DVA['clips'][ci]['name'] for ci in va_ids}
    sps = SPLIT['test'] + SPLIT['val']
    print('reading video.tar headers of %d speakers (Drive, a few minutes) ...' % len(sps))
    with ThreadPoolExecutor(len(sps)) as ex:
        scans = list(ex.map(lambda sp: scan_video_tar(mead, sp, exclude, tmp), sps))
    images = []
    for fr, front, info in scans:
        sp = info['speaker']
        FrontClipList.FRONT[sp] = front
        D = DTE if sp in SPLIT['test'] else DVA
        known = {c['name'] for c in D['clips'] if c['sp'] == sp}
        assert known <= set(front), '%s: %d phaseB_ready clips not in video.tar' % (sp, len(known - set(front)))
        cv2.imwrite(str(root / 'images' / (sp + '.png')), fr)
        images.append(info)
        print('image %s <- %s frame %d / %d (%dx%d) | %d front clips | %.0f s'
              % (sp, info['source_clip'], info['frame_index'], info['n_frames'], *info['size'], len(front), info['seconds']))

    amaps = {}
    for sp in SPLIT['test'] + SPLIT['val']:
        amaps[sp] = g['audio_map'](sp)
    mit_idx = sorted(ci for ci in te_ids if DTE['clips'][ci]['name'] in amaps[DTE['clips'][ci]['sp']][0])
    va_ok = [ci for ci in va_ids if DVA['clips'][ci]['name'] in amaps[DVA['clips'][ci]['sp']][0]]
    print('MakeItTalk test clips: %d (Table 2: 480) | val clips with audio: %d' % (len(mit_idx), len(va_ok)))
    assert len(mit_idx) == 480 and len(va_ok) == len(va_ids)

    # --- clips with emotion2vec+ (HAS) and SpEx-KD classes (KEEP3), as EVAL_MIT / MIT3 ---
    keep3 = [int(k) for k in torch.load(str(SPEX_CKPT), map_location='cpu', weights_only=False)['keep']]
    has = set()
    for sp in SPLIT['test']:
        with np.load(E2V_DIR / (sp + '.npz')) as f:
            has |= {str(n) for n, h in zip(f['names'], f['has']) if h}
    eval_mit = [ci for ci in mit_idx if DTE['clips'][ci]['name'] in has]
    mit3 = [ci for ci in eval_mit if DTE['clips'][ci]['emo'] in keep3]
    print('KEEP3 = %s | with emotion2vec+: %d / 480 | subset: %d clips (Table 2: 158)'
          % ([EMO_NAMES[k] for k in keep3], len(eval_mit), len(mit3)))
    assert len(mit3) == 158, len(mit3)

    # --- audio: same member, same ffmpeg call as MakeItTalk ---
    rows = []
    for split, D, ids in (('test', DTE, mit3), ('val', DVA, sorted(va_ok))):
        for ci in ids:
            c = D['clips'][ci]
            amap, tf = amaps[c['sp']]
            y = g['load_m4a'](tf, amap[c['name']], tmp=str(tmp / 'a'))
            shutil.copy(tmp / 'a' / 'x.wav', root / 'wav' / (c['name'] + '.wav'))
            rows.append(dict(clip_id=c['name'], speaker=c['sp'], split=split, clip_index=int(ci),
                             emotion=EMO_NAMES[c['emo']], level=c['lvl'], audio_s=round(len(y) / 16000, 3),
                             gt_frames_62p5=c['n'], audio_member=amap[c['name']]))
    print('wav: %d files' % len(rows))

    with open(root / 'list.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (root / 'info.json').write_text(json.dumps(dict(
        protocol='notebook 5 (Table 2): test = MakeItTalk subset x emotion2vec+ x KEEP3; val = MakeItTalk lag-search clips',
        seed=SEED, keep3=[EMO_NAMES[k] for k in keep3], n_test=len(mit3), n_val=len(va_ok),
        mit_idx_480=[int(i) for i in mit_idx], test_158=[int(i) for i in mit3], val_40=[int(i) for i in va_ids],
        audio='MEAD audio.tar m4a -> ffmpeg -ar 16000 -ac 1 (load_m4a of notebook 5)',
        clip_list_source=('audio_map of notebook 5 run unchanged; its clip-name list comes from the front videos of '
                          'video.tar (the <sp>_landmark_package.zip files are no longer in stage_b_mead_out)'),
        ffmpeg=__import__('subprocess').run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0],
        images=images), indent=1), encoding='utf-8')

    zp = out / 'baseline_inputs.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)
    print('\n%s | %.1f MB | %d test + %d val clips, %d images' % (zp, zp.stat().st_size / 1e6, len(mit3), len(va_ok), len(images)))


if __name__ == '__main__':
    main()
