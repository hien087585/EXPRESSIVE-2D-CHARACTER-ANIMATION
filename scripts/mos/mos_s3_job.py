"""S3 = landmark MakeItTalk goc qua CUNG bo ve lai nhu S1 (MOUTH_REDRAW + BROW_REDRAW + chop mat), cam xuc neutral ca cau.
Chay trong notebook 3 (g = globals). MakeItTalk khong co cam xuc -> khong gain 1.85, khong bieu cam tinh, khong hip,
khong mi gian; long may chi dong theo landmark MakeItTalk; chop mat cung lich seed nhu S1 (seed=0).
Dau vao: <MIT_DIR>\\<cau>_amp1.npy (N, 136) do dich so voi mat trung tinh MEAD cua speaker, 62.5 khung/giay
(dung y benchmark notebook 5: face_id = mat speaker, he so moi x1 = hang 'MakeItTalk goc (moi x1)' trong bang ket qua).
Do tre: khung k cua audio <-> khung MakeItTalk k + LEAD - 12 (mit_lag = 12); thieu o cuoi -> giu khung cuoi."""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

A = g['A']
MOS = Path(r'C:\hien\mos')
MIT_DIR = Path(ARGS_S3[0])
OUTD = Path(ARGS_S3[1]); OUTD.mkdir(parents=True, exist_ok=True)
ONLY = set(ARGS_S3[2].split(',')) if len(ARGS_S3) > 2 and ARGS_S3[2] else None
FF = (__import__('os').environ.get('FFMPEG') or __import__('shutil').which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
MIT_LAG, LEAD = 12, g['LEAD']
CHARS = ['anime_girl', 'boy', 'girl2']; SPK = ['M032', 'M037', 'W018', 'W019']; EMO = ['neutral', 'happy', 'angry']
g['SHOW_EMOTION_LABEL'] = False

clips = list(csv.DictReader(open(MOS / 'audio_list.csv', encoding='utf-8')))
READY = Path(r'G:\.shortcut-targets-by-id\1KaV-u_oFzL06sQpotcZhWayVLKmoq0ZS\DAT301m (1)\Data_student\Data_emotion_mead_task_b\phaseB_ready')
REFS = {sp: np.load(READY / (sp + '.npz'))['ref'].reshape(68, 2).astype(np.float64) for sp in SPK}   # chi doc


def sim_fit(src, dst):
    # dong dang (Umeyama) src -> dst: dst ~ s * src @ R + t
    ms, md = src.mean(0), dst.mean(0); a_, b_ = src - ms, dst - md
    U, S_, Vt = np.linalg.svd(a_.T @ b_)
    d = np.sign(np.linalg.det(U @ Vt)); D_ = np.diag([1, d])
    R_ = U @ D_ @ Vt; s_ = (S_ * np.diag(D_)).sum() / (a_ ** 2).sum()
    return s_, R_, md - s_ * ms @ R_


done = {}
for c in clips:
    clip = c['file'][:-4]
    if ONLY and clip not in ONLY:
        continue
    nm = CHARS[(SPK.index(c['slot_speaker']) + EMO.index(c['emotion'])) % 3]
    im = __import__('cv2').cvtColor(__import__('cv2').imread(str(MOS / 'characters' / (nm + '.png'))), 4)
    P11 = [tuple(p) for p in json.loads((MOS / 'characters' / (nm + '_points.json')).read_text())['points']]
    g['setup_character'](nm, g['char_from_image'](im, P11, nm, True, True), verbose=False)
    y = A.load_wav_16k(str(MOS / 'audio' / c['file']))
    disp = np.load(MIT_DIR / (clip + '_amp1.npy')).astype(np.float32)          # (N, 136) khong gian MEAD
    n = len(A.mel_autovc(y))                                                    # so khung 62.5 fps cua audio (nhu render)
    k = np.clip(np.arange(n) + LEAD - MIT_LAG, 0, len(disp) - 1)               # do tre benchmark; cuoi: giu khung cuoi
    dyn = disp[k]
    # khu tu the dau (nhanh 'speaker-aware' cua MakeItTalk xoay / dich ca mat; ELSA va bo warp khong co chuyen dong dau):
    # moi khung khop dong dang ve mat trung tinh tren cac diem it bi mieng anh huong (may 17-26, song mui 27-30, mat 36-47)
    ref = REFS[c['speaker']]; ST = list(range(17, 31)) + list(range(36, 48))
    X = ref[None] + dyn.reshape(n, 68, 2)
    for t in range(n):
        s_, R_, t_ = sim_fit(X[t, ST], ref[ST])
        X[t] = s_ * X[t] @ R_ + t_
    dyn = (X - ref[None]).reshape(n, 136).astype(np.float32)
    dyn = dyn - dyn.mean(0, keepdims=True)              # chi giu CHUYEN DONG (tru TB clip) - giong C4 o S2 (mead_residual)
    prof = dict(g['PROFILE']); prof['motion_gain'] = 1.0                        # khong khuech dai them
    trk = A.ExpressionTrack([(0.0, 'neutral')], g['BANK'], g['CH'], blink=prof.get('blink', True), seed=0,
                            overlays=prof.get('overlays', True))        # giong render() cua S1 (seed=0)
    q = trk.sample(n, A.FPS_MODEL)
    P = A.char_landmarks_track(g['CH'], g['FRAME'], q, dyn, prof)               # cung ham dua chuyen dong len nhan vat
    mc = {**(prof.get('mouth_ctrl') or {}), **g['MOUTH_LIMITS']}               # cung gioi han hinh hoc mieng nhu S2
    P = A.limit_mouth(P, g['BASE_PX'], open_ratio=mc['open_ratio'], jaw_drop=mc['jaw_drop'], max_width=mc.get('max_width'),
                      jaw_side=mc.get('jaw_side'))
    P = A.resample(P, A.FPS_MODEL, g['OUT_FPS'])
    P = A.smooth_contour(P, g['CH'].open_px)
    qo = trk.sample(len(P), g['OUT_FPS'])
    # cung bo ve lai nhu S1 (prep_face: mieng + may + mi, OPEN_SCALE / chuan hoa cuc tieu cuc bo / cong im lang giu nguyen)
    FDRAW, FX = g['prep_face'](P, qo, g['OUT_FPS'], True, True, y, n, 0)
    emo_max = {k[4:]: float(np.max(v)) for k, v in FX.items() if k.startswith('emo_')}
    raw = OUTD / ('S3_%s_768.mp4' % clip)
    sink = A.VideoSink(raw, g['OUT_FPS'], (g['CH'].img.shape[1], g['CH'].img.shape[0]), y)
    for i, p in enumerate(P):
        sink.write(g['PAINT'](FDRAW(i), p, qo, i, prof))
    sink.close()
    dst = OUTD / ('S3__%s__%s.mp4' % (clip, nm))
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(raw), '-vf', 'scale=512:512:flags=area', '-r', '25', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', str(dst)], check=True)
    raw.unlink()
    done[clip] = dst
    print('S3 %-22s %-10s MakeItTalk %d khung (%.2f s) | audio %d khung (%.2f s) | giu khung cuoi %d khung | chop %d | cam xuc max %s' % (
        clip, nm, len(disp), len(disp) / 62.5, n, n / 62.5, int((np.arange(n) + LEAD - MIT_LAG > len(disp) - 1).sum()),
        int((np.diff((FX['blink'] > 0.5).astype(int)) == 1).sum()), {k: round(v, 3) for k, v in emo_max.items()}))
