"""Bo video khao sat MOS (chay ben trong notebook 3 qua run_nb3.py job 'mos'; g = globals cua notebook).
S1 = ELSA day du (ve lai mieng + may + mi, cam xuc SpEx-KD, chuoi demo + gain 1.85)
S2 = ELSA chi warp (tat moi co ve lai, cung cam xuc / gain)
S3 = landmark MakeItTalk goc qua cung bo ve lai nhu S1, neutral (mos_s3_job.py) | S4 = Wav2Lip (tro thu A)
Thu muc: C:\\hien\\mos\\videos (ten mu), _key\\ (bang khoa + ban render goc), wav2lip_inputs\\, play_order.csv"""
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

A, EA = g['A'], g['EA']
MOS = Path(r'C:\hien\mos')
VID, KEY, RAW, W2L = MOS / 'videos', MOS / '_key', MOS / '_key' / 'raw', MOS / 'wav2lip_inputs'
for d in (VID, KEY, RAW, W2L):
    d.mkdir(parents=True, exist_ok=True)
FF = (__import__('os').environ.get('FFMPEG') or __import__('shutil').which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
E2V = Path(r'G:\.shortcut-targets-by-id\1KaV-u_oFzL06sQpotcZhWayVLKmoq0ZS\DAT301m (1)\Data_student\Data_emotion_mead_task_b\phaseF_e2v')
CHARS = ['anime_girl', 'boy', 'girl2']
SPK = ['M032', 'M037', 'W018', 'W019']
EMO = ['neutral', 'happy', 'angry']
SIZE, FPS = 512, 25
g['SHOW_EMOTION_LABEL'] = False

clips = list(csv.DictReader(open(MOS / 'audio_list.csv', encoding='utf-8')))
for c in clips:                                       # xoay vong kieu o vuong Latin: moi nhan vat 4 cau, du 3 cam xuc
    c['clip'] = c['file'][:-4]
    c['char'] = CHARS[(SPK.index(c['slot_speaker']) + EMO.index(c['emotion'])) % 3]   # theo o (speaker du kien)


def run(cmd):
    subprocess.run([FF, '-y', '-loglevel', 'error'] + cmd, check=True)


def probe(p):
    r = subprocess.run([FF, '-i', str(p)], capture_output=True, text=True).stderr
    t = [l for l in r.splitlines() if 'Duration' in l][0].split('Duration:')[1].split(',')[0].strip().split(':')
    vs = [l for l in r.splitlines() if 'Video:' in l]
    return dict(dur=int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2]), audio=any('Audio:' in l for l in r.splitlines()),
                size=vs[0].split(',')[2].strip().split(' ')[0] if vs else '', fps=[x.strip() for x in vs[0].split(',') if 'fps' in x][0] if vs else '')


def to512(src, dst):
    run(['-i', str(src), '-vf', 'scale=%d:%d:flags=area' % (SIZE, SIZE), '-r', str(FPS), '-c:v', 'libx264', '-preset', 'veryfast',
         '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', str(dst)])


# ------------------------------------------------------------ cam xuc tu SpEx-KD (phaseG_final) tren emotion2vec+ cua chinh clip
BR, F2M = g['get_branch']('phaseG')
SER = g['SER_CFG']
_feat = {}


def spex(c):
    sp = c['speaker']
    if sp not in _feat:
        _feat[sp] = dict(np.load(E2V / (sp + '.npz'), allow_pickle=True))
    z = _feat[sp]
    key = c['e2v_clip']                                   # ghep audio <-> clip theo thu tu nhom (nhu notebook 6)
    i = list(z['names']).index(key)
    emb = EA.pool(z['chunks'], z['counts'], int(z['coff'][i]), int(z['coff'][i + 1]))[None].astype(np.float32)
    mask = torch.full((8,), float('-inf')); mask[F2M['keep']] = 0.0
    with torch.no_grad():
        Sx, Lg, Lv = BR(torch.tensor(emb)); Lg = Lg + mask
        P = torch.softmax(Lg / SER.get('f2_temp', 1.0), -1).numpy()
        cond = BR.cond_vector(Lg, Lv).numpy()
    head = F2M.get('e2v_head')
    if head is not None:                                     # 'phaseG+e2v' nhu demo: tron lop phan loai goc emotion2vec+
        Ph = EA.head_probs(head, emb) * np.isin(np.arange(8), F2M['keep']); Ph /= max(Ph.sum(), 1e-8)
        P = (1 - SER.get('ens_w', 0.5)) * P + SER.get('ens_w', 0.5) * Ph
    P = P[0].copy(); P[0] *= SER.get('neutral_weight_f2', 1.0); P /= P.sum()
    e = A.EMO_NAMES[int(P.argmax())]; lvl = float(np.clip(Lv.numpy()[0], 1, 3))
    spec = A.focus_code(e, lvl) if g['FOCUS'] else (e if e == 'neutral' else (e, lvl))
    static = Sx.numpy() * F2M['zero_lmd']
    dur = float(c['duration_s'])
    f2 = dict(centers=np.array([0.0, dur + 1.0]), static=np.repeat(static, 2, 0), cond=np.repeat(cond, 2, 0))
    return spec, e, lvl, f2, P


# ------------------------------------------------------------ render S1 / S2
ready = {}
pred = {}
for c in clips:
    nm = c['char']
    im = __import__('cv2').cvtColor(__import__('cv2').imread(str(MOS / 'characters' / (nm + '.png'))), 4)   # BGR -> RGB
    P11 = [tuple(p) for p in json.loads((MOS / 'characters' / (nm + '_points.json')).read_text())['points']]
    g['setup_character'](nm, g['char_from_image'](im, P11, nm, True, True), verbose=False)
    y = A.load_wav_16k(str(MOS / 'audio' / c['file']))
    spec, e, lvl, f2, P = spex(c)
    pred[c['clip']] = dict(spec=str(spec), emo=e, level=round(lvl, 2), probs=np.round(P, 3).tolist())
    for sysn, flag in (('S1', True), ('S2', False)):
        _, v = g['render'](y, timeline=[(0.0, spec)], name='%s_%s' % (sysn, c['clip']), f2=f2, label=False, verbose=False,
                           mouth_redraw=flag, brow_redraw=flag)
        dst = RAW / ('%s__%s__%s.mp4' % (sysn, c['clip'], nm))
        to512(v, dst)
        ready[(sysn, c['clip'])] = dst
    print('%-22s %-10s that=%-7s SpEx-KD=%s' % (c['clip'], nm, c['emotion'], spec))

# ------------------------------------------------------------ 2 video kiem tra chu y: am thanh tre 1 s so voi khau hinh
att = sorted(clips, key=lambda c: -float(c['duration_s']))[:2]
for k, c in enumerate(att):
    src = ready[('S1', c['clip'])]
    dst = RAW / ('ATT%d__%s__%s.mp4' % (k + 1, c['clip'], c['char']))
    d = probe(src)['dur']
    run(['-i', str(src), '-map', '0:v', '-map', '0:a', '-af', 'adelay=1000:all=1,apad', '-t', '%.3f' % d,   # chen 1 s im lang that
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', str(dst)])
    ready[('ATT%d' % (k + 1), c['clip'])] = dst

# ------------------------------------------------------------ dau vao Wav2Lip cho tro thu A
import cv2
for c in clips:
    base = '%s__%s' % (c['clip'], c['char'])
    img = cv2.resize(cv2.imread(str(MOS / 'characters' / (c['char'] + '.png'))), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(W2L / (base + '.png')), img)
    shutil.copy2(MOS / 'audio' / c['file'], W2L / (base + '.wav'))
(W2L / 'HUONG_DAN_WAV2LIP.txt').write_text(
    'Dau vao Wav2Lip cho khao sat MOS (12 cap). Moi cap: <cau>__<nhan vat>.png (512x512) + <cau>__<nhan vat>.wav (16 kHz mono).\n'
    'Yeu cau dau ra: 1 video cho moi cap, TEN GIONG HET ten cap + .mp4, vd M032_neutral_L1_025__anime_girl.mp4\n'
    '  - 512x512, 25 fps, h264 + am thanh (aac), do dai = do dai file wav, KHONG chu / logo / nhan.\n'
    '  - Dung nguyen anh tinh (khong cat mat sang kich thuoc khac); neu Wav2Lip can --resize_factor / --pads thi ghi lai tham so.\n'
    '  - Dat tat ca vao thu muc C:\\hien\\mos\\wav2lip_outputs\\ va gui kem ghi chu: phien ban Wav2Lip, checkpoint (wav2lip / wav2lip_gan), tham so.\n'
    'Danh sach cap:\n' + '\n'.join('  %s__%s' % (c['clip'], c['char']) for c in clips) + '\n', encoding='utf-8')

# ------------------------------------------------------------ ten mu + bang khoa + thu tu trinh chieu
rng = random.Random(301)
ALPH = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
items = [(s, c) for s in ('S1', 'S2', 'S3', 'S4') for c in clips] + [('ATT%d' % (k + 1), c) for k, c in enumerate(att)]
codes = set()
rows = []
for s, c in items:
    while True:
        code = 'v_' + ''.join(rng.choice(ALPH) for _ in range(4))
        if code not in codes:
            codes.add(code); break
    src = ready.get((s, c['clip']))
    rows.append(dict(code=code, video=code + '.mp4', system=s if not s.startswith('ATT') else 'ATTENTION',
                     note='am thanh tre 1 s (S1)' if s.startswith('ATT') else '', clip=c['clip'], character=c['char'],
                     speaker=c['speaker'], emotion_true=c['emotion'], level_true=c['level'],
                     emotion_spexkd=pred[c['clip']]['spec'] if s in ('S1', 'S2') or s.startswith('ATT') else '',
                     mead_source=c['mead_source'], status='ready' if src else 'pending', raw_file=src.name if src else
                     ('S3: phaseH/mos_makeittalk' if s == 'S3' else 'S4: wav2lip_outputs\\%s__%s.mp4' % (c['clip'], c['char']))))
    if src:
        shutil.copy2(src, VID / (code + '.mp4'))
with open(KEY / 'key.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(KEY / 'spexkd_predictions.json').write_text(json.dumps(pred, indent=1, ensure_ascii=False))

order = None
for _ in range(5000):                                  # khong de 2 video cung cau dung lien nhau
    o = rows[:]; rng.shuffle(o)
    if all(o[i]['clip'] != o[i + 1]['clip'] for i in range(len(o) - 1)):
        order = o; break
assert order is not None
with open(MOS / 'play_order.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['order', 'video'])
    for i, r in enumerate(order):
        w.writerow([i + 1, r['video']])

# ------------------------------------------------------------ kiem tra cuoi
bad = []
for r in rows:
    if r['status'] != 'ready':
        continue
    p = probe(VID / r['video'])
    da = float(next(c for c in clips if c['clip'] == r['clip'])['duration_s'])
    ok = p['audio'] and p['size'] == '%dx%d' % (SIZE, SIZE) and p['fps'].startswith('25') and abs(p['dur'] - da) <= 0.05
    if not ok:
        bad.append((r['video'], p, da))
n_ready = sum(r['status'] == 'ready' for r in rows)
print('\nVIDEO san sang: %d | cho S3/S4: %d | tong ke hoach: %d | loi kiem tra: %d %s' % (
    n_ready, len(rows) - n_ready, len(rows), len(bad), bad[:3]))
