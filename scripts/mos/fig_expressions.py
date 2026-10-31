"""Fig. 8 - bieu cam: 3 hang = anime_girl, anime_boy, anime_girl2 (examples_anime, nhom tu ve);
cot 1 "Input image", tiep theo Neutral / Angry / Happy do SpEx-KD (phaseG_final) du doan tu giong noi.
Cau dung (chung cho 3 nhan vat): 3 cau MOS ma SpEx-KD du doan neutral / angry_2 / happy_2 (muc cao nhat che do gon).
Cam xuc: SpEx-KD tren dac trung emotion2vec+ TINH SAN cua tung cau (Data_emotion_mead_task_b/phaseF_e2v, nhu video MOS S1)
vi may nay khong chay duoc emotion2vec+. Render nhu S1: ve lai mieng + may + mat, chuoi demo + gain x1.85.
Moi hang 1 khung chung cho 3 cot: mieng mo ro o ca 3 (min do mo lon nhat), khong chop mat, bo 10 khung dau / cuoi.
Chay trong notebook 3 qua run_nb3.py job 'script' (anh / diem / audio truyen vao khong dung, chi can de khoi tao):
  python run_nb3.py <nb3> <out> script <png> <pts.json> <wav> <giay> fig_expressions.py
-> <MOS_DIR>/_check/fig_expressions.png (300 dpi). Duong dan may: scripts/local_paths.py (ELSA_* hoac local_paths.json)."""
import csv
import json
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from local_paths import local_path  # noqa: E402
MOS = local_path('MOS_DIR')
EX = REPO / 'examples_anime'
E2V = local_path('DRIVE') / 'Data_emotion_mead_task_b' / 'phaseF_e2v'
OUT_PNG = MOS / '_check' / 'fig_expressions.png'
ROWS = [('anime_girl', 'anime_girl'), ('anime_boy', 'boy'), ('anime_girl2', 'girl2')]     # ten demo, ten file
COLS = [('Neutral', 'M032_neutral_L1_026'), ('Angry', 'M032_angry_L3_010'), ('Happy', 'W018_happy_L3_016')]
TILE, GAP, HEAD_H, PAD = 430, 12, 70, 20
font = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', s)
EA = g['EA']
g['SHOW_EMOTION_LABEL'] = False
assert g['MOUTH_GAIN'] is not None, 'can gain x1.85 nhu S1 (run_nb3 khong dat NB3_NO_GAIN)'

clips = {c['file'][:-4]: c for c in csv.DictReader(open(MOS / 'audio_list.csv', encoding='utf-8'))}
BR, F2M = g['get_branch']('phaseG')
SER = g['SER_CFG']


def spex(c):  # = mos_job.spex: SpEx-KD (+ lop phan loai goc emotion2vec+) tren embedding emotion2vec+ ca cau
    z = dict(np.load(E2V / (c['speaker'] + '.npz'), allow_pickle=True))
    i = list(z['names']).index(c['e2v_clip'])
    emb = EA.pool(z['chunks'], z['counts'], int(z['coff'][i]), int(z['coff'][i + 1]))[None].astype(np.float32)
    mask = torch.full((8,), float('-inf')); mask[F2M['keep']] = 0.0
    with torch.no_grad():
        Sx, Lg, Lv = BR(torch.tensor(emb)); Lg = Lg + mask
        P = torch.softmax(Lg / SER.get('f2_temp', 1.0), -1).numpy()
        cond = BR.cond_vector(Lg, Lv).numpy()
    head = F2M.get('e2v_head')
    if head is not None:
        Ph = EA.head_probs(head, emb) * np.isin(np.arange(8), F2M['keep']); Ph /= max(Ph.sum(), 1e-8)
        P = (1 - SER.get('ens_w', 0.5)) * P + SER.get('ens_w', 0.5) * Ph
    P = P[0].copy(); P[0] *= SER.get('neutral_weight_f2', 1.0); P /= P.sum()
    e = A.EMO_NAMES[int(P.argmax())]; lvl = float(np.clip(Lv.numpy()[0], 1, 3))
    spec = A.focus_code(e, lvl) if g['FOCUS'] else (e if e == 'neutral' else (e, lvl))
    dur = float(c['duration_s'])
    f2 = dict(centers=np.array([0.0, dur + 1.0]), static=np.repeat(Sx.numpy() * F2M['zero_lmd'], 2, 0), cond=np.repeat(cond, 2, 0))
    return spec, f2


SP = {clip: spex(clips[clip]) for _, clip in COLS}
print('SpEx-KD:', {clip: SP[clip][0] for _, clip in COLS})
Wd = PAD + 4 * TILE + 3 * GAP + PAD
Hd = PAD + HEAD_H + len(ROWS) * TILE + (len(ROWS) - 1) * GAP + PAD
img = Image.new('RGB', (Wd, Hd), 'white'); d = ImageDraw.Draw(img)
for c, t in enumerate(['Input image'] + [t for t, _ in COLS]):
    d.text((PAD + c * (TILE + GAP) + TILE // 2, PAD + HEAD_H // 2), t, fill='black', font=font(40), anchor='mm')
for r, (nm, stem) in enumerate(ROWS):
    im = cv2.cvtColor(cv2.imread(str(EX / (stem + '.png'))), cv2.COLOR_BGR2RGB)
    P11 = [tuple(p) for p in json.loads((EX / (stem + '_points.json')).read_text())['points']]
    g['setup_character'](nm, g['char_from_image'](im, P11, nm, True, True), verbose=False)
    base = g['CH'].img                                                   # anh dau vao da thu ve kich thuoc render (BGR)
    s = base.shape[0] / 512.0
    FACE = [int(v * s) for v in (140, 380, 136, 376)]
    MOUTH = [int(v * s) for v in (285, 345, 200, 312)]
    EYES = [int(v * s) for v in (205, 255, 170, 345)]
    F = []
    for _, clip in COLS:
        y = A.load_wav_16k(str(MOS / 'audio' / clips[clip]['file']))
        spec, f2 = SP[clip]
        fr, _ = g['render'](y, timeline=[(0.0, spec)], name='fig8', f2=f2, save=False, keep=True, label=False, verbose=False,
                            mouth_redraw=True, brow_redraw=True)
        F.append(np.stack(fr))
    n = min(len(f) for f in F)
    reg = lambda f, b: np.abs(f[:n, b[0]:b[1], b[2]:b[3]].astype(np.float32) - base[b[0]:b[1], b[2]:b[3]]).mean((1, 2, 3))
    mouth = np.min([reg(f, MOUTH) for f in F], 0)
    ok = np.all([reg(f, EYES) <= np.median(reg(f, EYES)) * 1.5 for f in F], 0)
    ok[:10] = ok[-10:] = False
    i = int(np.argmax(np.where(ok, mouth, -1)))
    print('%-12s khung %d (%.2f s) | do mo mieng nho nhat 3 cot %.2f' % (nm, i, i / g['OUT_FPS'], mouth[i]))
    y0 = PAD + HEAD_H + r * (TILE + GAP)
    for c, f in enumerate([base] + [F_[i] for F_ in F]):
        crop = np.ascontiguousarray(f[FACE[0]:FACE[1], FACE[2]:FACE[3], ::-1])
        img.paste(Image.fromarray(crop).resize((TILE, TILE), Image.LANCZOS), (PAD + c * (TILE + GAP), y0))
img.save(OUT_PNG, dpi=(300, 300))
print('->', OUT_PNG, img.size)
