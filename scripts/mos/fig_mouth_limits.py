"""Fig. 2 - gioi han mieng: anime_girl (examples_anime), CHI WARP (tat moi co ve lai), gain x1.85 nhu S2.
Hang tren "Without mouth limits" (bo buoc A.limit_mouth), hang duoi "With mouth limits" (nhu demo);
5 cot neutral, angry_1, angry_2, happy_1, happy_2; cung audio, cung khung (khung lech nhieu nhat giua 2 hang o vung mieng-cam).
Chay trong notebook 3 qua run_nb3.py (g = globals notebook):
  python run_nb3.py <nb3> <out> script <examples_anime/anime_girl.png> <pts.json> <audio.wav> <giay> fig_mouth_limits.py
-> <MOS_DIR>/_check/fig_mouth_limits.png (300 dpi). Gain lay tu MOUTH_GAIN_PATH ma run_nb3 dat (nhu video MOS).
Duong dan may: scripts/local_paths.py (ELSA_* hoac local_paths.json)."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_paths import local_path  # noqa: E402
OUT_PNG = local_path('MOS_DIR') / '_check' / 'fig_mouth_limits.png'
COLS = [('neutral', 'Neutral'), ('angry_1', 'Angry (level 1)'), ('angry_2', 'Angry (level 2)'),
        ('happy_1', 'Happy (level 1)'), ('happy_2', 'Happy (level 2)')]
ROWS = ['Without mouth limits', 'With mouth limits']
TILE, GAP, LABEL_W, HEAD_H, PAD = 325, 10, 110, 70, 20
font = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', s)

g['SHOW_EMOTION_LABEL'] = False
assert g['MOUTH_GAIN'] is not None, 'can gain x1.85 nhu S2 (run_nb3 khong dat NB3_NO_GAIN)'
nm = 'anime_girl_warp'
g['setup_character'](nm, g['char_from_image'](img_rgb, pts[:5], nm, False, False), verbose=False)   # 5 diem: chi warp
H, W = g['CH'].img.shape[:2]
s = H / 512.0                                                           # anh 1024 -> 768 trong char_from_image
FACE = [int(v * s) for v in (140, 400, 126, 386)]                       # y0 y1 x0 x1: tran -> duoi cam (them cho cam bi keo)
MOUTH = [int(v * s) for v in (285, 400, 180, 332)]                      # vung mieng + cam (chon khung)

lim0 = A.limit_mouth
F = {}
for r, unlimited in enumerate((True, False)):
    A.limit_mouth = (lambda P, *a, **k: P) if unlimited else lim0
    try:
        for code, _ in COLS:
            fr, _ = g['render'](y, timeline=[(0.0, code)], name='lim', save=False, keep=True, label=False, verbose=False,
                                mouth_redraw=False, brow_redraw=False)
            F[(r, code)] = np.stack(fr)
    finally:
        A.limit_mouth = lim0
n = min(len(v) for v in F.values())
reg = lambda a: a[:n, MOUTH[0]:MOUTH[1], MOUTH[2]:MOUTH[3]].astype(np.float32)
diff = np.mean([np.abs(reg(F[(0, c)]) - reg(F[(1, c)])).mean((1, 2, 3)) for c, _ in COLS], 0)
diff[:10] = diff[-10:] = -1
i = int(np.argmax(diff))
print('fig_mouth_limits: khung %d (%.2f s) | lech TB mieng-cam %.2f | tung cot %s' % (
    i, i / g['OUT_FPS'], diff[i], [round(float(np.abs(reg(F[(0, c)])[i] - reg(F[(1, c)])[i]).mean()), 2) for c, _ in COLS]))

Wd = PAD + LABEL_W + len(COLS) * TILE + (len(COLS) - 1) * GAP + PAD
Hd = PAD + HEAD_H + 2 * TILE + GAP + PAD
img = Image.new('RGB', (Wd, Hd), 'white'); d = ImageDraw.Draw(img)
for c, (code, title) in enumerate(COLS):
    x = PAD + LABEL_W + c * (TILE + GAP)
    d.text((x + TILE // 2, PAD + HEAD_H // 2), title, fill='black', font=font(30), anchor='mm')
    for r in range(2):
        f = F[(r, code)][i][FACE[0]:FACE[1], FACE[2]:FACE[3], ::-1]      # BGR -> RGB
        img.paste(Image.fromarray(np.ascontiguousarray(f)).resize((TILE, TILE), Image.LANCZOS), (x, PAD + HEAD_H + r * (TILE + GAP)))
for r, name in enumerate(ROWS):
    t = Image.new('RGBA', (TILE, LABEL_W), (255, 255, 255, 0))
    ImageDraw.Draw(t).text((TILE // 2, LABEL_W // 2), name, fill='black', font=font(30), anchor='mm')
    t = t.rotate(90, expand=True)
    img.paste(t, (PAD, PAD + HEAD_H + r * (TILE + GAP)), t)
img.save(OUT_PNG, dpi=(300, 300))
print('->', OUT_PNG, img.size)
