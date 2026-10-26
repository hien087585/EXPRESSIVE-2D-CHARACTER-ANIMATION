"""Hinh so sanh dinh tinh (Fig. 11): 3 hang = 3 nhan vat (1 cau happy/angry moi nhan vat),
4 cot = Input image | MakeItTalk (S3) | Wav2Lip (S4) | ELSA (S1). Moi hang cung 1 khung (cung thoi diem audio),
chon khung mieng S1 mo ro nhat (bo khung dang chop mat va 10 khung dau/cuoi).
Cat vung mat, nen trang, 300 dpi, rong ~1800 px -> _check\\fig_compare_systems.png.
Them _check\\wav2lip_mouth_zoom.png: phong to vung mieng S4 | S1 cua 1 khung (do mo cua Wav2Lip); khung zoom chon
rieng (cau hang dau) sao cho mieng ca S4 lan S1 deu mo, de so do mo cong bang.
Video giai ma theo nhan mau cua file (S1/S3 khong nhan = BT.601 limited, S4 = BT.709 limited)."""
import csv
import os
import shutil
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

MOS = Path(r'C:\hien\mos'); FF = (os.environ.get('FFMPEG') or shutil.which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
ROWS = [('anime_girl', 'M032_angry_L3_010'), ('boy', 'M032_happy_L3_020'), ('girl2', 'M037_happy_L3_007')]
COLS = [('Input image', None), ('MakeItTalk', 'S3'), ('Wav2Lip', 'S4'), ('ELSA (ours)', 'S1')]
CROP = (140, 380, 136, 376)          # y0 y1 x0 x1 tren khung 512: tran -> cam
MOUTH = (285, 345, 200, 312)         # vung mieng (do mo mieng)
EYES = (205, 255, 170, 345)          # vung mat (bo khung chop mat)
TILE, GAP, HEAD_H, PAD = 430, 12, 70, 20
SWS = 'accurate_rnd+full_chroma_int'
font = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', s)


def frames(p):
    o = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-vf', 'scale=out_range=pc:flags=%s,format=rgb24' % SWS, '-f', 'rawvideo', '-'],
                       capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, 512, 512, 3)


def scores(F, png):  # do mo mieng (lech so voi anh dau vao) va khung hop le (khong chop mat, bo 10 khung dau/cuoi)
    reg = lambda b: F[:, b[0]:b[1], b[2]:b[3]].astype(float) - png[b[0]:b[1], b[2]:b[3]]
    mouth, eyes = np.abs(reg(MOUTH)).mean((1, 2, 3)), np.abs(reg(EYES)).mean((1, 2, 3))
    ok = eyes <= np.median(eyes) * 1.5
    ok[:10] = ok[-10:] = False
    return mouth, ok


def pick(F, png):
    mouth, ok = scores(F, png)
    return int(np.argmax(np.where(ok, mouth, -1)))


key = {(r['system'], r['clip'], r['character']): r['video'] for r in csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8'))}

W = PAD + 4 * TILE + 3 * GAP + PAD
H = PAD + HEAD_H + 3 * TILE + 2 * GAP + PAD
img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
for c, (title, _) in enumerate(COLS):
    d.text((PAD + c * (TILE + GAP) + TILE // 2, PAD + HEAD_H // 2), title, fill='black', font=font(40), anchor='mm')
zoom = None
for r, (char, clip) in enumerate(ROWS):
    png = np.asarray(Image.open(MOS / 'wav2lip_inputs' / ('%s__%s.png' % (clip, char))).convert('RGB'))
    F = {s: frames(MOS / 'videos' / key[(s, clip, char)]) for _, s in COLS if s}
    i = pick(F['S1'], png)
    print('%s %s: khung %d (%.2f s)' % (char, clip, i, i / 25))
    y = PAD + HEAD_H + r * (TILE + GAP)
    for c, (_, s) in enumerate(COLS):
        f = png if s is None else F[s][i]
        img.paste(Image.fromarray(np.ascontiguousarray(f[CROP[0]:CROP[1], CROP[2]:CROP[3]])).resize((TILE, TILE), Image.LANCZOS), (PAD + c * (TILE + GAP), y))
    if r == 0:
        zoom = (char, clip, png, F)
img.save(MOS / '_check' / 'fig_compare_systems.png', dpi=(300, 300))
print('fig_compare_systems.png', img.size)

# phong to vung mieng: S4 | S1, cung khung, phong to x6 (nearest de thay ro do mo / diem anh)
char, clip, png, F = zoom
(m1, ok), (m4, _) = scores(F['S1'], png), scores(F['S4'], png)
n = min(len(m1), len(m4)); both = np.minimum(m1[:n] / m1[:n].max(), m4[:n] / m4[:n].max())
i = int(np.argmax(np.where(ok[:n], both, -1))); s4, s1 = F['S4'][i], F['S1'][i]
Y0, Y1, X0, X1 = 270, 350, 196, 316                      # 80 x 120 px quanh mieng
Z, ZH, ZW = 6, 80 * 6, 120 * 6
W2 = PAD + 2 * ZW + GAP + PAD
H2 = PAD + HEAD_H + ZH + PAD
z = Image.new('RGB', (W2, H2), 'white'); dz = ImageDraw.Draw(z)
for c, (title, f) in enumerate([('Wav2Lip', s4), ('ELSA (ours)', s1)]):
    dz.text((PAD + c * (ZW + GAP) + ZW // 2, PAD + HEAD_H // 2), title, fill='black', font=font(44), anchor='mm')
    z.paste(Image.fromarray(np.ascontiguousarray(f[Y0:Y1, X0:X1])).resize((ZW, ZH), Image.NEAREST), (PAD + c * (ZW + GAP), PAD + HEAD_H))
z.save(MOS / '_check' / 'wav2lip_mouth_zoom.png', dpi=(300, 300))
print('wav2lip_mouth_zoom.png', z.size, '| %s %s khung %d' % (char, clip, i))
