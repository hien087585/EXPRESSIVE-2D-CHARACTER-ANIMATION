"""Hinh cho bai bao: anime_girl, 3 cot (neutral / happy / angry) x 2 hang (Warp only = S2 | Redrawing = S1).
Hai hang cung cau, cung khung hinh (cung thoi diem audio); khung chon de mieng mo ro o ca hai he.
Cat vung mat, nen trang, 300 dpi, rong ~1800 px -> _check\\fig_redraw_vs_warp.png."""
import os
import shutil
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

MOS = Path(r'C:\hien\mos'); FF = (os.environ.get('FFMPEG') or shutil.which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
COLS = [('Neutral', 'W019_neutral_L1_005', 28), ('Happy', 'W018_happy_L3_016', 165), ('Angry', 'M032_angry_L3_010', 149)]
ROWS = [('Warp only', 'S2'), ('Redrawing', 'S1')]
CROP = (165, 405, 136, 376)          # y0 y1 x0 x1 tren khung 512: tu tran den cam
TILE, GAP, LABEL_W, HEAD_H, PAD = 520, 16, 190, 80, 24


def frame(p, i):  # giai ma qua ffmpeg (S1/S2 khong nhan mau = BT.601 limited), RGB
    o = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-vf', 'select=eq(n\\,%d),scale=out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24' % i,
                        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(512, 512, 3)


font = lambda s: ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', s)
W = PAD + LABEL_W + 3 * TILE + 2 * GAP + PAD
H = PAD + HEAD_H + 2 * TILE + GAP + PAD
img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
for c, (title, clip, i) in enumerate(COLS):
    x = PAD + LABEL_W + c * (TILE + GAP)
    d.text((x + TILE // 2, PAD + HEAD_H // 2), title, fill='black', font=font(46), anchor='mm')
    for r, (name, sysn) in enumerate(ROWS):
        y = PAD + HEAD_H + r * (TILE + GAP)
        f = frame(MOS / '_key' / 'raw' / ('%s__%s__anime_girl.mp4' % (sysn, clip)), i)[CROP[0]:CROP[1], CROP[2]:CROP[3]]
        img.paste(Image.fromarray(f).resize((TILE, TILE), Image.LANCZOS), (x, y))
for r, (name, _) in enumerate(ROWS):
    y = PAD + HEAD_H + r * (TILE + GAP) + TILE // 2
    t = Image.new('RGBA', (TILE, LABEL_W), (255, 255, 255, 0))
    ImageDraw.Draw(t).text((TILE // 2, LABEL_W // 2), name, fill='black', font=font(46), anchor='mm')
    t = t.rotate(90, expand=True)
    img.paste(t, (PAD, y - TILE // 2), t)
img.save(MOS / '_check' / 'fig_redraw_vs_warp.png', dpi=(300, 300))
print(img.size)
