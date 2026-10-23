"""Sinh boy.svg, girl2.svg tu khuon girl.svg (cung mat / mui / mieng / vien), xuat PNG 1024 + 11 diem click."""
import json
import shutil
from pathlib import Path

import resvg_py

SRC = Path(r'C:\hien\elsa_test_files\girl.svg')
OUT = Path(r'C:\hien\mos\characters')
OUT.mkdir(parents=True, exist_ok=True)
g = SRC.read_text(encoding='utf-8')

# 11 diem (anh 1024): 2 mat, mui, 2 khoe, giua moi tren / duoi, 4 dau long may (giong girl, da kiem tra o v1)
PTS = [[420, 470], [604, 470], [512, 565], [482, 622], [542, 622], [512, 624], [512, 628],
       [378, 405], [456, 398], [568, 398], [646, 405]]
NAMES = ['mat_trai', 'mat_phai', 'mui', 'khoe_trai', 'khoe_phai', 'giua_moi_tren', 'giua_moi_duoi',
         'may_trai_ngoai', 'may_trai_trong', 'may_phai_trong', 'may_phai_ngoai']

BACK_LONG = [l for l in g.splitlines() if 'C230 250 360 150 512 150' in l][0]
LOCK_L = [l for l in g.splitlines() if 'M322 360 C300 470' in l][0]
LOCK_R = [l for l in g.splitlines() if 'M702 360 C724 470' in l][0]
CLIP = [l for l in g.splitlines() if 'hair clip' in l or 'circle cx="650" cy="300"' in l or 'circle cx="676" cy="318"' in l]
V_COL = [l for l in g.splitlines() if 'M430 800 L512 880 L594 800' in l or 'M470 840 L512 880 L554 840' in l]
WINGS = ['<path d="M366 460 L352 448 M372 450 L360 436"', '<path d="M658 460 L672 448 M652 450 L664 436"']


def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:60], s.count(a))
    return s.replace(a, b)


def drop(s, lines):
    for l in lines:
        s = rep(s, l + '\n', '')
    return s


# ---------------------------------------------------------------- boy: toc ngan den xanh, mat nau, may day, ao co tron
b = g
b = rep(b, '<stop offset="0" stop-color="#5a3a2e"/><stop offset="1" stop-color="#3b241c"/>',
        '<stop offset="0" stop-color="#2c3d63"/><stop offset="1" stop-color="#111a2f"/>')
b = rep(b, BACK_LONG, '  <path d="M286 440 C262 250 380 150 512 150 C644 150 762 250 738 440 C736 500 722 540 702 560 '
                      'L322 560 C302 540 288 500 286 440 Z" fill="url(#hair)"/>')
b = rep(b, LOCK_L, '  <path d="M322 360 C306 420 306 480 322 540 C340 500 350 460 350 420 Z" fill="url(#hair)" stroke="#2b1d18" stroke-width="4"/>')
b = rep(b, LOCK_R, '  <path d="M702 360 C718 420 718 480 702 540 C684 500 674 460 674 420 Z" fill="url(#hair)" stroke="#2b1d18" stroke-width="4"/>')
b = rep(b, 'stroke="#8a6352"', 'stroke="#5d79ab"')
b = rep(b, '<stop offset="0" stop-color="#b98be0"/><stop offset="0.6" stop-color="#7a4fb3"/><stop offset="1" stop-color="#3f2468"/>',
        '<stop offset="0" stop-color="#d3a26a"/><stop offset="0.6" stop-color="#8f5a2a"/><stop offset="1" stop-color="#4a2a10"/>')
b = rep(b, 'fill="#24133d"', 'fill="#2a1608"', 2)
b = rep(b, 'stroke="#3b241c" stroke-width="7"', 'stroke="#141b2c" stroke-width="12"', 2)      # may day hon
for w in WINGS:                                                               # boy: bo 2 net mi hat (kieu mat nu)
    b = drop(b, [l for l in b.splitlines() if w in l])
b = rep(b, '<stop offset="0" stop-color="#f4f1ec"/><stop offset="1" stop-color="#d9d3ca"/>',
        '<stop offset="0" stop-color="#5f84bb"/><stop offset="1" stop-color="#3d5e92"/>')
b = drop(b, V_COL + CLIP)
b = rep(b, '  <!-- neck -->', '  <path d="M440 800 C468 836 556 836 584 800" fill="none" stroke="#2b1d18" stroke-width="6"/>\n  <!-- neck -->')
b = rep(b, 'opacity="0.45"', 'opacity="0.22"', 2)                                              # ma nhat hon
b = rep(b, '<path d="M496 636 C506 640 518 640 528 636" fill="none" stroke="#f0a7a0" stroke-width="5" stroke-linecap="round" opacity="0.7"/>\n', '')

# ---------------------------------------------------------------- girl2: toc hong buoc 2 ben, mat xanh la
c = g
c = rep(c, '<stop offset="0" stop-color="#5a3a2e"/><stop offset="1" stop-color="#3b241c"/>',
        '<stop offset="0" stop-color="#f6a9cb"/><stop offset="1" stop-color="#d4628d"/>')
TAILS = ('  <path d="M300 360 C210 380 170 520 196 660 C214 760 206 830 236 880 C292 820 318 720 316 600 C314 500 322 420 336 380 Z" '
         'fill="url(#hair)" stroke="#2b1d18" stroke-width="5"/>\n'
         '  <path d="M724 360 C814 380 854 520 828 660 C810 760 818 830 788 880 C732 820 706 720 708 600 C710 500 702 420 688 380 Z" '
         'fill="url(#hair)" stroke="#2b1d18" stroke-width="5"/>\n'
         '  <path d="M286 440 C262 250 380 150 512 150 C644 150 762 250 738 440 L286 440 Z" fill="url(#hair)"/>')
c = rep(c, BACK_LONG, TAILS)
c = rep(c, 'stroke="#8a6352"', 'stroke="#ffd6e8"')
c = rep(c, '<stop offset="0" stop-color="#b98be0"/><stop offset="0.6" stop-color="#7a4fb3"/><stop offset="1" stop-color="#3f2468"/>',
        '<stop offset="0" stop-color="#a6eab0"/><stop offset="0.6" stop-color="#3aa857"/><stop offset="1" stop-color="#17532a"/>')
c = rep(c, 'fill="#24133d"', 'fill="#0f3519"', 2)
c = rep(c, 'stroke="#3b241c" stroke-width="7"', 'stroke="#9c3f63" stroke-width="7"', 2)
c = rep(c, '<stop offset="0" stop-color="#f4f1ec"/><stop offset="1" stop-color="#d9d3ca"/>',
        '<stop offset="0" stop-color="#fff6dc"/><stop offset="1" stop-color="#efd9a6"/>')
c = rep(c, 'fill="#7a4fb3" stroke="#2b1d18" stroke-width="4"', 'fill="#3aa857" stroke="#2b1d18" stroke-width="4"')
c = drop(c, CLIP)
c = rep(c, '</svg>', '  <!-- hair ties -->\n'
        '  <circle cx="318" cy="372" r="18" fill="#7fd6f0" stroke="#2b1d18" stroke-width="4"/>\n'
        '  <circle cx="706" cy="372" r="18" fill="#7fd6f0" stroke="#2b1d18" stroke-width="4"/>\n</svg>')

for name, svg in (('anime_girl', g), ('boy', b), ('girl2', c)):
    (OUT / (name + '.svg')).write_text(svg, encoding='utf-8')
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=1024, height=1024))
    (OUT / (name + '.png')).write_bytes(png)
    (OUT / (name + '_points.json')).write_text(json.dumps(dict(image=name + '.png', size=[1024, 1024], order=NAMES, points=PTS), indent=1))
    print(name, len(png))
