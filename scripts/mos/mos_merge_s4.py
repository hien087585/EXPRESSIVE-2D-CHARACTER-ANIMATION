"""CU - KHONG CHAY LAI: da thay bang mos_s4_from_avi.py (ma hoa tu AVI MJPEG cua A); ket qua cua script nay nam o
_key\\raw\\s4_fixed_old\\. Chay lai se ghi de videos\\ va key.csv.

Gop S4 (Wav2Lip, do tro thu A lam) vao bo mu.
Ban goc cua A giu nguyen trong _key\\raw\\s4_rar\\hien\\mos\\wav2lip_outputs\\. Chi sua 2 loi ma hoa cua A:
  (1) luong H.264 cua A khong co nhan mau (mac dinh = limited) nhung du lieu la full range, he so BT.601 (do: giai ma
      601/full khop PNG ~1,0, cac cach khac 2,2-12,8) -> giai ma dung cach roi ma hoa lai limited, he so + nhan BT.709;
  (2) vach xanh 2 px o mep tren / duoi: do tren YUV, luma moi hang mep dung (|dY| <= 0,7 so voi PNG), chi hang chroma
      dau va cuoi (4:2:0 -> phu hang 0-1 va 510-511) sai ~15-19 -> thay 2 hang chroma do bang hang chroma ke ben.
      Khong thay ca hang RGB vi PNG co duong toi that o 2 hang duoi cung (S1-S3 van giu).
Doi ma tran/range lam truc tiep tren YUV (filter colorspace, khong lay mau lai chroma, giu primaries -> khong doi gamut).
Khong lam net, khong chinh mau hay noi dung gi khac; am thanh chep nguyen (-c:a copy).
Ket qua: _key\\raw\\S4__<cau>__<nhan vat>.mp4 -> videos\\<ma da cap san>.mp4, cap nhat key.csv, kiem tra cuoi,
do do lech mau so voi PNG (S1-S4)."""
import csv
import os
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path

MOS = Path(r'C:\hien\mos'); FF = (os.environ.get('FFMPEG') or shutil.which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
A = MOS / '_key' / 'raw' / 's4_rar' / 'hien' / 'mos' / 'wav2lip_outputs'
S4_NOTE = 'S4 = Wav2Lip (assistant A); S4 re-encoded: colour-range tag corrected, 2-px encoder border replaced; no other processing'
SWS = 'accurate_rnd+full_chroma_int'


def fix(src, dst):
    dec = subprocess.run([FF, '-loglevel', 'error', '-i', str(src), '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, check=True).stdout
    v = np.frombuffer(dec, np.uint8).reshape(-1, 512 * 512 * 3 // 2).copy()
    for c in (v[:, 512 * 512:512 * 512 * 5 // 4], v[:, 512 * 512 * 5 // 4:]):   # U, V (256x256)
        c = c.reshape(-1, 256, 256); c[:, 0] = c[:, 1]; c[:, 255] = c[:, 254]
    subprocess.run([FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', '512x512', '-r', '25', '-i', '-', '-i', str(src),
                    '-map', '0:v', '-map', '1:a', '-c:a', 'copy',
                    '-vf', 'colorspace=ispace=bt470bg:irange=pc:iprimaries=bt709:itrc=bt709:space=bt709:range=tv:primaries=bt709:trc=bt709:format=yuv420p:dither=none:fast=0',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv', str(dst)],
                   input=v.tobytes(), check=True)
    return len(v)


rows = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
for r in rows:
    if r['system'] == 'S4':
        src = A / ('%s__%s.mp4' % (r['clip'], r['character']))
        dst = MOS / '_key' / 'raw' / ('S4__%s__%s.mp4' % (r['clip'], r['character']))
        n = fix(src, dst)
        shutil.copy2(dst, MOS / 'videos' / r['video'])
        r['status'], r['raw_file'], r['note'] = 'ready', dst.name, S4_NOTE
        print('S4 %s <- %s (%d khung)' % (r['video'], src.name, n))
with open(MOS / '_key' / 'key.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def rgb(p, idx):  # giai ma qua ffmpeg (theo nhan mau cua file; khong nhan = BT.601 limited), RGB
    sel = '+'.join('eq(n\\,%d)' % i for i in idx)
    o = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-vf', 'select=%s,scale=out_range=pc:flags=%s,format=rgb24' % (sel, SWS),
                        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, 512, 512, 3).astype(float)


def probe(p):
    e = subprocess.run([FF, '-i', str(p)], capture_output=True, text=True).stderr
    t = [l for l in e.splitlines() if 'Duration' in l][0].split('Duration:')[1].split(',')[0].strip().split(':')
    v = [l for l in e.splitlines() if 'Video:' in l][0]
    return int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2]), 'Audio:' in e, '512x512' in v, ' 25 fps' in v


# do lech mau so voi PNG: ngoai box Wav2Lip (170:364, 160:352) va bo 4 hang mep; 5 khung / video
m = np.ones((512, 512), bool); m[170:364, 160:352] = False; m[:4] = m[-4:] = False
col = {}
for r in rows:
    if r['system'] not in ('S1', 'S2', 'S3', 'S4'):
        continue
    png = cv2.imread(str(MOS / 'wav2lip_inputs' / ('%s__%s.png' % (r['clip'], r['character']))), 1)[..., ::-1].astype(float)
    srcs = [('S4_A_goc', A / ('%s__%s.mp4' % (r['clip'], r['character'])))] if r['system'] == 'S4' else []
    for name, p in srcs + [(r['system'], MOS / 'videos' / r['video'])]:
        F = rgb(p, [10, 30, 50, 70, 90])
        d = (F - png)[:, m]
        a, b = np.polyfit(np.tile(png[m].ravel(), len(F)), F[:, m].ravel(), 1)
        edge = np.abs(F[:, [0, 1, 510, 511]] - png[[0, 1, 510, 511]]).mean()
        col.setdefault(name, []).append((np.abs(d).mean(), *d.mean((0, 1)), a, edge))
print('\ndo lech mau so voi PNG (ngoai box, trung binh 12 video x 5 khung):')
print('%-9s %7s %21s %7s %9s' % ('he', '|d|', 'd R/G/B', 'do doc', 'mep 2px'))
for k in ('S1', 'S2', 'S3', 'S4_A_goc', 'S4'):
    v = np.array(col[k])
    print('%-9s %7.2f   %+5.2f %+5.2f %+5.2f %7.3f %9.1f   (|d| min-max %.2f-%.2f)' % (k, v[:, 0].mean(), *v[:, 1:4].mean(0), v[:, 4].mean(), v[:, 5].mean(), v[:, 0].min(), v[:, 0].max()))
for p in (MOS / 'videos' / next(r['video'] for r in rows if r['system'] == 'S4'),):
    e = subprocess.run([FF, '-hide_banner', '-i', str(p), '-c', 'copy', '-bsf:v', 'trace_headers', '-frames:v', '1', '-f', 'null', '-'], capture_output=True, text=True).stderr
    print('nhan mau S4 (%s):' % p.name, ', '.join(l.split()[-3] + '=' + l.split()[-1] for l in e.splitlines()
                                              if any(k in l for k in ('video_full_range_flag', 'colour_primaries', 'transfer_characteristics', 'matrix_coefficients'))))

# kiem tra cuoi
dur_a = {r['file'][:-4]: float(r['duration_s']) for r in csv.DictReader(open(MOS / 'audio_list.csv', encoding='utf-8'))}
bad, n_ok = [], 0
for r in rows:
    d, au, sz, fps = probe(MOS / 'videos' / r['video'])
    cap = cv2.VideoCapture(str(MOS / 'videos' / r['video'])); white = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        white = max(white, int((fr[-30:, :200].min(2) > 250).sum()))
    good = r['status'] == 'ready' and au and sz and fps and abs(d - dur_a[r['clip']]) <= 0.05 and white == 0
    n_ok += good
    if not good:
        bad.append((r['video'], r['system'], r['status'], d, au, sz, fps, white))
order = list(csv.DictReader(open(MOS / 'play_order.csv', encoding='utf-8')))
clip_of = {r['video']: r['clip'] for r in rows}
adj = sum(clip_of[order[i]['video']] == clip_of[order[i + 1]['video']] for i in range(len(order) - 1))
vids = {p.name for p in (MOS / 'videos').glob('*.mp4')}
print('\nkiem tra cuoi: dat %d/%d | loi: %s' % (n_ok, len(rows), bad or 'khong'))
print('play_order: %d video, trung key: %s, 2 video cung cau lien nhau: %d | thu muc videos: %d mp4, thua/thieu: %s / %s, file khac: %s' % (
    len(order), sorted(o['video'] for o in order) == sorted(clip_of), adj, len(vids), sorted(vids - set(clip_of)) or '-',
    sorted(set(clip_of) - vids) or '-', [p.name for p in (MOS / 'videos').iterdir() if p.suffix != '.mp4'] or '-'))
