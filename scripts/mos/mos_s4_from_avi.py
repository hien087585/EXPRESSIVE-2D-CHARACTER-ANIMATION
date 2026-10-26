"""Lam lai S4 (Wav2Lip, tro thu A) tu 12 file AVI MJPEG trung gian cua A (_key\\raw\\s4_avi\\), thay cho ban da sua
bang mos_merge_s4.py (ban do chuyen vao _key\\raw\\s4_fixed_old\\, khong dung).
  1. Kiem tra AVI: du 12, dung ten, khop mau PNG (ngoai box Wav2Lip), mep tren/duoi khong vien, mieng co chuyen dong.
  2. Ma hoa bang ffmpeg giong S1-S3 (scale 512 flags=area, 25 fps, libx264 veryfast crf 18 yuv420p, aac 128k);
     MJPEG la BT.601 full range -> doi sang BT.709 limited (cung buoc scale, them co accurate_rnd) va gan nhan BT.709. AVI khong co tieng -> ghep wav goc
     tu wav2lip_inputs\\. Giu nguyen moi khung (so khung = ceil(wav x 25)), khong keo gian, khong -shortest.
     Khong xu ly gi khac.
  3. _key\\raw\\S4__<cau>__<nhan vat>.mp4 -> videos\\<ma mu da cap san>.mp4, cap nhat key.csv.
  4. Bang do lech mau S1-S4 so voi PNG, kiem tra cuoi 50/50."""
import csv
import os
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path

MOS = Path(r'C:\hien\mos'); FF = (os.environ.get('FFMPEG') or shutil.which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
AVI = MOS / '_key' / 'raw' / 's4_avi'
OLD = MOS / '_key' / 'raw' / 's4_fixed_old'
S4_NOTE = "S4 = Wav2Lip, encoded from A's MJPEG intermediates with the same ffmpeg settings as S1\u2013S3; no other processing"
SWS = 'accurate_rnd+full_chroma_int'   # khong co accurate_rnd, doi 601 full -> 709 limited lam toi deu ~1,5 (|d| 1,9 thay vi ~1)
# ffmpeg 9 ghi nhan mau theo thuoc tinh khung (MJPEG: primaries/trc chua ro), nen dat ca tren khung lan tham so encoder
TAG = 'setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv'
BOX = (170, 364, 160, 352)   # --box cua A (top bottom left right)


def rgb(p, idx=None):  # giai ma qua ffmpeg theo nhan mau cua file (khong nhan = BT.601 limited), RGB float
    vf = 'scale=out_range=pc:flags=%s,format=rgb24' % SWS
    if idx is not None:
        vf = 'select=%s,' % '+'.join('eq(n\\,%d)' % i for i in idx) + vf
    o = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-vf', vf, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'],
                       capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, 512, 512, 3).astype(float)


def probe(p):
    e = subprocess.run([FF, '-i', str(p)], capture_output=True, text=True).stderr
    t = [l for l in e.splitlines() if 'Duration' in l][0].split('Duration:')[1].split(',')[0].strip().split(':')
    v = [l for l in e.splitlines() if 'Video:' in l][0]
    return int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2]), 'Audio:' in e, '512x512' in v, ' 25 fps' in v


rows = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
s4 = [r for r in rows if r['system'] == 'S4']
name = lambda r: '%s__%s' % (r['clip'], r['character'])
m = np.ones((512, 512), bool); m[BOX[0]:BOX[1], BOX[2]:BOX[3]] = False; m[:4] = m[-4:] = False

# 1. kiem tra AVI
want, have = {name(r) + '.avi' for r in s4}, {p.name for p in AVI.glob('*.avi')}
print('AVI: %d file, thieu %s, thua %s' % (len(have), sorted(want - have) or '-', sorted(have - want) or '-'))
assert want == have
print('%-34s %5s %6s %6s %7s %8s %8s %6s' % ('avi', 'khung', 'wav*25', '|d|', 'do doc', 'mep 2px', 'mieng', 'r(am)'))
for r in s4:
    p, wav = AVI / (name(r) + '.avi'), MOS / 'wav2lip_inputs' / (name(r) + '.wav')
    png = cv2.imread(str(wav.with_suffix('.png')), 1)[..., ::-1].astype(float)
    F = rgb(p)
    a = np.frombuffer(subprocess.run([FF, '-loglevel', 'error', '-i', str(wav), '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
                                     capture_output=True, check=True).stdout, np.int16).astype(float)
    n_wav = int(np.ceil(len(a) / 16000 * 25 - 1e-6))
    d = np.abs(F[:, m] - png[m]).mean()
    slope = np.polyfit(np.tile(png[m].ravel(), len(F)), F[:, m].ravel(), 1)[0]
    edge = np.abs(F[:, [0, 1, 510, 511]] - png[[0, 1, 510, 511]]).mean()
    mouth = F[:, (BOX[0] + BOX[1]) // 2:BOX[1], BOX[2]:BOX[3]].mean(3)                   # nua duoi box (mieng)
    mv = np.abs(np.diff(mouth, axis=0)).mean((1, 2))                                     # chuyen dong giua 2 khung
    rms = np.array([np.sqrt((a[i * 640:(i + 1) * 640] ** 2).mean()) if i * 640 < len(a) else 0 for i in range(len(F))])
    corr = np.corrcoef(np.abs(mouth - mouth.mean(0)).mean((1, 2)), rms)[0, 1]           # do mo mieng ~ nang luong am
    print('%-34s %5d %6d %6.2f %7.3f %8.2f %8.2f %6.2f' % (p.name, len(F), n_wav, d, slope, edge, mv.mean(), corr))
if os.environ.get('CHECK_ONLY'):
    raise SystemExit

# 2-3. ban da sua cu -> s4_fixed_old; ma hoa tu AVI; chep vao ma mu
OLD.mkdir(exist_ok=True)
for r in s4:
    dst = MOS / '_key' / 'raw' / ('S4__%s.mp4' % name(r))
    if dst.exists() and not (OLD / dst.name).exists():
        shutil.move(str(dst), str(OLD / dst.name))
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(AVI / (name(r) + '.avi')), '-i', str(MOS / 'wav2lip_inputs' / (name(r) + '.wav')),
                    '-map', '0:v', '-map', '1:a',
                    '-vf', 'scale=512:512:flags=area+%s:in_range=pc:in_color_matrix=bt601:out_range=tv:out_color_matrix=bt709,%s' % (SWS, TAG),
                    '-r', '25', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
                    '-c:a', 'aac', '-b:a', '128k', str(dst)], check=True)
    shutil.copy2(dst, MOS / 'videos' / r['video'])
    r['status'], r['raw_file'], r['note'] = 'ready', dst.name, S4_NOTE
    print('S4 %s <- %s.avi' % (r['video'], name(r)))
with open(MOS / '_key' / 'key.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# 4. do lech mau so voi PNG: ngoai box Wav2Lip va bo 4 hang mep; 5 khung / video
col = {}
for r in rows:
    if r['system'] not in ('S1', 'S2', 'S3', 'S4'):
        continue
    png = cv2.imread(str(MOS / 'wav2lip_inputs' / (name(r) + '.png')), 1)[..., ::-1].astype(float)
    srcs = [('S4_avi', AVI / (name(r) + '.avi')), ('S4_sua_cu', OLD / ('S4__%s.mp4' % name(r)))] if r['system'] == 'S4' else []
    for k, p in srcs + [(r['system'], MOS / 'videos' / r['video'])]:
        F = rgb(p, [10, 30, 50, 70, 90])
        d = (F - png)[:, m]
        a, b = np.polyfit(np.tile(png[m].ravel(), len(F)), F[:, m].ravel(), 1)
        edge = np.abs(F[:, [0, 1, 510, 511]] - png[[0, 1, 510, 511]]).mean()
        col.setdefault(k, []).append((np.abs(d).mean(), *d.mean((0, 1)), a, edge))
print('\ndo lech mau so voi PNG (ngoai box, trung binh 12 video x 5 khung):')
print('%-10s %7s %21s %7s %9s' % ('he', '|d|', 'd R/G/B', 'do doc', 'mep 2px'))
for k in ('S1', 'S2', 'S3', 'S4', 'S4_avi', 'S4_sua_cu'):
    v = np.array(col[k])
    print('%-10s %7.2f   %+5.2f %+5.2f %+5.2f %7.3f %9.1f   (|d| min-max %.2f-%.2f)' % (k, v[:, 0].mean(), *v[:, 1:4].mean(0), v[:, 4].mean(), v[:, 5].mean(), v[:, 0].min(), v[:, 0].max()))
p = MOS / 'videos' / s4[0]['video']
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
