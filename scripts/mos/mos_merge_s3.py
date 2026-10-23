"""Gop S3 vao bo mu: chep _key\\raw\\S3__*.mp4 vao videos\\<ma da cap san>.mp4, cap nhat key.csv (status, ghi chu),
kiem tra cuoi. Ma mu giu nguyen; ban S3 cu (chi warp) nam trong _key\\raw\\s3_warp_old\\ (khong dung)."""
import csv
import shutil
import subprocess
from pathlib import Path

MOS = Path(r'C:\hien\mos'); FF = (__import__('os').environ.get('FFMPEG') or __import__('shutil').which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
S3_NOTE = 'S3 = MakeItTalk landmarks + same redraw renderer, neutral'
rows = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
for r in rows:
    if r['system'] == 'S3':
        src = MOS / '_key' / 'raw' / ('S3__%s__%s.mp4' % (r['clip'], r['character']))
        if src.exists():
            shutil.copy2(src, MOS / 'videos' / r['video'])
            r['status'], r['raw_file'], r['note'] = 'ready', src.name, S3_NOTE
with open(MOS / '_key' / 'key.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def probe(p):
    e = subprocess.run([FF, '-i', str(p)], capture_output=True, text=True).stderr
    t = [l for l in e.splitlines() if 'Duration' in l][0].split('Duration:')[1].split(',')[0].strip().split(':')
    v = [l for l in e.splitlines() if 'Video:' in l][0]
    return int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2]), 'Audio:' in e, '512x512' in v, ' 25 fps' in v


import cv2
dur_a = {r['file'][:-4]: float(r['duration_s']) for r in csv.DictReader(open(MOS / 'audio_list.csv', encoding='utf-8'))}
tab, bad, n_ok = {}, [], 0
for r in rows:
    if r['status'] != 'ready':
        continue
    d, au, sz, fps = probe(MOS / 'videos' / r['video'])
    cap = cv2.VideoCapture(str(MOS / 'videos' / r['video'])); white = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        white = max(white, int((fr[-30:, :200].min(2) > 250).sum()))
    good = au and sz and fps and abs(d - dur_a[r['clip']]) <= 0.05 and white == 0
    n_ok += good
    if not good:
        bad.append((r['video'], r['system'], d, au, sz, fps, white))
    if r['system'] in ('S1', 'S2', 'S3'):
        tab.setdefault((r['clip'], r['character']), {})[r['system']] = d
print('%-22s %-10s %6s | %6s %6s %6s' % ('cau', 'nhan vat', 'audio', 'S1', 'S2', 'S3'))
for (c, ch), v in sorted(tab.items()):
    print('%-22s %-10s %6.2f | %6.2f %6.2f %6s' % (c, ch, dur_a[c], v.get('S1', 0), v.get('S2', 0), '%.2f' % v['S3'] if 'S3' in v else '-'))
ready = sum(r['status'] == 'ready' for r in rows)
print('\nvideo san sang: %d (dat kiem tra: %d) | cho S4 Wav2Lip: %d | tong ke hoach: %d | loi: %s' % (
    ready, n_ok, len(rows) - ready, len(rows), bad or 'khong'))
