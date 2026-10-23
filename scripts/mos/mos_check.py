import csv, subprocess, numpy as np, cv2
from pathlib import Path
from scipy.io import wavfile
MOS = Path(r'C:\hien\mos')
FF = (__import__('os').environ.get('FFMPEG') or __import__('shutil').which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
key = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
order = list(csv.DictReader(open(MOS / 'play_order.csv', encoding='utf-8')))
clip_of = {r['video']: r['clip'] for r in key}
print('play_order: %d video | 2 video lien nhau cung cau: %d' % (len(order), sum(clip_of[order[i]['video']] == clip_of[order[i + 1]['video']] for i in range(len(order) - 1))))
print('thu muc videos co file khoa?', any(p.suffix == '.csv' for p in (MOS / 'videos').iterdir()), '| so file video:', len(list((MOS / 'videos').glob('*.mp4'))))
def audio_of(p):
    r = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-f', 's16le', '-ac', '1', '-ar', '16000', '-'], capture_output=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32)
def env(y):  # bao bien do am luong 10 ms
    n = len(y) // 160; return np.sqrt((y[:n * 160].reshape(n, 160) ** 2).mean(1))
txt = 0
for r in key:
    if r['status'] != 'ready':
        continue
    p = MOS / 'videos' / r['video']
    cap = cv2.VideoCapture(str(p)); n = 0; white = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        white = max(white, int((f[-30:, :200].min(2) > 250).sum())); n += 1
    txt += white > 0
    if r['system'] == 'ATTENTION':
        a, b = env(audio_of(p)), env(audio_of(MOS / 'audio' / (r['clip'] + '.wav')))
        m = min(len(a), len(b)) - 150
        lag = int(np.argmax([np.dot(a[k:k + m], b[:m]) for k in range(0, 150)]))
        print('kiem tra chu y %s (%s, %s): am thanh tre %.2f s so voi goc, %d khung' % (r['video'], r['clip'], r['character'], lag / 100, n))
print('video co chu trang o goc trai duoi:', txt)
