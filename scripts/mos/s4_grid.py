import csv, os, shutil, subprocess, sys, cv2, numpy as np
from pathlib import Path
MOS = Path(r'C:\hien\mos'); FF = (os.environ.get('FFMPEG') or shutil.which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
A = MOS / '_key' / 'raw' / 's4_rar' / 'hien' / 'mos' / 'wav2lip_outputs'
key = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
PICK = ['W018_neutral_L1_031', 'M032_happy_L3_020', 'M032_angry_L3_010']      # 1 cau / cam xuc, 3 nhan vat khac nhau
lab = lambda t, s, y=16: (cv2.putText(t, s, (5, y), 0, 0.45, (0, 0, 0), 3, cv2.LINE_AA), cv2.putText(t, s, (5, y), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA), t)[-1]


def frames(p, idx):  # giai ma qua ffmpeg theo nhan mau cua file (cv2 bo qua nhan BT.709), BGR
    o = subprocess.run([FF, '-loglevel', 'error', '-i', str(p), '-vf', 'select=%s,scale=out_range=pc:flags=accurate_rnd+full_chroma_int,format=bgr24' % '+'.join('eq(n\\,%d)' % i for i in idx),
                        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, 512, 512, 3)


blocks = []
for clip in PICK:
    rows = []
    for sysn in ('S1', 'S3', 'S4'):
        r = next(x for x in key if x['clip'] == clip and x['system'] == sysn)
        n = int(cv2.VideoCapture(str(MOS / 'videos' / r['video'])).get(7)); idx = [int(n * f) for f in np.linspace(0.1, 0.9, 6)]
        F = frames(MOS / 'videos' / r['video'], idx)
        rows.append(np.concatenate([lab(cv2.resize(fr[150:390, 136:376], (180, 180), interpolation=cv2.INTER_AREA), ('%s  ' % sysn if j == 0 else '') + '%.1fs' % (i / 25))
                                    for j, (i, fr) in enumerate(zip(idx, F))], 1))
    # mep tren / duoi (8 hang, phong to x4 theo chieu doc): ban goc cua A (giai ma nhu nhan limited) | S4 da sua
    W = rows[0].shape[1]; e = []
    for name, p in (('A goc', A / ('%s__%s.mp4' % (clip, r['character']))), ('S4 sua', MOS / 'videos' / r['video'])):
        fr = frames(p, [idx[2]])[0] if name == 'S4 sua' else cv2.VideoCapture(str(p)).read()[1]
        s = np.concatenate([fr[:8], np.full((2, 512, 3), 255, np.uint8), fr[-8:]], 0)
        e.append(lab(cv2.resize(cv2.resize(s, (512, 18 * 4), interpolation=cv2.INTER_NEAREST), (W // 2, 18 * 4), interpolation=cv2.INTER_AREA), name + ': 8 hang tren | 8 hang duoi', 40))
    rows.append(np.concatenate(e, 1))
    head = np.full((26, W, 3), 40, np.uint8)
    cv2.putText(head, '%s | %s | %s | S4 = Wav2Lip (re-encoded: range tag + 2-px border)' % (clip, r['character'], r['emotion_true']), (6, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    blocks.append(np.concatenate([head] + rows, 0))
cv2.imwrite(str(MOS / '_check' / (sys.argv[1] if len(sys.argv) > 1 else 's4_check_grid.png')), np.concatenate(blocks, 0)); print('ok')
