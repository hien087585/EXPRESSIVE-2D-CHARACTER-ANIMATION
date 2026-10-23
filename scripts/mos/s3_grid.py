import csv, sys, cv2, numpy as np
from pathlib import Path
MOS = Path(r'C:\hien\mos')
key = list(csv.DictReader(open(MOS / '_key' / 'key.csv', encoding='utf-8')))
PICK = ['W018_neutral_L1_031', 'M032_happy_L3_020', 'M032_angry_L3_010']      # 1 cau / cam xuc, 3 nhan vat khac nhau
lab = lambda t, s, y=16: (cv2.putText(t, s, (5, y), 0, 0.45, (0, 0, 0), 3, cv2.LINE_AA), cv2.putText(t, s, (5, y), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA), t)[-1]
blocks = []
for clip in PICK:
    rows = []
    for sysn in ('S1', 'S2', 'S3'):
        r = next(x for x in key if x['clip'] == clip and x['system'] == sysn)
        cap = cv2.VideoCapture(str(MOS / 'videos' / r['video'])); n = int(cap.get(7)); tiles = []
        for j, f in enumerate(np.linspace(0.1, 0.9, 6)):
            i = int(n * f); cap.set(1, i); ok, fr = cap.read()
            t = cv2.resize(fr[150:390, 136:376], (180, 180), interpolation=cv2.INTER_AREA)
            tiles.append(lab(t, ('%s  ' % sysn if j == 0 else '') + '%.1fs' % (i / 25)))
        rows.append(np.concatenate(tiles, 1))
    head = np.full((26, rows[0].shape[1], 3), 40, np.uint8)
    cv2.putText(head, '%s | %s | that: %s | SpEx-KD: %s' % (clip, r['character'], r['emotion_true'],
                next(x for x in key if x['clip'] == clip and x['system'] == 'S1')['emotion_spexkd']), (6, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    blocks.append(np.concatenate([head] + rows, 0))
cv2.imwrite(str(MOS / '_check' / (sys.argv[1] if len(sys.argv) > 1 else 's3_vs_s1_grid_v2.png')), np.concatenate(blocks, 0)); print('ok')
