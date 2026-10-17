import sys, io, json, zipfile, collections
import numpy as np

manifest_path, zip_path = sys.argv[1], sys.argv[2]

# ---------------- 1) manifest ----------------
m = json.load(open(manifest_path, encoding='utf-8'))
clips = list(m['clips'].values())
print('actor %s | status %s | feature %s | expected %s | co trong manifest %d'
      % (m.get('actor_id'), m.get('status'), m.get('feature_version'), m.get('expected_clips'), len(clips)))
print('trang thai clip   :', dict(collections.Counter(c['status'] for c in clips)))
print('goc quay          :', dict(collections.Counter(c['angle'] for c in clips)))
print('fps nguon         :', sorted({round(c['src_fps'], 2) for c in clips}))
cut = [c for c in clips if c['n_src_frames'] > c['n_frames']]
print('clip bi cat       : %d / %d | n_frames max %d' % (len(cut), len(clips), max(c['n_frames'] for c in clips)))
bad = [c for c in clips if c['n_valid'] != c['n_frames'] or c['n_detected'] != c['n_frames']]
print('clip co frame loi : %d' % len(bad), '| synced=False: %d' % sum(not c.get('synced', True) for c in clips))
tab = collections.defaultdict(lambda: collections.Counter())
for c in clips:
    tab[(c['emotion_id'], c['emotion'])][c['level']] += 1
print('\nemotion_id emotion      L1   L2   L3   (tat ca goc quay)')
for (eid, name), cnt in sorted(tab.items()):
    print('%-10d %-11s %4d %4d %4d' % (eid, name, cnt[1], cnt[2], cnt[3]))
print('so cau (sentence) khac nhau:', len({c['sentence'] for c in clips}))

# ---------------- 2) zip ----------------
print('\n=== ZIP ===')
with zipfile.ZipFile(zip_path) as z:
    names = z.namelist()
    ext = collections.Counter(n.rsplit('.', 1)[-1].lower() if '.' in n else '(none)' for n in names)
    print('so file:', len(names), '| phan mo rong:', dict(ext))
    print('co audio :', any(n.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.mp4', '.aac')) for n in names),
          '| co mel :', any('mel' in n.lower() for n in names))
    print('--- 12 ten dau ---')
    for n in names[:12]:
        print('  %-70s %10d' % (n[:70], z.getinfo(n).file_size))
    arrs = [n for n in names if n.lower().endswith(('.npy', '.npz'))]
    for n in arrs[:2]:
        d = np.load(io.BytesIO(z.read(n)), allow_pickle=False)
        items = {k: d[k] for k in d.files} if hasattr(d, 'files') else {'(array)': d}
        for k, a in items.items():
            print('\n%s [%s] shape %s dtype %s' % (n, k, a.shape, a.dtype))
            if a.ndim >= 2 and a.size:
                print('   min %.4f max %.4f | NaN: %d' % (np.nanmin(a), np.nanmax(a), int(np.isnan(a).sum())))
            P = None
            if a.ndim == 3 and a.shape[1] == 68:
                P = a
            elif a.ndim == 2 and a.shape[1] in (136, 204):
                P = a.reshape(a.shape[0], 68, a.shape[1] // 68)
            if P is not None:
                groups = {'jaw': range(0, 17), 'brows': range(17, 27), 'nose': range(27, 36),
                          'eyes': range(36, 48), 'mouth': range(48, 68)}
                print('   so toa do/diem:', P.shape[2], '| std theo thoi gian (x,y):',
                      {g: round(float(P[:, list(i), :2].std(axis=0).mean()), 4) for g, i in groups.items()})
