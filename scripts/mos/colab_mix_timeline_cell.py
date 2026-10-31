# Cell them vao CUOI notebook 3_phaseC_demo_render.ipynb tren Colab (GPU), chay sau Cell 0 -> Cell 6 (QUICK_UI = True la du).
# Muc dich: luu du lieu de ve lai Fig. 9 (timeline cam xuc tren file ghep M037_MIX1) - Cell 12 chi luu hinh, khong luu so lieu.
#  - Dap an: phat lai DUNG bo sinh ngau nhien cua Cell 12 (random.Random(0), cung thu tu speaker / file / cau) toi file
#    M037_MIX1, roi KIEM TRA audio dung lai trung file .wav da luu trong phaseC_demo/test_audio/.
#  - Du doan: analyze_emotion tren CHINH file .wav da luu, 3 mo hinh nhu Cell 12: 'phaseG+e2v' (ELSA, ensemble),
#    'phaseG' (SpEx-KD), 'emotion2vec' (teacher). Khong train.
# Ghi 1 file MOI (khong ghi de): phaseC_demo/test_audio/M037_MIX1_timeline.npz
import random, tarfile
from scipy.io import wavfile

TARGET = 'M037_MIX1'
MIX_SPEAKERS, MIX_PER_SPK, SEG_PER_MIX, CLIPS_PER_SEG, MIX_LEVELS, GAP_SEC = split['test'], 5, 7, (2, 3), (2, 3), 0.3   # = Cell 12
CLS = ['neutral'] + list(KEEP_NAMES)
TEST_DIR = (DRIVE / 'student_a2l' / 'phaseC_demo' / 'test_audio')
OUT_NPZ = TEST_DIR / (TARGET + '_timeline.npz')
assert not OUT_NPZ.exists(), 'Da co %s - doi ten neu muon luu lai' % OUT_NPZ


def load_spk_clips(sp):                       # = Cell 12
    tmp = Path('/tmp/mixsrc') / sp; tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    with tarfile.open(MEAD_SRC / sp / 'audio.tar') as tf:
        mem = [m for m in tf.getmembers() if m.name.endswith('.m4a')]
        for m in mem:
            parts = Path(m.name).parts; emo, lvl = parts[-3], int(parts[-2].replace('level_', ''))
            if emo not in CLS or (emo != 'neutral' and lvl not in MIX_LEVELS):
                continue
            tf.extract(m, tmp, filter='data')
            out.setdefault(emo, []).append(A.load_wav_16k(tmp / m.name))
    return out


def make_mix(clips, rng):                     # = Cell 12
    seq, prev = [], None
    for _ in range(SEG_PER_MIX):
        e = 'neutral' if prev is None else rng.choice([c for c in CLS if c != prev])
        seq.append(e); prev = e
    gap = np.zeros(int(GAP_SEC * 16000), np.float32)
    ys, segs, t = [], [], 0.0
    for e in seq:
        n = rng.randint(*CLIPS_PER_SEG)
        part = np.concatenate(sum([[c, gap] for c in rng.sample(clips[e], n)], []))
        ys.append(part); segs.append((t, t + len(part) / 16000, e)); t += len(part) / 16000
    return np.concatenate(ys), segs


rng, found = random.Random(0), None
for sp in MIX_SPEAKERS:
    clips = load_spk_clips(sp)
    if any(len(clips.get(c, [])) < CLIPS_PER_SEG[1] for c in CLS):
        continue
    for k in range(MIX_PER_SPK):
        y_, segs = make_mix(clips, rng)
        name = '%s_MIX%d_%s' % (sp, k + 1, '-'.join(s[2][:3] for s in segs))
        if name.startswith(TARGET + '_'):
            found = (name, y_, segs); break
    if found:
        break
assert found, 'khong dung lai duoc %s' % TARGET
name, y_rebuilt, segs = found
wav = TEST_DIR / (name + '.wav')
sr, y_saved = wavfile.read(str(wav))
rebuilt16 = (np.clip(y_rebuilt, -1, 1) * 32767).astype(np.int16)
same = len(rebuilt16) == len(y_saved) and np.array_equal(rebuilt16, y_saved)
print('%s | dap an dung lai: %d doan | audio trung file da luu: %s (%d / %d mau)' % (name, len(segs), same, len(rebuilt16), len(y_saved)))
assert len(rebuilt16) == len(y_saved), 'do dai khac -> dap an dung lai khong khop file da luu'
y = y_saved.astype(np.float32) / 32767.0                        # du doan tren CHINH file da luu
save = dict(name=np.array(name), dur=np.array(len(y) / 16000), classes=np.array(CLS), audio_identical=np.array(same),
            gt_start=np.array([s[0] for s in segs]), gt_end=np.array([s[1] for s in segs]), gt_emo=np.array([s[2] for s in segs]))
for key, mdl in (('ens', 'phaseG+e2v'), ('spex', 'phaseG'), ('teacher', 'emotion2vec')):
    R = analyze_emotion(y, name, use=mdl, plot=False, verbose=False)
    save[key + '_centers'] = np.asarray(R['centers'], np.float32)
    save[key + '_prob'] = np.stack([np.asarray(R['prob'])[:, A.EMO_NAMES.index(c)] for c in CLS], 1).astype(np.float32)
    save[key + '_start'] = np.array([s['start'] for s in R['segs']], np.float32)
    save[key + '_end'] = np.array([s['end'] for s in R['segs']], np.float32)
    save[key + '_emo'] = np.array([s['emo'] for s in R['segs']])
    print('  %-12s %d doan: %s' % (mdl, len(R['segs']), ' '.join('%s %.1f-%.1f' % (s['emo'][:3], s['start'], s['end']) for s in R['segs'])))
np.savez(OUT_NPZ, **save)
print('Da luu', OUT_NPZ)
