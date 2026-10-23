"""Chay notebook 3 tren may (CPU), doc Drive G: CHI DOC, ghi ket qua vao thu muc out rieng.
python run_nb3.py <notebook> <out_dir> <job> [args...]
job: equal <img> <pts json> <audio> <seconds>   -> frames .npz (de so sanh ban cu / moi khi TAT)
     demo  <img> <pts json> <audio> <seconds>   -> video bat + tat ve lai mieng
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

NBP, OUT, JOB = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ARGS = sys.argv[4:]
S = Path(__file__).parent
REPO = Path(r'C:\hien\EXPRESSIVE-2D-CHARACTER-ANIMATION')
G = Path(r'G:\.shortcut-targets-by-id\1KaV-u_oFzL06sQpotcZhWayVLKmoq0ZS\DAT301m (1)\Data_student')
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / '_tmp'; TMP.mkdir(exist_ok=True)
os.environ['PATH'] = str(S / 'bin') + os.pathsep + os.environ['PATH']
os.environ['PHASEC_TEST_ROOT'] = str(OUT / '_troot'); (OUT / '_troot').mkdir(exist_ok=True)
os.environ['PHASEC_UI_TEST'] = '1'

nb = json.loads(NBP.read_text(encoding='utf-8'))
C = [''.join(c['source']) for c in nb['cells']]
g = {'__name__': '__main__'}
exec(C[2], g)                                                           # Cell 0 (khong phai Colab: chi dinh nghia)
c6 = C[6].replace('MOUTH_GAIN_PATH = None', 'MOUTH_GAIN_PATH = %r' % str(G / 'student_a2l' / 'phaseH' / 'mouth_gain.json'), 1)
exec(c6, g)                                                             # Cell 1
# duong dan: doc tu G: (chi doc), ghi vao OUT
g['REPO_DIR'] = REPO
g['PHASEB'] = G / 'student_a2l' / 'phaseB'
g['READY'] = G / 'Data_emotion_mead_task_b' / 'phaseB_ready'
g['OUT_DIR'] = OUT / 'demo'
g['REDRAW_DIR'] = OUT / 'demo_mouth_redraw'
g['CHAR_DIRS'] = [OUT / 'demo' / 'characters', REPO / 'examples_cartoon']
g['SER_CKPT'] = G / 'student_a2l' / 'ser_mead' / 'ser_best.pth'
g['F2_PATHS'] = {'phaseG': G / 'student_a2l' / 'phaseF' / 'phaseG_final.pth',
                 'phaseF2': G / 'student_a2l' / 'phaseF' / 'phaseF2_final.pth'}
g['F2_CKPT'] = next((p for p in g['F2_PATHS'].values() if p.exists()), g['F2_PATHS']['phaseF2'])
exec(C[4], g)                                                           # Cell 1b
exec(C[8], g)                                                           # Cell 2
A = g['A']
A.load_wav_16k.__defaults__ = (str(TMP),)
A.VideoSink.__init__.__defaults__ = (None, 16000, str(TMP))
for i in (10, 12, 14, 16, 26):                                          # Cell 3, 4, 5, 6, 11 (UI dung nhung khong mo)
    t0 = time.time(); exec(C[i], g); print('-- cell %d: %.1f s' % (i, time.time() - t0), flush=True)

import cv2

img_path, pts_path, audio, secs = ARGS[0], ARGS[1], ARGS[2], float(ARGS[3])
pts = [tuple(p) for p in json.loads(Path(pts_path).read_text())]
img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
y = A.load_wav_16k(audio)[:int(secs * 16000)]
TL = [(0.0, 'neutral'), (0.55 * secs, 'neutral'), (0.62 * secs, 'happy_2')] if 'happy_2' in A.PRESETS else None

if JOB == 'equal':
    res = {}
    # (a) nhan vat co san, (b) anh moi bam 5 diem (duong man hinh demo), che do mac dinh (khong truyen co moi)
    g['setup_character']('danbooru1', verbose=False)
    res['danbooru1'] = np.stack(g['render'](y, timeline=TL, name='eq_danbooru1', save=False, keep=True, verbose=False)[0])
    res['still'] = g['still']('happy_2' if TL else 'neutral')
    nm = 'eqchar'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts[:5], nm), verbose=False)
    res['anime5'] = np.stack(g['render'](y, timeline=TL, name='eq_anime5', save=False, keep=True, verbose=False)[0])
    res['hint'] = np.array([g['hint']([]), g['hint']([1] * 3), g['hint']([1] * 5)])
    np.savez_compressed(OUT / 'equal_frames.npz', **res)
    print('saved', OUT / 'equal_frames.npz', {k: v.shape for k, v in res.items()})
elif JOB == 'demo':
    nm = 'anime_girl'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm), verbose=False)
    print('nhan vat:', g['CH'].img.shape, '| 7 diem:', g['load_mouth_kp'](nm))
    _, v_on = g['render'](y, timeline=TL, name='mouth_redraw_on', mouth_redraw=True)
    _, v_off = g['render'](y, timeline=TL, name='mouth_redraw_off', mouth_redraw=False)
    mp = g['get_mouth']()[0]
    print('mau net', mp.line, '| mau da', mp.skin, '| day net %.1f px | rong mieng %.1f px | H %.1f px' % (mp.lw, mp.W0, mp.H))
    print('VIDEO', v_on, v_off)
elif JOB == 'v2':
    g['SHOW_EMOTION_LABEL'] = False
    g['MOUTH_REDRAW_SIZE'] = float(ARGS[4]) if len(ARGS) > 4 else 1.0
    nm = 'anime_girl'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm), verbose=False)
    _, v_on = g['render'](y, timeline=TL, name='mouth_redraw_v2', mouth_redraw=True)
    mp = g['get_mouth']()[0]
    R = np.load(str(v_on).replace('.mp4', '_mouth_params.npy'))
    sc = mp.cfg['open_scale']; o = R[:, 0] * sc
    gate = A.silence_gate(y, len(R), fps=g['OUT_FPS'])
    talk, sil = gate > 0.5, gate < 0.1
    pr = lambda nm_, x: print('  %-28s n=%3d  min %.3f  trung vi %.3f  max %.3f' % (nm_, len(x), x.min(), np.median(x), x.max()) if len(x) else '')
    print('THONG KE DO MO (x open_scale %.1f, nguong duong ke %.3f, size %.1f):' % (sc, mp.cfg['line_open'], mp.cfg['size']))
    pr('ca doan', o); pr('dang noi (gate>0.5)', o[talk]); pr('im lang (gate<0.1)', o[sil])
    neu = np.arange(len(R)) < int(0.55 * secs * g['OUT_FPS'])
    pr('doan trung tinh', o[neu]); pr('trung tinh + dang noi', o[neu & talk])
    K = [mp.kind(p) for p in R]
    for part, m in (('ca doan', np.ones(len(R), bool)), ('trung tinh + dang noi', neu & talk), ('im lang', sil)):
        ks = [k for k, mm in zip(K, m) if mm]
        print('  hinh mieng %-22s' % part, {k: '%.0f%%' % (100 * ks.count(k) / max(len(ks), 1)) for k in ('khep', 'he', 'mo vua', 'mo to', 'o')})
    print('  do tron: p50 %.3f p90 %.3f max %.3f' % tuple(np.percentile(R[:, 2], [50, 90, 100])))
    print('VIDEO', v_on)
elif JOB == 'v3':
    g['SHOW_EMOTION_LABEL'] = False
    mode = ARGS[4]
    tl = TL if mode == 'mixed' else [(0.0, 'angry_2')]
    nm = 'anime_girl'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm), verbose=False)
    _, v_on = g['render'](y, timeline=tl, name='mouth_redraw_v3_' + mode, mouth_redraw=True)
    mp = g['get_mouth']()[0]
    base = str(v_on)[:-4]
    R, E = np.load(base + '_mouth_params.npy'), np.load(base + '_mouth_emo.npy')
    print('timeline:', tl, '| open_scale %.1f size %.1f' % (mp.cfg['open_scale'], mp.cfg['size']))
    gate = A.silence_gate(y, len(R), fps=g['OUT_FPS'])
    O = np.array([mp._open(p, e) for p, e in zip(R, E)]); K = [mp.kind(p, e) for p, e in zip(R, E)]
    lab = np.where(E[:, 1] > 0.5, 'angry', np.where(E[:, 2] > 0.5, 'happy', np.where(E[:, 0] < 0.05, 'neutral', 'chuyen')))
    for seg in ('neutral', 'chuyen', 'happy', 'angry'):
        for part, m in (('dang noi', (lab == seg) & (gate > 0.5)), ('im lang', (lab == seg) & (gate < 0.1))):
            if m.sum() == 0:
                continue
            ks = [k for k, mm in zip(K, m) if mm]
            print('  %-8s %-9s n=%3d | do mo min %.3f trung vi %.3f max %.3f |' % (seg, part, m.sum(), O[m].min(), np.median(O[m]), O[m].max()),
                  ' '.join('%s %.0f%%' % (k, 100 * ks.count(k) / len(ks)) for k in ('khep', 'he', 'mo vua', 'mo to', 'o')))
    if mode == 'angry':
        import cv2 as _cv
        cap = _cv.VideoCapture(str(v_on)); fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        i = int(np.argmin(np.abs(O - np.percentile(O[gate > 0.5], 85))))
        _cv.imwrite(str(OUT / 'angry_face.png'), fr[i]); print('khung mat gian #%d do mo %.3f' % (i, O[i]))
    print('VIDEO', v_on)
elif JOB == 'plain':                                                    # render nhu notebook (khong truyen co moi)
    nm = 'anime_girl'
    g['SHOW_LABEL'] = False
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts[:5], nm), verbose=False)
    _, v = g['render'](y, timeline=TL, name='plain', label=False)
    trk = A.ExpressionTrack(TL, g['BANK'], g['CH'], blink=True, seed=0)
    print('blink times (ExpressionTrack, s):', np.round(trk.blinks[trk.blinks < secs], 2))
    np.save(OUT / 'eye_lm.npy', np.load(str(v)[:-4] + '_landmarks.npy'))
    print('VIDEO', v)
elif JOB in ('v4', 'v5'):
    import cv2 as _cv
    g['SHOW_EMOTION_LABEL'] = False
    nm = 'anime_girl'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm, True, True), verbose=False)
    for k in ('eye', 'mouth', 'brow'):
        print('diem bam', k, {a: np.round(b, 1).tolist() for a, b in g['load_kp_json'](nm, k).items()})

    def frames_of(v):
        cap = _cv.VideoCapture(str(v)); F = []
        while True:
            ok, f = cap.read()
            if not ok:
                return F
            F.append(f)
    res = {}
    for mode, tl in (('mixed', TL), ('angry', [(0.0, 'angry_2')])):
        _, v = g['render'](y, timeline=tl, name='redraw_%s_' % JOB + mode, mouth_redraw=True, brow_redraw=True)
        X = dict(np.load(str(v)[:-4] + '_face_params.npz')); res[mode] = (v, X)
        b = X['blink']; pk = [i for i in range(1, len(b) - 1) if b[i] >= 0.99 and b[i - 1] < 0.99]
        hap, sm = X['emo_happy'], X['smile'] > 0.05
        print('[%s] chop (tham so): %d lan tai %s s | khung nham han / lan: %.1f' % (mode, len(pk), [round(i / 25, 2) for i in pk], (b >= 0.99).sum() / max(len(pk), 1)))
        print('[%s] hip mat: %.1f%% khung | doan neutral (happy<0.05): %.1f%% | doan happy (>=0.95): %.1f%% | mi duoi max %.2f' % (
            mode, 100 * sm.mean(), 100 * sm[hap < 0.05].mean() if (hap < 0.05).any() else 0, 100 * sm[hap >= 0.95].mean() if (hap >= 0.95).any() else 0, X['eye_lo'].max()))
        print('[%s] mi tren: ha (cuoi) max %.2f | muc gian max %.2f' % (mode, X['eye_up'].max(), X['emo_angry'].max()))
    # --- anh kiem tra
    v, X = res['mixed']; F = frames_of(v)
    i = int(np.argmax(X['blink']))
    crop = lambda f, s=1.6: _cv.resize(f[300:400, 240:530], None, fx=s, fy=s, interpolation=_cv.INTER_CUBIC)
    lab = lambda t, s: (_cv.putText(t, s, (6, 18), 0, 0.5, (0, 0, 0), 3, _cv.LINE_AA), _cv.putText(t, s, (6, 18), 0, 0.5, (255, 255, 255), 1, _cv.LINE_AA), t)[-1]
    tiles = [lab(crop(F[k]), '#%d  %.2f s  nham %.2f' % (k, k / 25, X['blink'][k])) for k in range(i - 2, i + 4)]
    _cv.imwrite(str(OUT / ('eyes_blink_check_%s.png' % JOB)), np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0))
    mp = g['get_face'](True, True)['mp']
    K = np.array([mp.kind(p, e) for p, e in zip(X['mouth_params'], X['mouth_emo'])])
    hp = X['emo_happy'] >= 0.95; wr = X['mouth_params'][:, 1] / mp.w0
    pick = []
    for nm_, m in (('happy mieng he', hp & (K == 'he') & (X['smile'] < 0.02)), ('happy mieng mo vua', hp & (K == 'mo vua') & (X['smile'] < 0.02)),
                   ('happy cuoi to, mieng rong', hp & (X['smile'] > 0))):
        idx = np.where(m)[0]
        if len(idx):
            k = idx[np.argmax(X['smile'][idx])] if 'cuoi' in nm_ else idx[len(idx) // 2]
            pick.append(lab(_cv.resize(F[k][290:560, 230:540], None, fx=1.2, fy=1.2), '%s #%d hip %.2f rong %.2fx' % (nm_, k, X['smile'][k], wr[k])))
    _cv.imwrite(str(OUT / ('eyes_smile_check_%s.png' % JOB)), np.concatenate(pick, 1))
    # --- so sanh long may: anh goc | trung tinh | vui | gian | buon (khung tinh)
    g['MOUTH_REDRAW'] = g['BROW_REDRAW'] = True
    cb = lambda f: _cv.resize(f[255:400, 225:545], None, fx=1.3, fy=1.3, interpolation=_cv.INTER_CUBIC)
    tiles = [lab(cb(g['CH'].img), 'anh goc')] + [lab(cb(g['still'](sp)), nm_) for sp, nm_ in
                                                (('neutral', 'trung tinh'), ('happy_2', 'vui (happy_2)'), ('angry_2', 'gian (angry_2)'), ('sad', 'buon (sad muc 3)'))]
    _cv.imwrite(str(OUT / ('brows_compare_%s.png' % JOB)), np.concatenate(tiles, 1))
    g['MOUTH_REDRAW'] = g['BROW_REDRAW'] = False
    v, X = res['angry']; F = frames_of(v)
    k = int(np.argmin(np.abs(np.arange(len(F)) - 150) + 1000 * (X['blink'] > 0)))
    _cv.imwrite(str(OUT / ('angry_eye_%s.png' % JOB)), lab(_cv.resize(F[k][270:400, 240:530], None, fx=3, fy=3, interpolation=_cv.INTER_CUBIC), 'gian #%d' % k))
    print('VIDEO', res['mixed'][0], res['angry'][0])
elif JOB == 'char8':
    import cv2 as _cv
    g['SHOW_EMOTION_LABEL'] = False
    nm = ARGS[4]
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm, True, True), verbose=False)
    _, v = g['render'](y, timeline=TL, name=nm + '_check8s', mouth_redraw=True, brow_redraw=True)
    X = dict(np.load(str(v)[:-4] + '_face_params.npz'))
    cap = _cv.VideoCapture(str(v)); F = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        F.append(f)
    lab = lambda t, s: (_cv.putText(t, s, (6, 18), 0, 0.5, (0, 0, 0), 3, _cv.LINE_AA), _cv.putText(t, s, (6, 18), 0, 0.5, (255, 255, 255), 1, _cv.LINE_AA), t)[-1]
    crop = lambda f: _cv.resize(f[230:560, 220:548], (260, 260), interpolation=_cv.INTER_AREA)
    bl = int(np.argmax(X['blink'])); sm = int(np.argmax(X['smile']))
    idx = sorted(set([int(i) for i in np.linspace(0, len(F) - 1, 10)] + [bl, sm]))[:12]
    tiles = [lab(crop(F[i]), '%.2fs %s%s' % (i / 25, 'CHOP ' if X['blink'][i] > 0.5 else '', 'HIP' if X['smile'][i] > 0.3 else '')) for i in idx]
    g['MOUTH_REDRAW'] = g['BROW_REDRAW'] = True
    st = [lab(crop(g['still'](sp)), t) for sp, t in (('neutral', 'tinh: trung tinh'), ('happy_2', 'tinh: vui'), ('angry_2', 'tinh: gian'), ('sad', 'tinh: buon'))]
    g['MOUTH_REDRAW'] = g['BROW_REDRAW'] = False
    rows = [np.concatenate(tiles[r:r + 4], 1) for r in range(0, 12, 4)] + [np.concatenate(st, 1)]
    _cv.imwrite(str(OUT / (nm + '_check8s_grid.png')), np.concatenate(rows, 0))
    print('%s: chop %s s | hip %.0f%% khung | video %s' % (nm, [round(i / 25, 2) for i in range(1, len(X['blink'])) if X['blink'][i] >= 0.99 and X['blink'][i - 1] < 0.99], 100 * (X['smile'] > 0.05).mean(), v))
elif JOB == 'angry3':
    import cv2 as _cv
    g['MOUTH_REDRAW'] = g['BROW_REDRAW'] = True
    lab = lambda t, s: (_cv.putText(t, s, (8, 22), 0, 0.7, (0, 0, 0), 3, _cv.LINE_AA), _cv.putText(t, s, (8, 22), 0, 0.7, (255, 255, 255), 1, _cv.LINE_AA), t)[-1]
    rows = []
    for nm in ('anime_girl', 'boy', 'girl2'):
        im = _cv.cvtColor(_cv.imread(r'C:\hien\mos\characters\%s.png' % nm), _cv.COLOR_BGR2RGB)
        P11 = [tuple(p) for p in json.loads(Path(r'C:\hien\mos\characters\%s_points.json' % nm).read_text())['points']]
        g['setup_character'](nm, g['char_from_image'](im, P11, nm, True, True), verbose=False)
        a, n = g['still']('angry_2'), g['still']('neutral')
        c = lambda f: _cv.resize(f[270:400, 240:530], None, fx=2, fy=2, interpolation=_cv.INTER_CUBIC)
        rows.append(np.concatenate([lab(c(n), nm + ' trung tinh'), lab(c(a), nm + ' gian')], 1))
    _cv.imwrite(str(OUT / 'angry_eyes_3chars.png'), np.concatenate(rows, 0)); print('ok angry3')
elif JOB == 'dump':                                                     # khung KHONG nen -> .npy (de so sanh bit)
    g['SHOW_EMOTION_LABEL'] = False
    nm = 'anime_girl'
    g['setup_character'](nm, g['char_from_image'](img_rgb, pts, nm, True, True), verbose=False)
    for mode, tl in (('mixed', TL), ('angry', [(0.0, 'angry_2')])):
        fr, _ = g['render'](y, timeline=tl, name='dump_' + mode, save=False, keep=True, verbose=False, mouth_redraw=True, brow_redraw=True)
        np.save(OUT / ('frames_%s.npy' % mode), np.stack(fr))
    print('dump ok')
elif JOB == 'mos':
    exec(open(S / 'mos_job.py', encoding='utf-8').read(), {'g': g, '__file__': str(S / 'mos_job.py'), '__name__': 'mos_job'})
elif JOB == 's3':
    exec(open(S / 'mos_s3_job.py', encoding='utf-8').read(), {'g': g, '__file__': str(S / 'mos_s3_job.py'), '__name__': 'mos_s3',
                                                              'ARGS_S3': ARGS[4:]})
