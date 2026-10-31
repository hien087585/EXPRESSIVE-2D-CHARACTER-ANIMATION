# ==== BASELINE SadTalker + Wav2Lip: chi so nhu Bang 2 (158 clip neutral/angry/happy) ==========================
# Dan cell nay vao CUOI notebook 5_phaseE_benchmark.ipynb (KHONG sua notebook) va chay SAU khi da chay het notebook
# voi cau hinh cua Bang 2: STATIC_SOURCE = 'spex_kd', APPLY_DEMO_CHAIN = True, MOUTH_GAIN_PATH = mouth_gain.json,
# BOOT_CI = True, RUN_MAKEITTALK = True (cung phien Colab: dung lai DTE, DVA, PRED, MIT_PRED, CLF, clip_stats, aggregate,
# boot_ci, ST, ST_GT, LAG ... cua notebook).
# - Chi DOC: baseline_landmarks.npz (eval_baselines.py extract), params.json cua lan chay SadTalker/Wav2Lip.
# - Chi GHI vao thu muc MOI phaseH/baselines_158/, ten file co thoi diem -> khong ghi de file nao.
# Quy tac lag (nhu MakeItTalk o Bang 2): best_lag() cua notebook tren 40 clip val, L = -12..24 (khung 62.5 fps);
#   khung GT k <-> khung du doan k + LEAD - L; voi ELSA / SadTalker / Wav2Lip, L = LEAD = 8 la khop thang (dich = L - 8).
import ast, csv, json, time, textwrap
from pathlib import Path
import numpy as np, torch

LM_NPZ = DRIVE / 'baselines' / 'baseline_landmarks.npz'            # tai len tu <LOCAL_OUT>/baselines
RUNS_DIR = DRIVE / 'baselines' / 'out'                              # thu muc OUT_ROOT cua colab_sadtalker_wav2lip_cell.py
OUT_B = DRIVE / 'student_a2l' / 'phaseH' / 'baselines_158'
OUT_B.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime('%Y%m%d_%H%M%S')
TABLE2_158 = {'MakeItTalk': dict(LMD=0.0613, M_LMD_mieng=0.0453, Tuong_quan=0.275, Bien_do=0.569, Khep_khi_im=0.0934, Emo_Acc=0.013),
              'MakeItTalk x2': dict(LMD=0.0630, M_LMD_mieng=0.0506, Tuong_quan=0.278, Bien_do=1.120, Khep_khi_im=0.1098, Emo_Acc=0.007),
              'ELSA (fully automatic)': dict(LMD=0.0343, M_LMD_mieng=0.0270, Tuong_quan=0.580, Bien_do=0.931, Khep_khi_im=0.0492,
                                             Emo_Acc=0.934)}

# ------------------------------------------------------------------ kiem tra dung phien notebook 5 (cau hinh Bang 2)
for _v in ('DTE', 'DVA', 'PRED', 'MIT_PRED', 'ELSA_NAME', 'MIT3', 'KEEP3', 'ST', 'ST_GT', 'clip_stats', 'aggregate', 'boot_ci',
           'LAG', 'RES_MIT3', 'SPX_BR', 'GAIN_ARGS', 'demo_chain', 'predict_clip', 'C4_PTS', 'NPARAM'):
    assert _v in globals(), 'Thieu %s: chay het notebook 5 voi cau hinh Bang 2 truoc (xem dau cell)' % _v
assert ELSA_NAME and MIT3 and BOOT_CI, 'Can STATIC_SOURCE=spex_kd, RUN_MAKEITTALK, BOOT_CI=True'
MIT1, MIT2 = 'MakeItTalk goc (moi x1)', 'MakeItTalk goc (moi x2)'
NAMES = {MIT1: 'MakeItTalk', MIT2: 'MakeItTalk x2', ELSA_NAME: 'ELSA (fully automatic)'}

# ------------------------------------------------------------------ ham cu: best_lag / align (notebook 5), sim_fit (MOS S3)
_src5 = next(''.join(c['source']) for c in json.loads((REPO_DIR / 'notebooks' / '5_phaseE_benchmark.ipynb').read_text(
    encoding='utf-8'))['cells'] if 'def best_lag(' in ''.join(c['source']))
for _n in ast.walk(ast.parse(_src5)):
    if isinstance(_n, ast.FunctionDef) and _n.name in ('best_lag', 'align'):
        exec(textwrap.dedent(' ' * _n.col_offset + ast.get_source_segment(_src5, _n)), globals())   # ham long trong 'if'
_s3 = (REPO_DIR / 'scripts' / 'mos' / 'mos_s3_job.py').read_text(encoding='utf-8')
exec(ast.get_source_segment(_s3, next(n for n in ast.parse(_s3).body if isinstance(n, ast.FunctionDef) and n.name == 'sim_fit')))
assert 'ST = list(range(17, 31)) + list(range(36, 48))' in _s3
POSE_PTS = list(range(17, 31)) + list(range(36, 48))              # may 17-26, song mui 27-30, mat 36-47 (nhu S3)

# ------------------------------------------------------------------ clip val (cung cach chon voi MakeItTalk) + du doan baseline
VAL_IDS = sorted(np.random.default_rng(SEED).choice(len(DVA['clips']), min(MIT_VAL_CLIPS, len(DVA['clips'])), replace=False).tolist())
Z = np.load(LM_NPZ); ZMETA = json.loads(str(Z['_meta']))
print('baseline_landmarks.npz: mediapipe %s | %d mang' % (ZMETA['mediapipe'], len(Z.files) - 1))
te_name = {c['name']: ci for ci, c in enumerate(DTE['clips'])}
va_name = {c['name']: ci for ci, c in enumerate(DVA['clips'])}
BASE = {'sadtalker': 'SadTalker', 'wav2lip': 'Wav2Lip'}
RAW_TE, RAW_VA = {}, {}
for m, nm in BASE.items():
    RAW_TE[nm] = {te_name[k.split('/', 1)[1]]: Z[k] for k in Z.files if k.startswith(m + '/') and k.split('/', 1)[1] in te_name}
    RAW_VA[nm] = {va_name[k.split('/', 1)[1]]: Z[k] for k in Z.files if k.startswith(m + '/') and k.split('/', 1)[1] in va_name}
    miss_v = [ci for ci in VAL_IDS if ci not in RAW_VA[nm]]
    print('  %-9s test %d / %d clip | val %d / %d clip%s' % (nm, len([ci for ci in MIT3 if ci in RAW_TE[nm]]), len(MIT3),
                                                          len(VAL_IDS) - len(miss_v), len(VAL_IDS), (' | THIEU val: %s' % miss_v) if miss_v else ''))

# ------------------------------------------------------------------ ELSA tu dong tren 40 clip val (cung code / checkpoint nhu test)
EMB_V, HAS_V = np.zeros((len(DVA['clips']), SPX['branch_cfg']['in_dim']), np.float32), np.zeros(len(DVA['clips']), bool)
for sp in SPLIT['val']:
    with np.load(E2V_DIR / (sp + '.npz')) as f:
        chunks, counts, coff, has, fnames = f['chunks'], f['counts'], f['coff'], f['has'], [str(x) for x in f['names']]
    row = {n: i for i, n in enumerate(fnames)}
    for ci, c in enumerate(DVA['clips']):
        i = row.get(c['name']) if c['sp'] == sp else None
        if i is None or not has[i]:
            continue
        w_ = counts[coff[i]:coff[i + 1]].astype(np.float32)
        EMB_V[ci] = (chunks[coff[i]:coff[i + 1]].astype(np.float32) * w_[:, None]).sum(0) / max(w_.sum(), 1.0); HAS_V[ci] = True
    del chunks
with torch.no_grad():
    _S, _L, _V = SPX_BR(torch.from_numpy(EMB_V), torch.zeros(len(EMB_V), 7), torch.zeros(len(EMB_V)))
    _km = torch.full((8,), float('-inf')); _km[KEEP3] = 0.0
    _p = torch.softmax(_L.float() + _km, -1)
    S_V = (_S.numpy() * float(SPX['zero_lmd'])).astype(np.float32)
    COND_V = (_p[:, 1:] * (_V.float()[:, None] / 3.0)).numpy().astype(np.float32)
_m = StudentAudio2Landmark.from_checkpoint(str(MODELS[GAIN_BASE]), map_location='cpu').to(DEVICE).eval()
_mask = np.zeros((68, 2), np.float32); _mask[C4_PTS] = 1; _mask = _mask.reshape(136)
_PV = np.full_like(DVA['raw'], np.nan)
for ci in VAL_IDS:
    c = DVA['clips'][ci]
    if HAS_V[ci]:
        p = predict_clip(_m, DVA['mel'][c['mel_i']], COND_V[ci], True)
        _PV[c['pos']:c['pos'] + c['n']] = S_V[ci] + (p - p.mean(0)) * _mask
_g = globals(); _dte = _g['DTE']; _g['DTE'] = DVA                  # demo_chain cua notebook duyet DTE -> tam tro sang DVA
try:
    ELSA_VA = demo_chain(_PV.astype(np.float16), S_V, **(GAIN_ARGS or {})).astype(np.float32)
finally:
    _g['DTE'] = _dte
del _m, _PV
RAW_VA['ELSA (fully automatic)'] = {ci: ELSA_VA[DVA['clips'][ci]['pos']:DVA['clips'][ci]['pos'] + DVA['clips'][ci]['n']]
                                    for ci in VAL_IDS if HAS_V[ci]}
print('ELSA tu dong tren val: %d / %d clip co emotion2vec+' % (len(RAW_VA['ELSA (fully automatic)']), len(VAL_IDS)))

# ------------------------------------------------------------------ lag: cung quy tac cho moi he thong
LAGS = {'MakeItTalk': dict(lag=int(LAG), shift=int(LAG) - LEAD, source='notebook 5 (best_lag tren cung 40 clip val)'),
        'MakeItTalk x2': dict(lag=int(LAG), shift=int(LAG) - LEAD, source='lag cua MakeItTalk x1 (nhu notebook 5)')}
for nm, raw in RAW_VA.items():
    L_, c_ = best_lag(raw, DVA)
    c8 = best_lag(raw, DVA, lags=[LEAD])[1]
    LAGS[nm] = dict(lag=int(L_), shift=int(L_) - LEAD, corr_best=float(c_), corr_no_shift=float(c8), n_val=len(raw))
for nm, d in LAGS.items():
    print('  lag %-24s L = %3d (dich %+d khung = %+.0f ms)%s' % (nm, d['lag'], d['shift'], d['shift'] * 16,
          ' | tuong quan val %.3f (khong dich %.3f)' % (d['corr_best'], d['corr_no_shift']) if 'corr_best' in d else ''))

# ------------------------------------------------------------------ du doan test (khung 62.5 fps cua GT), NaN = khong co
def full_from(raw_by_ci, L):
    P = np.full_like(DTE['raw'], np.nan, dtype=np.float32)
    for ci, p in raw_by_ci.items():
        c = DTE['clips'][ci]; P[c['pos']:c['pos'] + c['n']] = align(p, c, L)
    return P


NEW = {nm: full_from(RAW_TE[nm], LAGS[nm]['lag']) for nm in BASE.values()}
ELSA_TUNED = None
if LAGS['ELSA (fully automatic)']['lag'] != LEAD:
    ELSA_TUNED = 'ELSA (tuned lag)'
    _pe = PRED[ELSA_NAME].astype(np.float32)
    NEW[ELSA_TUNED] = full_from({ci: _pe[DTE['clips'][ci]['pos']:DTE['clips'][ci]['pos'] + DTE['clips'][ci]['n']]
                                 for ci in MIT3}, LAGS['ELSA (fully automatic)']['lag'])
    LAGS[ELSA_TUNED] = LAGS['ELSA (fully automatic)']
LAGS['ELSA (fully automatic)'] = dict(LAGS['ELSA (fully automatic)'], used_in_main_row=LEAD)   # dong chinh: lag 0 nhu bai
t0 = time.time()
ST_NEW = {nm: clip_stats(P, DTE) for nm, P in NEW.items()}
print('Thong ke tung clip (baseline moi): %.0f s' % (time.time() - t0))
STX = dict(ST, **ST_NEW)
SYS = [MIT1, MIT2] + list(BASE.values()) + [ELSA_NAME] + ([ELSA_TUNED] if ELSA_TUNED else [])
label = lambda k: NAMES.get(k, k)

# tap clip chung: 158 clip, bo clip ma BAT KY he thong nao khong co du doan hop le
IDS = [ci for ci in MIT3 if all(STX[k]['valid'][ci] for k in SYS)]
DROP = [DTE['clips'][ci]['name'] for ci in MIT3 if ci not in IDS]
print('\nTap so sanh: %d / %d clip%s' % (len(IDS), len(MIT3), (' | bo (thieu du doan hop le): %s' % DROP) if DROP else ''))

# ------------------------------------------------------------------ doi chieu voi Bang 2 (ELSA, MakeItTalk tren dung 158 clip)
KEYS = ['LMD', 'M_LMD_mieng', 'Tuong_quan', 'Bien_do', 'Khep_khi_im', 'Emo_Acc']
w158 = np.zeros(len(DTE['clips'])); w158[MIT3] = 1.0
print('\nDoi chieu Bang 2 (158 clip, lag nhu bai):')
MISMATCH = []
for k in (MIT1, MIT2, ELSA_NAME):
    r = aggregate(STX[k], w158, ST_GT)
    for m_ in KEYS:
        same_nb = abs(r[m_] - RES_MIT3[k][m_]) < 1e-6 * max(1, abs(r[m_]))
        dec = 3 if m_ in ('Tuong_quan', 'Bien_do', 'Emo_Acc') else 4
        same_tb = round(r[m_], dec) == round(TABLE2_158[label(k)][m_], dec)
        if not (same_nb and same_tb):
            MISMATCH.append((label(k), m_, r[m_], RES_MIT3[k][m_], TABLE2_158[label(k)][m_]))
    print('  %-24s %s' % (label(k), ' | '.join('%s %.4f' % (m_, r[m_]) for m_ in KEYS)))
print('  -> TRUNG Bang 2' if not MISMATCH else '  !! LECH: ' + '; '.join('%s %s = %.4f (notebook %.4f, bai %.4f)' % x for x in MISMATCH))

# ------------------------------------------------------------------ bang chinh + CI bootstrap (cung mau clip cho moi dong)
_g['ST'], _ST0 = STX, ST
try:
    CI_B = boot_ci({k: None for k in SYS}, IDS)
finally:
    _g['ST'] = _ST0
w = np.zeros(len(DTE['clips'])); w[IDS] = 1.0
RES_B = {k: aggregate(STX[k], w, ST_GT) for k in SYS}
NA_W2L = {'LMD', 'LMD_mieng', 'Emo_Acc', 'Emo_UAR', 'Emo_Acc_cd', 'Emo_UAR_cd'}  # Wav2Lip: nua tren mat la anh tinh
PAR = json.loads((RUNS_DIR / 'params.json').read_text()) if (RUNS_DIR / 'params.json').exists() else {}
par_of = {MIT1: NPARAM.get('MakeItTalk goc'), MIT2: NPARAM.get('MakeItTalk goc'), ELSA_NAME: NPARAM.get(ELSA_NAME),
          'SadTalker': sum(PAR.get('sadtalker', {}).values()) or None, 'Wav2Lip': sum(PAR.get('wav2lip', {}).values()) or None}
if ELSA_TUNED:
    par_of[ELSA_TUNED] = NPARAM.get(ELSA_NAME)
summ = []
print('\n158 clip (neutral / angry / happy), %d clip chung | gia tri [CI 95%%]' % len(IDS))
for k in SYS:
    cells = []
    for m_ in KEYS:
        na = label(k) == 'Wav2Lip' and m_ in NA_W2L
        v, (lo, hi) = (float('nan'), (float('nan'), float('nan'))) if na else (RES_B[k][m_], CI_B[k][m_])
        summ.append(dict(system=label(k), metric=m_, value='n/a' if na else v, ci_lo='' if na else lo, ci_hi='' if na else hi))
        cells.append('%s n/a' % m_ if na else '%s %.4f [%.4f, %.4f]' % (m_, v, lo, hi))
    summ.append(dict(system=label(k), metric='params', value=par_of.get(k) or '', ci_lo='', ci_hi=''))
    print('  %-24s %s' % (label(k), ' | '.join(cells)))

# ------------------------------------------------------------------ Wilcoxon ghep cap + Holm (3 so sanh / chi so)
from scipy.stats import wilcoxon
PER = dict(LMD=lambda S: S['lmd'] / np.maximum(S['n'], 1), M_LMD_mieng=lambda S: S['mlmd'] / np.maximum(S['n'], 1),
           Tuong_quan=lambda S: S['corr'], Bien_do_lech_1=lambda S: np.abs(S['amp'] - 1.0),
           Khep_khi_im=lambda S: np.where(S['nsil'] > 0, S['sil'] / np.maximum(S['nsil'], 1), np.nan),
           Emo_Acc=lambda S: S['emo_ok'].astype(float))
WIL = []
for m_, f in PER.items():
    fam = []
    for b in (MIT1, 'SadTalker', 'Wav2Lip'):
        if label(b) == 'Wav2Lip' and m_ in NA_W2L:
            continue
        x, y = f(STX[ELSA_NAME])[IDS], f(STX[b])[IDS]
        ok = ~(np.isnan(x) | np.isnan(y))
        try:
            p_ = float(wilcoxon(x[ok], y[ok]).pvalue)
        except ValueError:
            p_ = float('nan')
        fam.append(dict(metric=m_, baseline=label(b), n=int(ok.sum()), median_elsa=float(np.median(x[ok])),
                        median_base=float(np.median(y[ok])), mean_elsa=float(np.mean(x[ok])), mean_base=float(np.mean(y[ok])), p=p_))
    order = sorted(range(len(fam)), key=lambda i: fam[i]['p'])     # Holm: p_(i) * (m - i), don dieu, toi da 1
    run = 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, fam[i]['p'] * (len(fam) - r_)))
        fam[i]['p_holm'], fam[i]['holm_family'] = run, len(fam)
    WIL += fam
print('\nWilcoxon ghep cap theo clip (2 phia), ELSA tu dong vs baseline, Holm trong moi chi so:')
for r in WIL:
    print('  %-15s vs %-10s n=%3d | median %.4g vs %.4g | p = %.3g | Holm (%d) = %.3g'
          % (r['metric'], r['baseline'], r['n'], r['median_elsa'], r['median_base'], r['p'], r['holm_family'], r['p_holm']))

# ------------------------------------------------------------------ do nhay: khu tu the dau TUNG khung (moi he thong + GT)
def pose_free(P, D, ids):
    Q = np.full_like(P, np.nan, dtype=np.float32)
    for ci in ids:
        c = D['clips'][ci]; a, b = c['pos'], c['pos'] + c['n']
        if np.isnan(P[a:b]).any():
            continue
        ref = D['refs'][c['sp']].reshape(68, 2).astype(np.float64)
        X = ref[None] + P[a:b].astype(np.float64).reshape(-1, 68, 2)
        for t in range(len(X)):
            s_, R_, t_ = sim_fit(X[t, POSE_PTS], ref[POSE_PTS])
            X[t] = s_ * X[t] @ R_ + t_
        Q[a:b] = (X - ref[None]).reshape(-1, 136)
    return Q


t0 = time.time()
D_PF = dict(DTE, raw=pose_free(DTE['raw'], DTE, IDS))
ST_GT_PF = clip_stats(D_PF['raw'], D_PF)
ALLP = {**{k: (MIT_PRED[k] if k in MIT_PRED else PRED[k]) for k in (MIT1, MIT2, ELSA_NAME)}, **NEW}
ST_PF = {k: clip_stats(pose_free(ALLP[k].astype(np.float32), DTE, IDS), D_PF) for k in SYS}
SENS = []
print('\nDo nhay (khu tu the dau tung khung tren may/mui/mat, ca GT; chi so hinh hoc, Emo-Acc n/a) - %.0f s' % (time.time() - t0))
for k in SYS:
    r = aggregate(ST_PF[k], w, ST_GT_PF)
    row = dict(system=label(k))
    for m_ in KEYS[:5]:
        row[m_] = 'n/a' if (label(k) == 'Wav2Lip' and m_ in NA_W2L) else r[m_]
    row['Emo_Acc'] = 'n/a'
    SENS.append(row)
    print('  %-24s %s' % (label(k), ' | '.join('%s %s' % (m_, v if v == 'n/a' else '%.4f' % v) for m_, v in row.items() if m_ != 'system')))

# ------------------------------------------------------------------ luu (file MOI, co thoi diem)
def wcsv(name, rows):
    f = OUT_B / ('baselines_158_%s_%s.csv' % (name, STAMP))
    assert not f.exists()
    with open(f, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return f


pc = []
for k in SYS:
    S, Sp = STX[k], ST_PF[k]
    for ci in MIT3:
        c = DTE['clips'][ci]
        d = dict(clip_index=ci, clip_id=c['name'], speaker=c['sp'], emotion=EMO_NAMES[c['emo']], level=c['lvl'], system=label(k),
                 lag_L=LAGS.get(label(k), {}).get('used_in_main_row', LAGS.get(label(k), {}).get('lag', LEAD)),
                 in_common_set=int(ci in IDS), valid=int(S['valid'][ci]), n_frames=int(S['n'][ci]))
        for q in ('lmd', 'lmd_m', 'mlmd', 'lvd', 'corr', 'amp', 'sil', 'nsil', 'emo_n', 'emo_ok', 'emo_ok_cd'):
            d[q] = float(S[q][ci])
        d.update(lmd_mean=d['lmd'] / max(d['n_frames'], 1), mlmd_mean=d['mlmd'] / max(d['n_frames'], 1),
                 closure=d['sil'] / d['nsil'] if d['nsil'] > 0 else '',
                 sens_lmd_mean=float(Sp['lmd'][ci] / max(Sp['n'][ci], 1)), sens_mlmd_mean=float(Sp['mlmd'][ci] / max(Sp['n'][ci], 1)),
                 sens_corr=float(Sp['corr'][ci]), sens_amp=float(Sp['amp'][ci]),
                 sens_closure=float(Sp['sil'][ci] / Sp['nsil'][ci]) if Sp['nsil'][ci] > 0 else '')
        if label(k) == 'Wav2Lip':
            for q in ('lmd', 'lmd_m', 'lmd_mean', 'emo_ok', 'emo_ok_cd', 'sens_lmd_mean'):
                d[q] = 'n/a'
        pc.append(d)
files = [wcsv('per_clip', pc), wcsv('summary', summ), wcsv('wilcoxon', WIL), wcsv('sensitivity', SENS)]
fl = OUT_B / ('baselines_158_lags_%s.json' % STAMP)
fl.write_text(json.dumps(dict(LAGS, _rule='best_lag() notebook 5, 40 clip val, L in -12..24; dich = L - %d' % LEAD,
                              _common_clips=len(IDS), _dropped=DROP, _table2_check='ok' if not MISMATCH else MISMATCH,
                              _mediapipe=ZMETA['mediapipe']), indent=1, default=str))
files.append(fl)
print('\nDa luu (file moi):'); [print('  ', f) for f in files]
print('Tai cac file nay ve may roi chay: python scripts/baselines/eval_baselines.py report --runs <...> --colab <thu muc tai ve>')
