# Cell them vao CUOI notebook 8_phaseH_mouth_amplitude.ipynb (chay sau Cell 9), tren Colab T4.
# Muc dich: luu du lieu do mo moi cua 2 clip test trong phaseH_mouth_open.png de ve lai hinh tieng Anh tren may
# (scripts/mos/fig_lip_gap_en.py). Notebook 8 KHONG luu du lieu nay (du doan C4 chi nam trong cache /content).
# Khong train: notebook 8 chi suy luan C4 + tim gain tren VAL (~10-15 phut). Ghi 1 file MOI, khong ghi de file cu:
#   student_a2l/phaseH/phaseH_mouth_open_curves.npz
# Dung cung `pick`, `CURVES`, `mouth_open`, `DTE` cua Cell 9 -> trung khop phaseH_mouth_open.png.
_save = {}
for k, i in enumerate(pick):
    c = DTE['clips'][i]; a_, b_ = c['pos'], c['pos'] + c['n']; ref = DTE['refs'][c['sp']]
    _save['c%d_name' % k] = np.array('%s | %s level %d' % (c['name'], EMO_NAMES[c['emo']], c['lvl']))
    _save['c%d_t' % k] = np.arange(c['n']) / 62.5
    _save['c%d_real' % k] = mouth_open(DTE['raw'][a_:b_] + ref)
    for j, (lab, P_, _) in enumerate(CURVES):     # 0 C4 truoc, 1 C4 + gain, 2 demo cu, 3 demo + gain
        _save['c%d_curve%d' % (k, j)] = mouth_open(P_[a_:b_] + ref)
_save['curve_labels'] = np.array([lab for lab, _, _ in CURVES])
_f = OUT / 'phaseH_mouth_open_curves.npz'
assert not _f.exists(), 'Da co %s - doi ten neu muon luu lai' % _f
np.savez(_f, **_save)
print('Da luu', _f, '| clip:', [str(_save['c%d_name' % k]) for k in range(len(pick))])
