"""Phan tich ket qua MOS tu 2 file CSV xuat tu Google Sheets (Form 1, Form 2).

Gia dinh ve form (in ra anh xa cot de kiem tra truoc khi tin ket qua):
  - Cot 0 = Timestamp; tiep theo la khoi cau hoi nhan khau hoc (so cot tu suy ra hoac --n-demo1/--n-demo2);
    sau do moi video 4 cau theo dung thu tu: lip sync (1-5), emotion match (1-5), naturalness (1-5), emotion chon.
  - Video trong form theo thu tu play_order.csv; Form 1 = vi tri 1..SPLIT, Form 2 = SPLIT+1..50 (--split, mac dinh 25).
  - Ma an danh (neu co) la 1 cot trong khoi nhan khau hoc, tim theo ten cot (--id-col hoac tu dong: 'ma'/'code'/'id').
    Co ma o ca 2 form -> ghep nguoi cham giua 2 form; khong co -> phan tich RIENG tung form va bao.
Loai nguoi cham: lip sync >= 4 o bat ky video kiem tra chu y (system ATTENTION) nguoi do da xem (ghep: loai ca 2 form).
Thong ke: diem moi he = trung binh theo nguoi cham (TB cac video cua he ma nguoi do xem), 95% CI bootstrap 1000 lan
theo nguoi cham; Wilcoxon signed-rank S1 vs S2/S3/S4 (cap theo nguoi cham), hieu chinh Holm trong tung tieu chi
(3 so sanh); ti le nhan dung cam xuc theo he (gop moi luot cham + TB theo nguoi cham).

Chay:  python mos_analyze.py form1.csv form2.csv [--out ket_qua.md]
Chay thu bang du lieu GIA:  python mos_analyze.py --fake 20 --fake-dir <thu muc tam>   (moi ket qua deu ghi "DU LIEU GIA")"""
import argparse
import csv
import re
import sys
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

MOS = Path(r'C:\hien\mos')
SYSTEMS = ['S1', 'S2', 'S3', 'S4']
NAMES = {'S1': 'ELSA (full)', 'S2': 'ELSA (warp only)', 'S3': 'MakeItTalk', 'S4': 'Wav2Lip'}
QS = ['lip', 'emo', 'nat', 'choice']
QNAME = {'lip': 'Lip sync', 'emo': 'Emotion match', 'nat': 'Naturalness'}
EMO_MAP = {'neutral': 'neutral', 'trung tinh': 'neutral', 'trung tính': 'neutral', 'binh thuong': 'neutral', 'bình thường': 'neutral',
           'happy': 'happy', 'vui': 'happy', 'vui ve': 'happy', 'vui vẻ': 'happy',
           'angry': 'angry', 'gian': 'angry', 'giận': 'angry', 'gian du': 'angry', 'giận dữ': 'angry', 'tuc gian': 'angry', 'tức giận': 'angry'}
ID_RE = re.compile(r'(^|\b)(m[aã]|code|id|anonymous|an danh|ẩn danh)(\b|$)', re.I)


def likert(v):
    m = re.match(r'\s*([1-5])', str(v))
    return float(m.group(1)) if m else np.nan


def emotion(v):
    s = str(v).strip().lower()
    return EMO_MAP.get(s) or next((e for k, e in EMO_MAP.items() if s.startswith(k)), None)


def read_form(path, positions, n_demo, id_col, order, label):
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    head, data = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    nv = len(positions)
    if n_demo is None:
        n_demo = len(head) - 1 - 4 * nv
    if n_demo < 0 or 1 + n_demo + 4 * nv != len(head):
        sys.exit('%s: %d cot, nhung 1 (timestamp) + %s (nhan khau hoc) + 4 x %d video = %d -> sai so cot; kiem tra --split / --n-demo'
                 % (label, len(head), n_demo, nv, 1 + max(n_demo, 0) + 4 * nv))
    demo = list(range(1, 1 + n_demo))
    idc = None
    if id_col:
        idc = next((i for i in demo if head[i].strip().lower() == id_col.strip().lower()), None)
        if idc is None:
            sys.exit('%s: khong thay cot ma an danh %r trong khoi nhan khau hoc' % (label, id_col))
    else:
        idc = next((i for i in demo if ID_RE.search(head[i])), None)
    print('\n=== %s: %s | %d nguoi tra loi | %d cot = timestamp + %d nhan khau hoc + 4 x %d video'
          % (label, Path(path).name, len(data), len(head), n_demo, nv))
    for i in demo:
        print('  cot %3d  [nhan khau hoc%s] %s' % (i, ', MA AN DANH' if i == idc else '', head[i][:70]))
    cols = {}
    for j, pos in enumerate(positions):
        v = order[pos]
        for q, qn in enumerate(QS):
            c = 1 + n_demo + 4 * j + q
            cols[(v, qn)] = c
            print('  cot %3d  vi tri %2d  %-7s %-6s <- %s' % (c, pos, v, qn, head[c][:60]))
    if idc is None:
        print('  (khong tim thay cot ma an danh)')
    raters = []
    for k, r in enumerate(data):
        rid = r[idc].strip().upper() if idc is not None else ''
        rec = {'form': label, 'id': rid or '%s#%d' % (label, k + 1), 'has_id': bool(rid), 'ans': {}}
        for (v, qn), c in cols.items():
            rec['ans'][(v, qn)] = emotion(r[c]) if qn == 'choice' else likert(r[c])
        raters.append(rec)
    return raters, idc is not None


def boot_ci(x, n=1000, seed=0):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def analyse(raters, key, title, fake):
    tag = ' [DU LIEU GIA - KHONG DUNG CHO BAI BAO]' if fake else ''
    att = [v for v, r in key.items() if r['system'] == 'ATTENTION']
    excl = [r for r in raters if any(r['ans'].get((v, 'lip'), np.nan) >= 4 for v in att)]
    keep = [r for r in raters if r not in excl]
    out = ['## %s%s' % (title, tag), '',
           'Raters: %d total, %d excluded by the attention check (lip sync >= 4 on the delayed-audio video), %d analysed.'
           % (len(raters), len(excl), len(keep)), '']
    # diem TB theo nguoi cham x he x tieu chi
    M = {q: {s: np.array([np.nanmean([r['ans'][(v, q)] for v in key if key[v]['system'] == s and (v, q) in r['ans']] or [np.nan])
                          for r in keep]) for s in SYSTEMS} for q in QNAME}
    out += ['| System | ' + ' | '.join('%s (95%% CI)' % QNAME[q] for q in QNAME) + ' |',
            '|---|' + '---|' * len(QNAME)]
    for s in SYSTEMS:
        cells = []
        for q in QNAME:
            x = M[q][s]; lo, hi = boot_ci(x)
            cells.append('%.2f (%.2f–%.2f)' % (np.nanmean(x), lo, hi) if np.isfinite(x).any() else '–')
        out.append('| %s %s | %s |' % (s, NAMES[s], ' | '.join(cells)))
    out += ['', 'Wilcoxon signed-rank, S1 vs each system (paired by rater), Holm-corrected within each criterion:', '',
            '| Criterion | Comparison | n pairs | Mean diff (S1 − Sx) | W | p | p (Holm) |', '|---|---|---|---|---|---|---|']
    for q in QNAME:
        res = []
        for s in SYSTEMS[1:]:
            a, b = M[q]['S1'], M[q][s]; ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            if ok.sum() >= 2 and np.any(d != 0):
                w = wilcoxon(a[ok], b[ok]); res.append((s, ok.sum(), d.mean(), w.statistic, w.pvalue))
            else:
                res.append((s, ok.sum(), d.mean() if ok.any() else np.nan, np.nan, np.nan))
        ph = holm([np.nan_to_num(r[4], nan=1.0) for r in res])
        for (s, n, md, w, p), pa in zip(res, ph):
            out.append('| %s | S1 vs %s | %d | %+.2f | %s | %s | %s |' % (QNAME[q], s, n, md, '%.1f' % w if np.isfinite(w) else '–',
                                                                          '%.4f' % p if np.isfinite(p) else '–', '%.4f' % pa if np.isfinite(p) else '–'))
    out += ['', 'Emotion recognition (chosen emotion = true emotion of the audio):', '',
            '| System | Correct / answered (pooled) | Accuracy (pooled) | Mean per-rater accuracy (95% CI) |', '|---|---|---|---|']
    for s in SYSTEMS:
        vids = [v for v in key if key[v]['system'] == s]
        hit = tot = 0; per = []
        for r in keep:
            a = [(r['ans'][(v, 'choice')] == key[v]['emotion_true']) for v in vids if r['ans'].get((v, 'choice'))]
            hit += sum(a); tot += len(a)
            per.append(np.mean(a) if a else np.nan)
        per = np.array(per); lo, hi = boot_ci(per)
        out.append('| %s %s | %d / %d | %s | %s |' % (s, NAMES[s], hit, tot, '%.1f%%' % (100 * hit / tot) if tot else '–',
                                                    '%.1f%% (%.1f–%.1f%%)' % (100 * np.nanmean(per), 100 * lo, 100 * hi) if np.isfinite(per).any() else '–'))
    out.append('')
    return out


def make_fake(n, d, order, key, split):
    """Du lieu GIA: n nguoi cham, moi nguoi lam ca 2 form (cung ma an danh), diem ngau nhien quanh muc theo he.
    Chi de chay thu script. 2 nguoi co y tra loi sai cau kiem tra chu y."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    base = {'S1': 3.8, 'S2': 3.3, 'S3': 3.1, 'S4': 2.6, 'ATTENTION': 1.6}
    emos = ['Neutral', 'Happy', 'Angry']
    paths = []
    for f, poss in (('1', range(1, split + 1)), ('2', range(split + 1, len(order) + 1))):
        head = ['Timestamp', 'Mã ẩn danh (FAKE)', 'Tuổi', 'Giới tính']
        for p in poss:
            head += ['Video %d - Lip sync' % p, 'Video %d - Emotion match' % p, 'Video %d - Naturalness' % p, 'Video %d - Emotion' % p]
        rows = [head]
        for k in range(n):
            r = ['2026-10-26 10:%02d' % k, 'FAKE%03d' % k, str(rng.integers(19, 40)), rng.choice(['Nam', 'Nữ'])]
            bias = rng.normal(0, 0.4)
            for p in poss:
                kv = key[order[p]]; mu = base[kv['system']] + bias
                if kv['system'] == 'ATTENTION' and k < 2:
                    mu = 4.6
                sc = [str(int(np.clip(round(rng.normal(mu, 0.8)), 1, 5))) for _ in range(3)]
                ok = rng.random() < {'S1': .7, 'S2': .6, 'S3': .4, 'S4': .35, 'ATTENTION': .5}[kv['system']]
                r += sc + [kv['emotion_true'].capitalize() if ok else rng.choice(emos)]
            rows.append(r)
        p = d / ('FAKE_form%s.csv' % f)
        with open(p, 'w', newline='', encoding='utf-8-sig') as fh:
            csv.writer(fh).writerows(rows)
        paths.append(p)
    return paths


def main():
    sys.stdout.reconfigure(encoding='utf-8')     # ten cot tieng Viet tren console Windows (cp1252)
    ap = argparse.ArgumentParser()
    ap.add_argument('form1', nargs='?'); ap.add_argument('form2', nargs='?')
    ap.add_argument('--play-order', default=str(MOS / 'play_order.csv'))
    ap.add_argument('--key', default=str(MOS / '_key' / 'key.csv'))
    ap.add_argument('--split', type=int, default=25, help='Form 1 = vi tri 1..SPLIT trong play_order, Form 2 = phan con lai')
    ap.add_argument('--n-demo1', type=int); ap.add_argument('--n-demo2', type=int)
    ap.add_argument('--id-col', help='ten cot ma an danh (mac dinh: tu tim)')
    ap.add_argument('--out', help='ghi bang markdown ra file nay')
    ap.add_argument('--fake', type=int, help='chay thu voi N nguoi cham GIA')
    ap.add_argument('--fake-dir', help='thu muc ghi CSV gia (bat buoc khi --fake)')
    a = ap.parse_args()
    order = {int(r['order']): r['video'][:-4] for r in csv.DictReader(open(a.play_order, encoding='utf-8'))}
    key = {r['code']: r for r in csv.DictReader(open(a.key, encoding='utf-8'))}
    fake = bool(a.fake)
    if fake:
        if not a.fake_dir:
            sys.exit('--fake can --fake-dir (thu muc tam, KHONG phai thu muc ket qua)')
        a.form1, a.form2 = make_fake(a.fake, Path(a.fake_dir), order, key, a.split)
        print('*** DU LIEU GIA (%d nguoi cham ngau nhien) - chi de chay thu script ***' % a.fake)
        if a.out and 'FAKE' not in Path(a.out).name.upper():
            sys.exit('--out voi du lieu gia phai co chu FAKE trong ten file')
    if not (a.form1 and a.form2):
        sys.exit('can 2 file CSV (form1, form2)')
    n = len(order)
    r1, id1 = read_form(a.form1, list(range(1, a.split + 1)), a.n_demo1, a.id_col, order, 'Form 1')
    r2, id2 = read_form(a.form2, list(range(a.split + 1, n + 1)), a.n_demo2, a.id_col, order, 'Form 2')
    out = []
    if id1 and id2:
        by = {}
        for r in r1 + r2:
            if not r['has_id']:
                continue
            m = by.setdefault(r['id'], {'id': r['id'], 'ans': {}, 'forms': []})
            if r['form'] in m['forms']:
                print('  [canh bao] ma %s nop %s nhieu lan - giu lan cuoi' % (r['id'], r['form']))
            else:
                m['forms'].append(r['form'])
            m['ans'].update(r['ans'])
        both = sum(len(m['forms']) == 2 for m in by.values())
        noid = sum(not r['has_id'] for r in r1 + r2)
        print('\nGhep theo ma an danh: %d ma, %d co ca 2 form, %d chi 1 form; %d luot khong co ma (bo qua)'
              % (len(by), both, len(by) - both, noid))
        out = analyse(list(by.values()), key, 'MOS results (raters matched across Form 1 and Form 2: %d with both forms, %d with one)'
                      % (both, len(by) - both), fake)
    else:
        print('\n[!] Khong co cot ma an danh o %s -> KHONG ghep duoc nguoi cham; phan tich rieng tung form.'
              % ' va '.join(f for f, ok in (('Form 1', id1), ('Form 2', id2)) if not ok))
        out = analyse(r1, key, 'MOS results - Form 1 only', fake) + analyse(r2, key, 'MOS results - Form 2 only', fake)
    print('\n' + '\n'.join(out))
    if a.out:
        Path(a.out).write_text(('<!-- DU LIEU GIA -->\n' if fake else '') + '\n'.join(out), encoding='utf-8')
        print('-> %s' % a.out)


if __name__ == '__main__':
    main()
