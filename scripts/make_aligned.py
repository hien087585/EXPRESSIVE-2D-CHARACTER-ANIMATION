import os, sys, pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                        # thư mục chứa make_aligned.py

# Đường dẫn lấy theo thứ tự: tham số dòng lệnh -> biến môi trường -> thư mục chứa script
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    os.environ.get('AUTOVC_DATA_DIR', HERE))

STD_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].strip() else None  # STD_FACE_LANDMARKS.txt (không bắt buộc)
if STD_PATH is None:
    for cand in (DATA_DIR / 'STD_FACE_LANDMARKS.txt', HERE / 'STD_FACE_LANDMARKS.txt'):
        if cand.exists():
            STD_PATH = cand
            break

PICKLE_NAME = sys.argv[3] if len(sys.argv) > 3 else 'autovc_retrain_mel_train_fl.pickle'
PICKLE_PATH = DATA_DIR / PICKLE_NAME

if not PICKLE_PATH.exists():
    raise SystemExit(
        'Khong tim thay file pickle:\n  %s\n'
        'Hay truyen thu muc du lieu: python make_aligned.py "<thu_muc_chua_pickle>" '
        '[STD_FACE_LANDMARKS.txt] [ten_file.pickle]' % PICKLE_PATH)

STABLE = [27, 28, 29, 30, 36, 39, 42, 45]

def load(split):
    return pickle.load(open(DATA_DIR / ('autovc_retrain_mel_%s_fl.pickle' % split), 'rb'))

train_raw = load('train')
sample = np.concatenate([np.asarray(f, np.float64)[::10] for f, _ in train_raw])
REF = sample.mean(axis=0).reshape(68, 3)
Rs = REF[STABLE]; cR = Rs.mean(axis=0); Rc = Rs - cR

def align(P):                        # P: (T,68,3) -> (T,68,3)
    Xs = P[:, STABLE]
    cX = Xs.mean(axis=1, keepdims=True)
    Xc = Xs - cX
    H = np.einsum('tki,kj->tij', Xc, Rc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(P), 3)); D[:, 2] = d
    Rm = np.einsum('tij,tj,tjk->tik', U, D, Vt)
    s = (S * D).sum(axis=1) / (Xc ** 2).sum(axis=(1, 2))
    return np.einsum('tki,tij->tkj', P - cX, Rm) * s[:, None, None] + cR

# Tham chieu 2D cho Phase B va retarget: luu lai de dung chung ve sau
REF_ALIGNED = np.mean(np.concatenate([align(np.asarray(f, np.float64).reshape(-1, 68, 3)[::10])
                                      for f, _ in train_raw]), axis=0)[:, :2]
np.savez(DATA_DIR / 'align_ref.npz', ref3d=REF, stable=np.array(STABLE), ref2d_mean=REF_ALIGNED)

for split in ('train', 'val', 'test'):
    raw = train_raw if split == 'train' else load(split)
    out = []
    for fl, meta in raw:
        P = np.asarray(fl, np.float64).reshape(-1, 68, 3)
        A = align(P)[:, :, :2].reshape(-1, 136).astype(np.float32)
        assert np.isfinite(A).all(), 'NaN/Inf o video %s' % (meta[0],)
        out.append((A, meta))
    path = DATA_DIR / ('autovc_retrain_mel_%s_fl2d_aligned.pickle' % split)
    pickle.dump(out, open(path, 'wb'))
    print(split, len(out), 'video ->', path.name, '| frame mau:', out[0][0].shape)