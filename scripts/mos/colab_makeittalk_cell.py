# ==== MOS - S3: chay MakeItTalk GOC (Zhou et al. 2020) cho 12 cau da chon -> landmark .npy ====================
# Dan nguyen cell nay vao 1 notebook Colab moi (GPU T4 hoac CPU deu duoc) roi chay. KHONG can notebook nao khac.
# - Chi DOC: audio.tar MEAD, checkpoint MakeItTalk, notebook 5 (lay code MakeItTalkTeacher da dung o benchmark).
# - Chi GHI vao thu muc MOI: Data_student/student_a2l/phaseH/mos_makeittalk/ (file da co thi bo qua, khong ghi de).
# LAN 2 (dung y benchmark notebook 5): face_id = mat trung tinh cua speaker; ghi vao mos_makeittalk/bench_face/.
# Moi cau -> <ten>_amp1.npy (bang ket qua dung he so moi x1), <ten>_amp2.npy, *_raw.npy; 62.5 khung/giay.
import os, sys, glob, json, ast, subprocess, tarfile, time
from pathlib import Path
import numpy as np

if 'google.colab' in sys.modules and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')


def first(pats, need):
    for p in pats:
        for c in sorted(glob.glob(p)):
            if os.path.exists(os.path.join(c, need)):
                return Path(c)
    raise FileNotFoundError('Khong tim thay %s trong %s' % (need, pats))


DRIVE = first(['/content/drive/MyDrive/DAT301m/Data_student', '/content/drive/.shortcut-targets-by-id/*/DAT301m*/Data_student',
               '/content/drive/MyDrive/*/Data_student'], 'student_a2l')
MEAD = first(['/content/drive/.shortcut-targets-by-id/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j/MEAD',
              '/content/drive/.shortcut-targets-by-id/*/MEAD', '/content/drive/MyDrive/MEAD', '/content/drive/MyDrive/*/MEAD'],
             'M032/audio.tar')
CKPT = DRIVE / 'student_a2l' / 'makeittalk_ckpt'
OUT = DRIVE / 'student_a2l' / 'phaseH' / 'mos_makeittalk'
OUT.mkdir(parents=True, exist_ok=True)
print('Data_student:', DRIVE, '\nMEAD        :', MEAD, '\nCheckpoint  :', CKPT, '\nGhi vao     :', OUT)

# 12 cau (C:\hien\mos\audio_list.csv): (ten, speaker, file trong audio.tar)
CLIPS = [('M032_neutral_L1_026', 'M032', 'audio/neutral/level_1/026.m4a'),
         ('M032_happy_L3_020', 'M032', 'audio/happy/level_3/020.m4a'),
         ('M032_angry_L3_006', 'M032', 'audio/angry/level_3/006.m4a'),
         ('M037_neutral_L1_021', 'M037', 'audio/neutral/level_1/021.m4a'),
         ('M037_happy_L3_007', 'M037', 'audio/happy/level_3/007.m4a'),
         ('M032_angry_L3_010', 'M032', 'audio/angry/level_3/010.m4a'),
         ('W018_neutral_L1_031', 'W018', 'audio/neutral/level_1/031.m4a'),
         ('W018_happy_L3_016', 'W018', 'audio/happy/level_3/016.m4a'),
         ('W018_angry_L3_019', 'W018', 'audio/angry/level_3/019.m4a'),
         ('W019_neutral_L1_005', 'W019', 'audio/neutral/level_1/005.m4a'),
         ('W019_happy_L3_009', 'W019', 'audio/happy/level_3/009.m4a'),
         ('W019_angry_L3_013', 'W019', 'audio/angry/level_3/013.m4a')]

# --- code + repo can thiet (giong notebook 5_phaseE_benchmark)
ELSA = Path('/content/MakeItTalk')                          # repo ELSA (ten thu muc giong cac notebook khac)
if not (ELSA / 'notebooks' / '5_phaseE_benchmark.ipynb').exists():
    ELSA = Path('/content/elsa_repo')
    if not ELSA.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hien087585/EXPRESSIVE-2D-CHARACTER-ANIMATION.git',
                        str(ELSA)], check=True)
MIT_DIR = Path('/content/MakeItTalk_orig')                  # repo goc MakeItTalk
if not (MIT_DIR / 'src').exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/yzhou359/MakeItTalk.git', str(MIT_DIR)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'setuptools<81', 'pysptk', 'resemblyzer', 'librosa'], check=False)
nb5 = json.loads((ELSA / 'notebooks' / '5_phaseE_benchmark.ipynb').read_text(encoding='utf-8'))
src = next(''.join(c['source']) for c in nb5['cells'] if ''.join(c['source']).startswith('MIT_TEACHER_SOURCE'))
LIB = Path('/content/mos_lib'); LIB.mkdir(exist_ok=True)
(LIB / 'mit_teacher.py').write_text(ast.literal_eval(src.split('\n', 1)[0][len('MIT_TEACHER_SOURCE = '):]))
sys.path.insert(0, str(LIB))
import torch, mit_teacher
from scipy.io import wavfile

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T = mit_teacher.MakeItTalkTeacher(MIT_DIR, CKPT, DEV)
READY = DRIVE / 'Data_emotion_mead_task_b' / 'phaseB_ready'
OUT2 = OUT / 'bench_face'                                   # lan 2: dung y cach benchmark notebook 5 (khong dung file cu)
OUT2.mkdir(exist_ok=True)
std2 = T.std_face[:, :2]
info = dict(model='MakeItTalk goc (Zhou et al. 2020), checkpoint ' + str(CKPT), fps=62.5, window=18, lag_best=12,
            note=('giong mit_raw() notebook 5: face_id = mat trung tinh MEAD cua speaker (phaseB_ready/<sp>.npz ref) khop '
                  'dong dang vao mat chuan MakeItTalk; <ten>_amp1.npy / <ten>_amp2.npy = (N, 136) do dich so voi ref, khong '
                  'gian MEAD (nhu bang ket qua); *_raw.npy = (N, 68, 3) dau ra goc. Khung j <-> au[j:j+18]; benchmark: '
                  'khung GT k <-> khung MakeItTalk k + 8 - 12'),
            device=DEV, clips={})
t0 = time.time()
for name, spk, member in CLIPS:
    ref = np.load(READY / (spk + '.npz'))['ref'].reshape(68, 2)
    s_, R_, t_ = mit_teacher.similarity_fit(ref, std2)        # MEAD -> MakeItTalk (nhu benchmark)
    face = T.std_face.copy(); face[:, :2] = s_ * ref @ R_ + t_
    tmp = Path('/tmp/mos_%s' % name); tmp.mkdir(exist_ok=True)
    with tarfile.open(MEAD / spk / 'audio.tar') as tf:
        (tmp / 'a.m4a').write_bytes(tf.extractfile(member).read())
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tmp / 'a.m4a'), '-ar', '16000', '-ac', '1', str(tmp / 'a.wav')], check=True)
    sr, y = wavfile.read(tmp / 'a.wav'); y = y.astype(np.float32) / 32768.0
    au = emb = None
    for amp in (1.0, 2.0):
        fa, fr = OUT2 / ('%s_amp%d.npy' % (name, amp)), OUT2 / ('%s_amp%d_raw.npy' % (name, amp))
        if fa.exists() and fr.exists():
            continue                                          # da co -> khong ghi de
        if au is None:
            au, emb = T.autovc_au(y), T.spk_emb(y)
        fl = T.predict(y, face, amp_lip=amp, au=au, emb=emb)
        xy = ((fl[:, :, :2] - t_) @ R_.T) / s_                  # MakeItTalk -> MEAD (nhu benchmark)
        np.save(fa, (xy - ref[None]).reshape(len(xy), 136).astype(np.float32))
        np.save(fr, fl.astype(np.float32))
    p = np.load(OUT2 / (name + '_amp1.npy'))
    info['clips'][name] = dict(source='%s/audio.tar:%s' % (spk, member), audio_s=round(len(y) / 16000, 3),
                               frames=int(len(p)), landmark_s=round(len(p) / 62.5, 3))
    print('%-22s audio %.2f s | landmark %s = %.2f s' % (name, len(y) / 16000, p.shape, len(p) / 62.5))
info_f = OUT2 / 'mos_makeittalk_info.json'
if not info_f.exists():
    info_f.write_text(json.dumps(info, indent=1, ensure_ascii=False))
print('\nXong %.0f s. File trong %s:' % (time.time() - t0, OUT2))
for f in sorted(OUT2.iterdir()):
    print('  %-34s %8.1f KB' % (f.name, f.stat().st_size / 1024))
