# ==== BASELINE SadTalker + Wav2Lip tren MEAD (198 clip: 158 test + 40 val de do lag) -> mp4 tren Drive ===========
# Dan nguyen cell nay vao 1 notebook Colab MOI (GPU T4) roi chay. Chay lai bao nhieu lan cung duoc: clip da co mp4 thi bo qua.
# - Chi DOC: baseline_inputs.zip (tao bang scripts/baselines/prepare_baseline_inputs.py, ban tu tai len Drive).
# - Chi GHI vao thu muc MOI OUT_ROOT: sadtalker/<clip>.mp4, wav2lip/<clip>.mp4, *_timing.csv, params.json, env.json.
# SadTalker: repo OpenTalker/SadTalker + checkpoint chinh thuc, --still --preprocess full, KHONG --enhancer, size 256.
# Wav2Lip  : repo Rudrabha/Wav2Lip, wav2lip_gan.pth DUNG file da dung cho MOS (kiem SHA-256), anh tinh (--static), fps 25.
import os, sys, re, csv, json, glob, time, random, shutil, hashlib, zipfile, subprocess, functools
from pathlib import Path

# ------------------------------------------------------------------ DUONG DAN (sua o day neu can)
ZIP_PATH = '/content/drive/MyDrive/DAT301m/Data_student/baselines/baseline_inputs.zip'
OUT_ROOT = '/content/drive/MyDrive/DAT301m/Data_student/baselines/out'
WAV2LIP_SHA256 = 'CA9AB7B7B812C0E80A6E70A5977C545A1E8A365A6C49D5E533023C034D7AC3D8'
WAV2LIP_URL = 'https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth'
S3FD_URL = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
RUN = ['sadtalker', 'wav2lip']          # bo bot 1 ten neu chi muon chay 1 phuong phap
SEED = 0

if 'google.colab' in sys.modules and not os.path.isdir('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')
OUT = Path(OUT_ROOT); OUT.mkdir(parents=True, exist_ok=True)
IN = Path('/content/baseline_inputs')
if not (IN / 'list.csv').exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(IN)
CLIPS = list(csv.DictReader(open(IN / 'list.csv', encoding='utf-8')))
print('Dau vao: %d clip (%d test, %d val), %d anh' % (len(CLIPS), sum(c['split'] == 'test' for c in CLIPS),
                                                      sum(c['split'] == 'val' for c in CLIPS), len(list((IN / 'images').glob('*.png')))))
for m in RUN:
    (OUT / m).mkdir(exist_ok=True)
    done = sum((OUT / m / (c['clip_id'] + '.mp4')).exists() for c in CLIPS)
    print('  %-9s da xong %3d / %d | con lai %d' % (m, done, len(CLIPS), len(CLIPS) - done))


def sh(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest().upper()


def git_head(d):
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=d, capture_output=True, text=True).stdout.strip()


def patch_file(p, old, new, regex=False, must=True):
    s = Path(p).read_text(encoding='utf-8')
    s2 = re.sub(old, new, s) if regex else s.replace(old, new)
    if s2 == s and must and new not in s:
        print('  [patch] KHONG thay %r trong %s' % (old, p))
    Path(p).write_text(s2, encoding='utf-8')


def patch_numpy_names(root):
    # numpy >= 1.24 / 2.x da bo np.float, np.int, np.bool, np.VisibleDeprecationWarning
    for p in glob.glob(str(root) + '/**/*.py', recursive=True):
        s = Path(p).read_text(encoding='utf-8', errors='ignore')
        s2 = re.sub(r'np\.float(?![\w])', 'np.float64', s)
        s2 = re.sub(r'np\.int(?![\w])', 'np.int64', s2)
        s2 = re.sub(r'np\.bool(?![\w])', 'bool', s2)
        s2 = s2.replace('np.VisibleDeprecationWarning', 'Warning')
        if s2 != s:
            Path(p).write_text(s2, encoding='utf-8')


def append_timing(name, row):
    f = OUT / ('%s_timing.csv' % name); new = not f.exists()
    with open(f, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def count(m):
    return int(sum(p.numel() for p in m.parameters()))


def seed_all(s):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ cai dat + patch moi truong Colab hien tai
sh('pip install -q face_alignment imageio imageio-ffmpeg librosa resampy pydub kornia yacs safetensors facexlib av '
   'scikit-image joblib "setuptools<81"')
subprocess.run('pip install -q basicsr gfpgan', shell=True)          # chi de import (khong dung --enhancer); loi thi dung stub
import numpy as np, torch, torchvision, librosa
_tl = torch.load
torch.load = functools.partial(_tl, weights_only=False) if 'weights_only' in _tl.__code__.co_varnames else _tl   # torch >= 2.6
ft = Path(torchvision.transforms.__file__).parent / 'functional_tensor.py'      # basicsr can module da bi xoa o torchvision >= 0.17
if not ft.exists():
    ft.write_text('from torchvision.transforms._functional_tensor import *\n'
                  'from torchvision.transforms.functional import rgb_to_grayscale\n')
try:
    import gfpgan  # noqa: F401
except Exception as e:                                                          # khong dung enhancer -> stub cho import
    import types
    print('  gfpgan khong import duoc (%s) -> stub (khong dung --enhancer)' % type(e).__name__)
    stub = types.ModuleType('gfpgan'); stub.GFPGANer = None; sys.modules['gfpgan'] = stub
ENV = dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__,
           librosa=librosa.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('Moi truong:', ENV)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PARAMS = json.loads((OUT / 'params.json').read_text()) if (OUT / 'params.json').exists() else {}

# ================================================================== SadTalker
if 'sadtalker' in RUN:
    ST = Path('/content/SadTalker')
    if not (ST / 'inference.py').exists():
        sh('git clone -q https://github.com/OpenTalker/SadTalker.git %s' % ST)
        patch_numpy_names(ST / 'src')
        patch_file(ST / 'src/face3d/util/preprocess.py', 'trans_params = np.array([w0, h0, s, t[0], t[1]])',
                   'trans_params = np.array([w0, h0, s, float(np.ravel(t)[0]), float(np.ravel(t)[1])])')
        patch_file(ST / 'src/utils/audio.py', 'librosa.filters.mel(hp.sample_rate, hp.n_fft,',
                   'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,')
    if not (ST / 'checkpoints' / 'SadTalker_V0.0.2_256.safetensors').exists():
        sh('bash scripts/download_models.sh', cwd=ST)                          # checkpoint chinh thuc (GitHub release)
    ENV['sadtalker_commit'] = git_head(ST)
    ENV['sadtalker_ckpt_sha256'] = {p.name: sha256(p) for p in sorted((ST / 'checkpoints').glob('*')) if p.is_file()}
    os.chdir(ST); sys.path.insert(0, str(ST))
    from src.utils.preprocess import CropAndExtract
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    from src.utils.init_path import init_path
    # = cac gia tri mac dinh cua inference.py, tru --still --preprocess full (khong enhancer)
    A = dict(size=256, preprocess='full', still=True, pose_style=0, batch_size=2, expression_scale=1.0, old_version=False)
    paths = init_path(str(ST / 'checkpoints'), str(ST / 'src' / 'config'), A['size'], A['old_version'], A['preprocess'])
    t0 = time.time()
    pre, a2c, anim = CropAndExtract(paths, DEV), Audio2Coeff(paths, DEV), AnimateFromCoeff(paths, DEV)
    print('SadTalker: nap mo hinh %.0f s' % (time.time() - t0))
    if 'sadtalker' not in PARAMS:                                              # dem tren mo hinh da nap tu checkpoint
        nets = {'net_recon (3DMM / coeff extractor, CropAndExtract)': pre.net_recon,
                'audio2exp (audio -> expression)': a2c.audio2exp_model, 'audio2pose (audio -> pose, CVAE)': a2c.audio2pose_model,
                'mapping (3DMM -> motion)': anim.mapping, 'kp_extractor (canonical keypoints)': anim.kp_extractor,
                'he_estimator (head pose / expression)': anim.he_estimator, 'generator (face renderer)': anim.generator}
        for nm, attr in (('face detector (facexlib RetinaFace, preprocess)', 'det_net'),
                         ('landmark detector (FAN, preprocess)', 'detector')):
            m = getattr(getattr(pre.propress, 'predictor', None), attr, None)
            if isinstance(m, torch.nn.Module):
                nets[nm] = m
        PARAMS['sadtalker'] = {k: count(v) for k, v in nets.items()}
    cache = {}
    for c in CLIPS:
        dst = OUT / 'sadtalker' / (c['clip_id'] + '.mp4')
        if dst.exists():
            continue
        sp, wav = c['speaker'], str(IN / 'wav' / (c['clip_id'] + '.wav'))
        try:
            if sp not in cache:                                                # tien xu ly anh: 1 lan / speaker (tat dinh)
                d = Path('/content/st_pre') / sp; d.mkdir(parents=True, exist_ok=True)
                t1 = time.time()
                cache[sp] = pre.generate(str(IN / 'images' / (sp + '.png')), str(d), A['preprocess'], source_image_flag=True,
                                         pic_size=A['size']) + (time.time() - t1,)
                print('  SadTalker tien xu ly anh %s: %.1f s' % (sp, cache[sp][3]))
            first_coeff, crop_pic, crop_info, t_pre = cache[sp]
            seed_all(SEED + int(c['clip_index']))                             # chop mat ngau nhien -> co dinh theo clip
            save = Path('/content/st_out') / c['clip_id']; shutil.rmtree(save, ignore_errors=True); save.mkdir(parents=True)
            if DEV == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            batch = get_data(first_coeff, wav, DEV, None, still=A['still'])
            coeff = a2c.generate(batch, str(save), A['pose_style'], None)
            data = get_facerender_data(coeff, crop_pic, first_coeff, wav, A['batch_size'], None, None, None,
                                       expression_scale=A['expression_scale'], still_mode=A['still'],
                                       preprocess=A['preprocess'], size=A['size'])
            res = anim.generate(data, str(save), str(IN / 'images' / (sp + '.png')), crop_info, enhancer=None,
                                background_enhancer=None, preprocess=A['preprocess'], img_size=A['size'])
            if DEV == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t1
            shutil.copy(res, str(dst) + '.part'); os.replace(str(dst) + '.part', dst)
            append_timing('sadtalker', dict(clip_id=c['clip_id'], audio_s=c['audio_s'], seconds=round(dt, 3),
                                            s_per_audio_s=round(dt / float(c['audio_s']), 4),
                                            image_preprocess_s=round(t_pre, 2), device=ENV['gpu']))
            print('  SadTalker %-30s %.1f s (%.2f s / s audio)' % (c['clip_id'], dt, dt / float(c['audio_s'])))
        except Exception as e:
            import traceback; traceback.print_exc()
            print('  !! SadTalker LOI %s: %s' % (c['clip_id'], e))
    os.chdir('/content'); sys.path.remove(str(ST))
    for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:  # Wav2Lip cung khong dung 'src'; don cho sach
        del sys.modules[k]

# ================================================================== Wav2Lip
if 'wav2lip' in RUN:
    WL = Path('/content/Wav2Lip')
    if not (WL / 'inference.py').exists():
        sh('git clone -q https://github.com/Rudrabha/Wav2Lip.git %s' % WL)
        patch_numpy_names(WL)
        patch_file(WL / 'audio.py', 'librosa.filters.mel(hp.sample_rate, hp.n_fft,', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,')
    ck = WL / 'checkpoints' / 'wav2lip_gan.pth'
    if not ck.exists():
        sh('wget -q -O %s %s' % (ck, WAV2LIP_URL))
    h = sha256(ck)
    assert h == WAV2LIP_SHA256, 'wav2lip_gan.pth SAI checkpoint: %s (can %s)' % (h, WAV2LIP_SHA256)
    print('Wav2Lip: wav2lip_gan.pth SHA-256 khop', h)
    s3 = WL / 'face_detection' / 'detection' / 'sfd' / 's3fd.pth'
    if not s3.exists():
        sh('wget -q -O %s %s' % (s3, S3FD_URL))
    ENV['wav2lip_commit'] = git_head(WL)
    ENV['wav2lip_ckpt_sha256'] = {'wav2lip_gan.pth': h, 's3fd.pth': sha256(s3)}
    os.chdir(WL); sys.path.insert(0, str(WL)); (WL / 'temp').mkdir(exist_ok=True)
    sys.argv = ['inference.py', '--checkpoint_path', str(ck), '--face', str(IN / 'images' / 'M032.png'),
                '--audio', str(IN / 'wav' / (CLIPS[0]['clip_id'] + '.wav')), '--static', 'True', '--fps', '25']
    import inference as W2L                                                   # = inference.py, tham so mac dinh + --static --fps 25
    W2L.args.static = True
    _model = {}
    _load = W2L.load_model
    W2L.load_model = lambda p: _model.setdefault(p, _load(p))                # nap 1 lan (thoi gian do khong tinh nap mo hinh)
    t0 = time.time(); g = W2L.load_model(str(ck)); print('Wav2Lip: nap mo hinh %.0f s' % (time.time() - t0))
    if 'wav2lip' not in PARAMS:
        import face_detection
        fd = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=DEV)
        PARAMS['wav2lip'] = {'generator (Wav2Lip, wav2lip_gan.pth)': count(g),
                             'face detector (S3FD, s3fd.pth)': count(fd.face_detector.face_detector)}
        del fd
    for c in CLIPS:
        dst = OUT / 'wav2lip' / (c['clip_id'] + '.mp4')
        if dst.exists():
            continue
        W2L.args.face = str(IN / 'images' / (c['speaker'] + '.png'))
        W2L.args.audio = str(IN / 'wav' / (c['clip_id'] + '.wav'))
        W2L.args.outfile = '/content/w2l_out/%s.mp4' % c['clip_id']; Path('/content/w2l_out').mkdir(exist_ok=True)
        try:
            seed_all(SEED + int(c['clip_index']))
            if DEV == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time(); W2L.main()
            if DEV == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t1
            shutil.copy(W2L.args.outfile, str(dst) + '.part'); os.replace(str(dst) + '.part', dst)
            append_timing('wav2lip', dict(clip_id=c['clip_id'], audio_s=c['audio_s'], seconds=round(dt, 3),
                                          s_per_audio_s=round(dt / float(c['audio_s']), 4), device=ENV['gpu']))
            print('  Wav2Lip   %-30s %.1f s (%.2f s / s audio)' % (c['clip_id'], dt, dt / float(c['audio_s'])))
        except Exception as e:
            import traceback; traceback.print_exc()
            print('  !! Wav2Lip LOI %s: %s' % (c['clip_id'], e))
    os.chdir('/content')

# ------------------------------------------------------------------ tong ket (moi lan chay)
(OUT / 'params.json').write_text(json.dumps(PARAMS, indent=1))
envs = json.loads((OUT / 'env.json').read_text()) if (OUT / 'env.json').exists() else []
envs.append(dict(ENV, run_at=time.strftime('%Y-%m-%d %H:%M:%S')))
(OUT / 'env.json').write_text(json.dumps(envs, indent=1))
print('\nSo tham so (dem tren mo hinh da nap tu checkpoint):')
for m, d in PARAMS.items():
    for k, v in d.items():
        print('  %-9s %-52s %8.2fM' % (m, k, v / 1e6))
    print('  %-9s %-52s %8.2fM' % (m, 'TONG (moi mang tren, deu dung khi chay)', sum(d.values()) / 1e6))
for m in RUN:
    done = sum((OUT / m / (c['clip_id'] + '.mp4')).exists() for c in CLIPS)
    f = OUT / ('%s_timing.csv' % m)
    if f.exists():
        r = list(csv.DictReader(open(f)))
        x = np.array([float(i['s_per_audio_s']) for i in r])
        print('%-9s xong %d / %d | thoi gian / 1 s audio: TB %.3f s, trung vi %.3f s (%d clip; khong tinh nap mo hinh%s)'
              % (m, done, len(CLIPS), x.mean(), np.median(x), len(x), '; SadTalker: tien xu ly anh tinh rieng' if m == 'sadtalker' else ''))
    else:
        print('%-9s xong %d / %d' % (m, done, len(CLIPS)))
