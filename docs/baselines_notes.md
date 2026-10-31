# Baseline SadTalker + Wav2Lip trên MEAD: ghi chú kết quả (2026-10-29)

Lần chạy Colab: Tesla T4, python 3.13.16, torch 2.11.0+cu130, numpy 2.1.3, librosa 0.11.0 (env.json, run_at 2026-10-29 02:52:40).
- SadTalker commit cd4c0465, checkpoint SadTalker_V0.0.2_256.safetensors (SHA-256 C211F5D6…), `--still --preprocess full`, size 256, không dùng enhancer.
- Wav2Lip commit bac9a81e, wav2lip_gan.pth SHA-256 CA9AB7B7… (khớp file đã dùng cho MOS), s3fd.pth 619A3168…, `--static`, 25 fps.
- 198 / 198 clip mỗi phương pháp (158 test + 40 val). Mỗi clip có đúng 1 dòng trong *_timing.csv, không trùng, không thiếu.

## 1. Định dạng mp4 SadTalker
Đã ffprobe toàn bộ mp4 trên Drive: SadTalker 198/198 và Wav2Lip 198/198 đều là **1920x1080, 25 fps**.
Như vậy file SadTalker là bản `*_full.mp4` (paste_pic/seamlessClone dán lại khung gốc), đổi tên thành `<clip>.mp4`; không phải bản crop 256.
Toạ độ khớp với ảnh đầu vào 1920x1080 nên dùng được cho Procrustes (một lần fit từ ảnh đầu vào → ref).

## 2. Trích landmark → baseline_landmarks.npz
- mp4 copy từ Drive (G:, chỉ đọc) về `<LOCAL_OUT>/baselines/runs/{sadtalker,wav2lip}/` (+ params.json, env.json, *_timing.csv).
- Môi trường: `<LOCAL_OUT>/baselines/venv_mp` (requirements_baselines.txt: mediapipe 1.2.0, opencv 5.0.0, numpy 2.4.6,
  đúng bộ đã PASS ở `check`: mean 0.00001, max 0.00012).
- Lệnh: `eval_baselines.py extract --drive <DRIVE>/Data_student --inputs baseline_inputs --runs runs` → log `extract.log`.
- Kết quả: `<LOCAL_OUT>/baselines/baseline_landmarks.npz` (50.1 MB, 396 mảng).
  **SadTalker 198 / 198 ok, Wav2Lip 198 / 198 ok, 0 clip lỗi**: mọi khung đều dò được mặt, không có NaN, không có cảnh báo tỉ lệ khung
  (tất cả 1920x1080, fps đọc được 25.0).
- Procrustes ảnh đầu vào → ref (1 fit / speaker): s = 0.964–1.018, xoay ≤ 1.3°, sai số dư TB 0.009–0.028.
- Độ dài: 32 clip có audio > 6 s bị cắt còn 150 khung (= 6.0 s @25 fps) do `MAX_FRAMES = 150` của notebook 0b. Đây là cùng trần
  đã áp cho landmark MEAD thật (150 khung @30 fps = 5 s), nên 6 s vẫn phủ hết phần GT để so sánh. Ngoài các clip đó, video
  ngắn hơn audio rất ít: SadTalker trung vị 0.02 s; Wav2Lip ~0.1–0.125 s (Wav2Lip bỏ vài khung cuối do chia cửa sổ mel).

## 3. Số tham số (params.json, đếm trên mô hình nạp từ checkpoint chính thức)

| SadTalker | Tham số |
| :--- | ---: |
| net_recon (3DMM / coeff extractor) | 24.03M |
| audio2exp | 2.85M |
| audio2pose (CVAE) | 11.25M |
| mapping (3DMM → motion) | 10.22M |
| kp_extractor | 41.94M |
| he_estimator | 30.25M |
| generator (face renderer) | 100.99M |
| face detector (facexlib RetinaFace, preprocess) | 27.29M |
| landmark detector (FAN, preprocess) | 24.15M |
| **Tổng** | **272.98M** |

| Wav2Lip | Tham số |
| :--- | ---: |
| generator (wav2lip_gan.pth) | 36.30M |
| face detector (S3FD) | 22.46M |
| **Tổng** | **58.76M** |

## 4. Thời gian / 1 s audio (Colab T4, không tính thời gian nạp mô hình)

| | Trung vị | TB | Min – max | Trung vị test (158) | Trung vị val (40) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SadTalker (end-to-end, gồm cả seamlessClone) | **26.76 s** | 27.56 s | 23.75 – 37.94 | 26.26 s | 31.95 s |
| Wav2Lip | **1.85 s** | 2.63 s | 1.37 – 39.31 | 1.84 s | 1.86 s |

- SadTalker: tiền xử lý ảnh (CropAndExtract) tính riêng, 1 lần/speaker: 0.45–0.47 s (3.3 s cho M032 là lần đầu trong phiên, 3.24 s cho W019 sau khi phiên khởi động lại).
  `seconds` trong sadtalker_timing.csv = audio2coeff + Face Renderer + paste_pic (seamlessClone 1920x1080) + ghép audio.
- Wav2Lip: max 39.31 s/s là clip đầu tiên của lần chạy (M032__front__angry__L1__002, khởi động CUDA/cuDNN); clip thứ hai lớn nhất chỉ 5.41 s/s. Clip ngắn có s/s cao hơn vì chi phí cố định (S3FD trên ảnh tĩnh).

### Tách Face Renderer và seamlessClone (SadTalker)
Nguồn: **log stdout của cell Colab** (thanh tqdm `Face Renderer:` và `seamlessClone:`), chỉ **44 / 198 clip** có thanh
tiến trình trong log. Số do người chạy tính từ log và gửi lại (2026-10-29); mình không có file log gốc để kiểm lại.

| SadTalker (44 clip có log) | Trung vị / 1 s audio | TB / 1 s audio |
| :--- | ---: | ---: |
| Face Renderer | 13.66 s | 13.56 s |
| seamlessClone (dán lại khung 1920x1080) | 14.32 s | 17.29 s |
| Phần còn lại (audio2coeff, ghép audio, …) | ~0.95 s | |

- seamlessClone chiếm trung vị **50%** tổng thời gian mỗi clip.
- Lưu ý: trung vị từng phần không cộng lại thành trung vị tổng (13.66 + 14.32 + 0.95 = 28.93 s, còn trung vị tổng trên 198 clip
  là 26.76 s); ngoài ra 44 clip này là tập con nên không đại diện cho cả 198 clip.
- Khi báo: Face Renderer ~13.7 s / 1 s audio (chi phí của mô hình) và end-to-end 26.8 s / 1 s audio (gồm seamlessClone full-frame).

## 5. Kết quả cuối: metric trên 158 clip test (2026-10-31)

Nguồn: lần chạy Colab `colab_eval_baselines_cell.py`, timestamp **20261031_012330**. 5 file tải từ Drive (zip
`baselines_158-20261031T013238Z-1-001.zip`, SHA-256 F9ABC78E…), giải nén vào `<LOCAL_OUT>/baselines/baselines_158/`:
- `<LOCAL_OUT>/baselines/baselines_158/baselines_158_per_clip_20261031_012330.csv` (790 dòng = 158 clip × 5 hệ; SHA-256 87703670…)
- `<LOCAL_OUT>/baselines/baselines_158/baselines_158_summary_20261031_012330.csv` (A3EE99E2…)
- `<LOCAL_OUT>/baselines/baselines_158/baselines_158_sensitivity_20261031_012330.csv` (60ECC66A…)
- `<LOCAL_OUT>/baselines/baselines_158/baselines_158_wilcoxon_20261031_012330.csv` (5CAFA8E7…)
- `<LOCAL_OUT>/baselines/baselines_158/baselines_158_lags_20261031_012330.json` (DEDFA24F…; `_common_clips` 158, `_dropped` rỗng, `_table2_check` ok, mediapipe 1.2.0)

Lệnh (venv_mp): `eval_baselines.py report --runs <LOCAL_OUT>/baselines/runs --colab <LOCAL_OUT>/baselines/baselines_158`
→ log `<LOCAL_OUT>/baselines/report.log`, bảng `<REPO>/results/baselines_158.csv`.
- `--colab` phải trỏ vào thư mục con `baselines_158\` (script glob không đệ quy), không phải `<LOCAL_OUT>/baselines`.
- Trên console Windows (cp1252) cần `PYTHONIOENCODING=utf-8` (hoặc `python -X utf8`), nếu không sẽ lỗi `UnicodeEncodeError` ở ký tự ↓.

| Hệ | Dịch (khung @62.5) | LMD ↓ | M-LMD ↓ | Tương quan miệng ↑ | Biên độ (1) | Khép khi im ↓ | Emo-Acc ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MakeItTalk | +4 | 0.0613 [0.0587, 0.0640] | 0.0453 [0.0434, 0.0474] | 0.275 | 0.569 | 0.0934 | 0.013 |
| MakeItTalk x2 | +4 | 0.0630 [0.0606, 0.0653] | 0.0506 [0.0488, 0.0526] | 0.278 | 1.120 | 0.1098 | 0.007 |
| SadTalker | **−4 (= −64 ms)** | **0.0487** [0.0459, 0.0516] | 0.0314 [0.0300, 0.0331] | 0.370 | 0.696 | 0.0945 | 0.112 |
| Wav2Lip | 0 | n/a | **0.0304** [0.0290, 0.0320] | 0.447 | 0.748 | 0.0757 | n/a |
| ELSA (fully automatic) | 0 | **0.0343** [0.0319, 0.0367] | 0.0270 [0.0258, 0.0285] | 0.580 | 0.931 | 0.0492 | 0.934 |

(95% CI bootstrap 1000 lần trên clip; CI đủ cho mọi cột có trong `report.log`.)

**Đối chiếu với output Colab: khớp cả 4 số.**
- ELSA LMD 0.0343 (0.034326) ✔
- SadTalker LMD 0.0487 (0.048748) ✔
- Wav2Lip M-LMD 0.0304 (0.030387) ✔
- Lag SadTalker −64 ms ✔: shift −4 khung @62.5 fps = −4 × 16 ms; MakeItTalk-lag L = 4 (shift = L − 8), corr val 0.363 → 0.480 sau khi dịch.
  Wav2Lip và ELSA có lag tốt nhất ở shift 0 (corr val 0.506 và 0.613).

Wilcoxon (Holm): ELSA tốt hơn mọi baseline ở mọi metric, p (Holm) ≤ 4.3e-07 (Khép khi im, n = 50 clip có đoạn im lặng);
các metric còn lại n = 158, p (Holm) ≤ 1.7e-07. Sensitivity (fit theo lông mày/mũi/mắt) giữ nguyên thứ tự xếp hạng
(LMD: ELSA 0.0272 < SadTalker 0.0379 < MakeItTalk 0.0426). Tham số và thời gian chạy trong report khớp mục 3–4.
