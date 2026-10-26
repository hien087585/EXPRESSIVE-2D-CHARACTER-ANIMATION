# MOS survey video set

Scripts that build the blind video set for the MOS listening/viewing test (raters score 1–5).
They run notebook `notebooks/3_phaseC_demo_render.ipynb` locally (CPU) through `run_nb3.py`, read the shared
Drive data **read-only**, and write everything to `C:\hien\mos\`. Videos and the answer key are not stored in the repo.

## Systems

| Code | System | How it is made |
| --- | --- | --- |
| S1 | ELSA, full | `MOUTH_REDRAW` + `BROW_REDRAW` (eyes redrawn too), emotion from SpEx-KD (`phaseG_final.pth`) on the clip's emotion2vec+ features, demo chain + mouth gain 1.85 |
| S2 | ELSA, warp only | same emotion and gain as S1, all redraw flags off |
| S3 | MakeItTalk landmarks + same redraw renderer, neutral | `_amp1` output (lip gain ×1, the "MakeItTalk goc (moi x1)" row used in the README), benchmark face_id and lag (12 frames), head pose removed per frame, clip mean removed, then drawn by the **same renderer as S1** (`MOUTH_REDRAW` + `BROW_REDRAW` + blinks, same image, 11 points, same `MOUTH_REDRAW_OPEN_SCALE` and local-minimum normalisation). MakeItTalk has no emotion, so the emotion timeline is neutral for the whole sentence: no mouth gain 1.85, no static expression, no smiling or angry eyelids; brows move only with the MakeItTalk landmarks; blinks use the same seed schedule as S1 |
| S4 | Wav2Lip | made separately from `wav2lip_inputs\` (see `HUONG_DAN_WAV2LIP.txt` there) by assistant A ([`wav2lip_gan.pth`](https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth), fixed `--box`, details in `_key\wav2lip_info.txt`). **S4 = Wav2Lip, encoded from A's MJPEG intermediates with the same ffmpeg settings as S1–S3; no other processing** (see below) |
| ATT | attention check | an S1 video with the audio delayed by 1 s (real silence inserted) |

S1 vs S3 compares the motion source (ELSA vs MakeItTalk) with the renderer held fixed; S1 vs S2 is the ablation of
the redraw renderer (S2 = ELSA with warp only). An earlier S3 (MakeItTalk through the S2 warp) is kept outside the test in
`_key\raw\s3_warp_old\` and is not used.

All videos: 512×512, 25 fps, h264 + AAC, no text, same length as the audio (±0.05 s).

## Inputs

- Characters: `make_chars.py` builds `boy.svg` and `girl2.svg` from the self-drawn `girl.svg` (same face layout),
  renders 1024 px PNGs and writes the 11 click points (eyes, nose, mouth corners, upper/lower lip, 4 brow ends).
- Audio: `select_audio.py` picks 12 MEAD sentences from the 4 test speakers (M032, M037, W018, W019):
  4 neutral + 4 happy (level 3) + 4 angry (level 3), 4–8 s, all sentence numbers different, and only clips that have
  precomputed emotion2vec+ features. `e2v_map.py` maps audio files to feature names the same way as notebook 6
  (by order inside each emotion/level group). M037 has no angry features at levels 2–3, so its angry slot uses
  another speaker's level-3 clip.
- Characters are assigned to sentences with a Latin-square rule, so each character gets 4 sentences and all 3 emotions.

## Steps

1. `python make_chars.py`
2. `python select_audio.py` → `audio\` + `audio_list.csv`
3. `python run_nb3.py <notebook> C:\hien\mos\_key\work mos <img> <points> <wav> 3`
   → S1, S2, attention checks, `wav2lip_inputs\`, blind names `videos\v_XXXX.mp4`, `_key\key.csv` (codes for S3/S4 are
   reserved in advance), `play_order.csv` (no two videos of the same sentence back to back). Seeds are fixed.
4. Run `colab_makeittalk_cell.py` on Colab → `student_a2l/phaseH/mos_makeittalk/bench_face/`; copy it to
   `C:\hien\mos\makeittalk\bench_face\`.
5. `python run_nb3.py <notebook> C:\hien\mos\_key\work_s3 s3 <img> <points> <wav> 3 C:\hien\mos\makeittalk\bench_face C:\hien\mos\_key\raw`
   then `python mos_merge_s3.py` (copies S3 into its reserved codes, writes the S3 note in `key.csv`, runs the final checks).
6. `mos_check.py` checks play order, attention-check delay and text; `s3_grid.py` writes a side-by-side S1 | S2 | S3
   frame grid (`_check\s3_vs_s1_grid_v2.png`).
7. S4: A's 12 MJPEG AVI intermediates (frames before A's own H.264 encode) are kept unchanged in `_key\raw\s4_avi\`.
   `python mos_s4_from_avi.py` checks them (12 files, names, colour vs the PNG, no edge border, mouth moves with the audio),
   encodes them with the S1–S3 settings (`scale=512:512:flags=area`, 25 fps, libx264 veryfast crf 18 yuv420p, AAC 128k),
   muxes the original wav from `wav2lip_inputs\` (the AVIs have no audio), copies the result into the reserved codes,
   updates `key.csv`, measures colour error against the PNG and runs the final checks; `s4_grid.py` writes
   `_check\s4_check_grid.png` (S1 | S3 | S4, plus A's AVI vs S4 edge rows).
   **S4 = Wav2Lip, encoded from A's MJPEG intermediates with the same ffmpeg settings as S1–S3; no other processing.**
   - The MJPEG is BT.601 full range; it is converted to BT.709 limited range in the same `scale` step and tagged BT.709
     (`setparams`, because ffmpeg 9 writes the frame's colour properties, not the encoder options). `accurate_rnd+full_chroma_int`
     is added to that step: without it the matrix/range conversion darkens everything by ~1.5 levels. S1–S3 carry no colour
     tags (implicit BT.601 limited); colour error is measured by decoding each file per its own tags.
   - Every frame is kept (frame count = ceil(wav × 25)), no stretching, no `-shortest`; video is < 1 frame longer than audio.
   - Colour error vs the PNG (outside the Wav2Lip box): S1 2.85, S2 3.14, S3 2.97, A's AVI 0.72, S4 1.03.
   - The earlier S4 (`mos_merge_s4.py`, fixes on A's H.264 output, below) is moved to `_key\raw\s4_fixed_old\` and not used.

   Earlier S4, superseded: A's MP4s are kept unchanged in `_key\raw\s4_rar\`. `mos_merge_s4.py` fixed only A's two encoder
   faults (colour-range tag, 2-px border):
   - A's stream is BT.601 full-range data with no colour tags (players read it as limited range: ~16 % more contrast and
     saturation, highlights clipped). It is converted in YUV (`colorspace` filter, no chroma resampling, primaries kept)
     to BT.709 limited range and tagged BT.709.
   - The green 2-px border is a chroma fault only: A's luma rows are correct, the first and last chroma rows (luma rows 0–1
     and 510–511 in 4:2:0) are wrong, so those two chroma rows are replaced by the adjacent chroma row. Whole RGB rows are not
     replaced because the character PNG has a real dark line in its bottom two rows.
   - Re-encoded with the same x264 settings as S1–S3; audio stream copied; frame count unchanged. Colour error vs the PNG
     (outside the Wav2Lip box): S1 2.85, S2 3.14, S3 2.97, A's original 12.94, S4 after the fix 1.02.

`run_nb3.py` holds the local paths (repo, Drive root); edit them for another machine. ffmpeg is taken from `FFMPEG`,
then `PATH`, then `bin\ffmpeg.exe` next to the scripts. Python 3.11 with numpy, scipy, opencv, librosa, torch (CPU),
gradio and resvg-py.
