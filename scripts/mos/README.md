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
| S4 | Wav2Lip | made separately from `wav2lip_inputs\` (see `HUONG_DAN_WAV2LIP.txt` there) |
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

`run_nb3.py` holds the local paths (repo, Drive root); edit them for another machine. ffmpeg is taken from `FFMPEG`,
then `PATH`, then `bin\ffmpeg.exe` next to the scripts. Python 3.11 with numpy, scipy, opencv, librosa, torch (CPU),
gradio and resvg-py.
