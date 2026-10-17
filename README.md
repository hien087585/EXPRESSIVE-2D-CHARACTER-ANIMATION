# Expressive 2D Character Animation

**One 2D character image + one speech clip → a talking video with lip sync and facial expressions that follow the emotion in the voice.**

Course project for DAT301m. The system combines a small lip-sync model (3.29M parameters) with an emotion branch that learns expressions directly from speech, so no one has to label the emotion of each sentence.

![Pipeline](docs/pipeline.png)

## Research goals

Animating talking characters (lecture videos, virtual assistants, games, VTubers) takes a lot of manual work. Published models such as MakeItTalk produce lip motion but no emotion; adding emotion usually requires a manual label for every sentence, and the models are fairly heavy (almost 20 million parameters).

The project sets four goals:

1. **Accurate yet lightweight lip sync:** a model several times smaller than MakeItTalk that still syncs better and runs on a CPU.
2. **Emotion taken from the voice itself:** remove manual emotion labelling.
3. **Follow emotion changes in long audio:** detect when the speaker switches from neutral to angry or happy.
4. **Usable in practice:** any character image (anime, cartoon, human face) + any audio → video, with a demo interface.

Research questions:

- Can a small 1D-CNN replace heavy Transformer / LSTM models for audio-driven lip motion?
- Does separating the static expression (whole sentence) from mouth motion (per sound) help the model learn?
- Can a large pretrained speech-emotion model (emotion2vec+) be reused to generate facial expressions, instead of training from scratch on little data?

Scope: works at the level of 68 facial landmarks followed by image warping (no pixel-level generation); 3 emotions in the demo (neutral, angry, happy); no head motion; English speech by actors.

## Main results

Measured on 4 speakers never seen during training (MEAD dataset).

| | Our model (C4) | Original MakeItTalk |
| --- | --- | --- |
| Full-face landmark error (LMD, lower is better) | **0.0414** | 0.0651 |
| Mouth-opening correlation with the real face | **0.574** | 0.263 |
| Emotion recognisable from the generated face (Emo-Acc) | **0.832** | 0.136 |
| Parameters | **3.29M** | 19.29M |
| Time per 1 s of audio | **37 ms CPU · 5.2 ms GPU** | 27.7 ms GPU |

- Expressions learned from speech, no labels needed: **+18.6%** better than the "neutral face" baseline (previous approach, classify emotion then look up a table: +15.2%).
- Emotion tracking over time on audio with mixed emotions: **UAR 0.906**, **83%** of emotion changes detected within ±1 s, on par with emotion2vec+ (0.905).

| Input image and generated expressions |
| --- |
| ![Expressions](docs/expressions.png) |

<details>
<summary>More figures: comparison with MakeItTalk, emotion timeline, demo screens</summary>

![Comparison with MakeItTalk](docs/makeittalk_comparison.png)

![Emotion timeline](docs/emotion_timeline.png)

![Demo tab 1](docs/demo_tab1_emotion.png)

![Demo tab 2](docs/demo_tab2_video.png)

</details>

## How it works

The same audio goes through two branches:

1. **Lip sync (what is said):** audio → AutoVC mel spectrogram (80 bands, 62.5 frames/s) → student model **C4** (1D-CNN + FC) → jaw and mouth motion.
2. **Emotion (how it is said):** audio → **emotion2vec+** (pretrained, frozen, 2.5 s windows) → **EmoBranch** (0.5M) → neutral / angry / happy over time plus a static full-face expression. EmoBranch also feeds the emotion condition into C4.

Both branches are merged into 68 facial landmarks per frame, retargeted to the character's face proportions, limited in mouth opening to prevent distortion, used to warp the image, and written to an mp4 video with sound.

## Repository layout

```
notebooks/            Colab notebooks, numbered in running order
src/models/           model_audio2landmark.py: the StudentAudio2Landmark student model
src/dataset/utils/    Mean / std used to normalise the mel spectrogram
scripts/              make_aligned.py (align Obama landmarks), check_emotion_pkg.py (check MEAD packages)
examples_cartoon/     Sample character images and landmark files (from MakeItTalk)
facewarp/             Original MakeItTalk image-warping tool (not used by the current demo)
docs/                 Figures for this README
Data_examble/, Data_student/   Placeholder folders describing the data layout on Google Drive
```

## Running order

All notebooks run on Google Colab. Each notebook clones this repo to get the model, reads data and saves results to Google Drive for the next step.

| # | Notebook | What it does | Output (in `Data_student/student_a2l/`) |
| --- | --- | --- | --- |
| 0a | `0a_stageA_makeittalk_landmarks` | Run the original MakeItTalk to obtain audio-driven landmarks | |
| 0b | `0b_stageB_mead_landmarks` | Extract MEAD landmarks with MediaPipe FaceLandmarker (478 → 68 points), packaged per actor | `Data_emotion_mead_task_b/stage_b_mead_out/` |
| 1 | `1_phaseA_student` | Train the student model on Obama data | `phaseA/phaseA_student_best.pth` |
| 2a | `2a_phaseB_prepare_data` | Convert MEAD audio to mel and pair it with landmarks | `Data_emotion_mead_task_b/phaseB_ready/` |
| 2b | `2b_phaseB_emotion_finetune` | Fine-tune on MEAD with emotion input, select **C4** | `phaseB/C4_motion_emo_e20_lr3e4/phaseB_best.pth` |
| 3 | `3_phaseC_demo_render` | **Demo**: render video, Gradio interface, evaluation on mixed-emotion audio | `phaseC_demo/` |
| 4 | `4_phaseD_ser_mead` | Train our own speech-emotion model (SERNet) and compare with emotion2vec+ | `ser_mead/ser_best.pth` |
| 5 | `5_phaseE_benchmark` | Compare with the original MakeItTalk on 480 clips, measure speed | `phaseE/` |
| 6 | `6_phaseF_emotion_from_audio` | Train EmoBranch (F: 8 emotions, F2: neutral + angry + happy) | `phaseF/phaseF2_final.pth` |
| 7 | `7_phaseG_teacher_student` | Further train EmoBranch with emotion2vec+ as teacher on CREMA-D and RAVDESS | `phaseF/phaseG_final.pth` |

### Running only the demo

1. Have the checkpoints on Drive: `phaseB/…/phaseB_best.pth` (C4) and `phaseF/phaseG_final.pth` (or `phaseF2_final.pth`).
2. Open `notebooks/3_phaseC_demo_render.ipynb` on Colab (GPU), keep `QUICK_UI = True`, and run Cell 0 to Cell 11.
3. Open the Gradio link: tab 1 analyses emotion in an audio clip; tab 2 takes an image, 5 clicked face points (cartoon) or MediaPipe auto-detection (human face), and an audio clip, then renders the video.

Best input images: frontal face, mouth closed or slightly open, plain background, at least 512 px.

## Installation

On Colab, each notebook installs what it needs. To run locally:

```
pip install -r requirements.txt
```

## Data, checkpoints and results (Google Drive)

Code comes straight from this repo (each notebook clones it when it runs). Only the [**Data_student**](https://drive.google.com/drive/folders/1uYDtQQwLglkdYr9iEFueLsjZFlQ50ZtI) folder is shared on Google Drive: processed data, checkpoints and the results of every phase.

To re-run the notebooks:

1. Open the **Data_student** link, choose **Add shortcut to Drive**, and place the shortcut in `MyDrive`.
2. Open a notebook on Colab. Cell 0 finds the `Data_student` folder through the shortcut. If it picks the wrong folder, set `DRIVE_ROOT_OVERRIDE`, e.g. `/content/drive/MyDrive/Data_student`.
3. Notebooks `1_phaseA_student` and `2b_phaseB_emotion_finetune` contain the fixed path `/content/drive/MyDrive/DAT301m/Data_student`; change it to your shortcut path before running.

| Folder in `Data_student` | Contents |
| --- | --- |
| [Data_maketalk_landmard_task_a](https://drive.google.com/drive/folders/17Bi8fdauuQrj-fZvjWkDoLqatpxnUfiC) | Obama data for Phase A (mel + landmarks) |
| [Data_emotion_mead_task_b](https://drive.google.com/drive/folders/1jOxdzWDUUwrhM4LKRvLPYvWXFa-7L2GL) | MEAD landmarks (Stage B), `phaseB_ready` data, emotion2vec+ features |
| [student_a2l](https://drive.google.com/drive/folders/1oBZb5C32npb5anWjR0TKRbj8qZSPxwkV) | Checkpoints and outputs of each phase: |
| &nbsp;&nbsp;&nbsp;├ [phaseA](https://drive.google.com/drive/folders/1-gQP-sTU01kVVNVyqmGVxi03kBZEhhBl) | Student model, loss curves, evaluation tables |
| &nbsp;&nbsp;&nbsp;├ [phaseB](https://drive.google.com/drive/folders/1pBrX8Onwr7LK_vaXeqhRAcKOqs_Bb11t) | C4 and variants B1–C5, `phaseB_evaluation.json` |
| &nbsp;&nbsp;&nbsp;├ [phaseC_demo](https://drive.google.com/drive/folders/1TdcXdMUlZ4H05YBENzazcVA6kqRJ_7C0) | Demo videos, expression images, mixed-emotion evaluation figures |
| &nbsp;&nbsp;&nbsp;├ [ser_mead](https://drive.google.com/drive/folders/1H3y1Oj-JLSVIBbnG8s4qKjjiAKVBkUvH) | SERNet (Phase D), confusion matrix |
| &nbsp;&nbsp;&nbsp;├ [phaseE](https://drive.google.com/drive/folders/1ak3VCYX6LDk4wNouj93cSo2_gDIR4GZU) | Tables and charts comparing with MakeItTalk |
| &nbsp;&nbsp;&nbsp;├ [phaseF](https://drive.google.com/drive/folders/1K4xQSkgzU36xo3tlWaOuPOpFCbFyCnj-) | EmoBranch: `phaseF2_final.pth`, `phaseG_final.pth`, evaluation tables |
| &nbsp;&nbsp;&nbsp;└ [makeittalk_ckpt](https://drive.google.com/drive/folders/1L8QaVQVoTaC9Hw3SCQsu2UrK__itdoFe) | Original MakeItTalk checkpoints used for comparison |

The original MEAD, CREMA-D and RAVDESS data are not redistributed here; download them from the authors' pages (table below).

## Datasets

| Dataset | Used for |
| --- | --- |
| Obama (preprocessed release from MakeItTalk) | Phase A |
| [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html): 41 speakers, 8 emotions × 3 intensities | Phases B – G. Split by speaker: 33 train / 4 val / 4 test |
| [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D): 91 speakers, audio only | Phase G |
| [RAVDESS](https://zenodo.org/records/1188976): 24 actors, audio only | Phase G |

Data and checkpoints are not stored in Git; see the Google Drive section above.

## Limitations

- The mouth opens less than in reality (amplitude ratio 0.62 vs 1.0) and motion is somewhat over-smoothed.
- Emotion is expressed mainly through the static expression, not yet through motion; emotion intensity is not learned.
- The demo uses only 3 emotions (neutral, angry, happy); no head motion.
- No user study (MOS) yet.

## Credits and licences

- **MakeItTalk** (Zhou et al., SIGGRAPH Asia 2020): [github.com/yzhou359/MakeItTalk](https://github.com/yzhou359/MakeItTalk). The first part of `src/models/model_audio2landmark.py`, `facewarp/`, `examples_cartoon/` and the AutoVC mel extractor come from this project, under Adobe's licence (research use only).
- **emotion2vec+ large** (Ma et al., 2024), run through [FunASR](https://github.com/modelscope/FunASR): used frozen as the emotion feature extractor and as the teacher.
- **MediaPipe** (Google): facial landmark detection for MEAD and for human-face images in the demo.
- MEAD, CREMA-D and RAVDESS belong to their original authors; check each dataset's licence before use.

Developed by the team: the `StudentAudio2Landmark` student model, C4, SERNet, EmoBranch, teacher–student training, the rendering pipeline and mouth limits, the Gradio demo, and the full evaluation protocol.

## Team

- **Vũ Thị Thu Hiền**: main author (model design, training, evaluation, demo, report).
- With support from the other team members.

DAT301m course project.
