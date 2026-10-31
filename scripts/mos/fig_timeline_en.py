"""Fig. 9 (ban tieng Anh): timeline cam xuc tren file ghep M037_MIX1 (4 speaker test, ghep tu MEAD).
Tren: xac suat 3 lop cua ensemble (SpEx-KD + lop phan loai goc emotion2vec+). Duoi: 4 dai Ground truth / ELSA (ensemble) /
SpEx-KD / Teacher (emotion2vec+). Doc phaseC_demo/test_audio/M037_MIX1_timeline.npz (tao bang colab_mix_timeline_cell.py
o cuoi notebook 3; chi doc Drive) -> <MOS_DIR>/_check/fig_timeline_en.png, 300 dpi.
  python fig_timeline_en.py [npz] [png]
Mac dinh npz = <DRIVE>/student_a2l/phaseC_demo/test_audio/M037_MIX1_timeline.npz (scripts/local_paths.py)."""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_paths import local_path  # noqa: E402
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    local_path('DRIVE') / 'student_a2l' / 'phaseC_demo' / 'test_audio' / 'M037_MIX1_timeline.npz'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else local_path('MOS_DIR') / '_check' / 'fig_timeline_en.png'
COL = {'neutral': '#9aa0a6', 'angry': '#d93025', 'happy': '#f29900'}
NAME = {'neutral': 'Neutral', 'angry': 'Angry', 'happy': 'Happy'}

z = np.load(SRC)
cls, dur = [str(c) for c in z['classes']], float(z['dur'])
print('%s | %.1f s | audio dung lai trung file: %s' % (z['name'], dur, bool(z['audio_identical'])))
rows = [('Ground truth', z['gt_start'], z['gt_end'], z['gt_emo'])] + \
       [(lab, z[k + '_start'], z[k + '_end'], z[k + '_emo']) for lab, k in (('ELSA (ensemble)', 'ens'), ('SpEx-KD', 'spex'), ('Teacher', 'teacher'))]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
fig, ax = plt.subplots(len(rows) + 1, 1, figsize=(6.0, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2.6] + [0.62] * len(rows)))
for j, c in enumerate(cls):
    ax[0].plot(z['ens_centers'], z['ens_prob'][:, j], color=COL[c], lw=1.6, label=NAME[c])
ax[0].set_ylim(0, 1.02); ax[0].set_ylabel('Probability'); ax[0].grid(alpha=.3)
for a, (lab, s0, s1, e) in zip(ax[1:], rows):
    for t0, t1, ee in zip(s0, s1, e):
        a.barh(0, float(t1) - float(t0), left=float(t0), height=0.8, color=COL[str(ee)], edgecolor='white', linewidth=0.8)
    a.set_yticks([0]); a.set_yticklabels([lab]); a.set_ylim(-0.5, 0.5)
    for sp in ('top', 'right', 'left'):
        a.spines[sp].set_visible(False)
    a.tick_params(axis='y', length=0)
ax[-1].set_xlabel('Time (s)'); ax[-1].set_xlim(0, dur)
fig.legend(handles=[Patch(color=COL[c], label=NAME[c]) for c in cls], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=300, facecolor='white')
print('->', OUT)
