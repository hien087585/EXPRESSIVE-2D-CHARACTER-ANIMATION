"""Fig. 6 (ban tieng Anh): do mo moi theo thoi gian, 2 clip test, 3 duong (Real, ELSA before / after calibration).
Doc student_a2l/phaseH/phaseH_mouth_open_curves.npz (tao bang colab_phaseH_curves_cell.py o cuoi notebook 8; chi doc Drive)
-> C:\\hien\\mos\\_check\\fig_lip_gap_en.png, 300 dpi.
Before / after = chuoi demo (kep PCA + lam muot 110 ms) khong gain / gain x1.85 - dung chuoi ELSA dung de render;
doi CURVE_BEFORE / CURVE_AFTER = 0 / 1 de dung C4 tho (khong hau xu ly) nhu hai duong lien trong hinh goc."""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Admin\My Drive\DAT301m\Data_student\student_a2l\phaseH\phaseH_mouth_open_curves.npz')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else r'C:\hien\mos\_check\fig_lip_gap_en.png')
CURVE_BEFORE, CURVE_AFTER = 2, 3            # chi so trong curve_labels: 0 C4 truoc, 1 C4 + gain, 2 demo cu, 3 demo + gain
EMO = {'neutral': 'Neutral', 'angry': 'Angry', 'happy': 'Happy', 'sad': 'Sad', 'disgusted': 'Disgusted',
       'fear': 'Fear', 'surprised': 'Surprised', 'contempt': 'Contempt'}

z = np.load(SRC)
print('curve_labels:', list(z['curve_labels']), '| before =', z['curve_labels'][CURVE_BEFORE], '| after =', z['curve_labels'][CURVE_AFTER])
n = len([k for k in z.files if k.endswith('_t')])
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
fig, axs = plt.subplots(n, 1, figsize=(7.0, 2.6 * n), squeeze=False)
for k, ax in enumerate(axs[:, 0]):
    t = z['c%d_t' % k]
    ax.plot(t, z['c%d_real' % k], color='black', lw=1.8, label='Real')
    ax.plot(t, z['c%d_curve%d' % (k, CURVE_BEFORE)], color='#1f77b4', lw=1.2, ls='--', label='ELSA before calibration')
    ax.plot(t, z['c%d_curve%d' % (k, CURVE_AFTER)], color='#d62728', lw=1.2, label='ELSA after calibration')
    name, emo = str(z['c%d_name' % k]).split(' | ')
    spk = name.split('__')[0]
    ax.set_title('Test clip %d (speaker %s, %s)' % (k + 1, spk, EMO.get(emo.split()[0], emo.split()[0])), fontsize=11)
    ax.set_ylabel('Lip gap (normalised)'); ax.grid(alpha=.3)
    ax.set_xlim(t[0], t[-1])
axs[-1, 0].set_xlabel('Time (s)')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT, dpi=300, facecolor='white')
print('->', OUT)
