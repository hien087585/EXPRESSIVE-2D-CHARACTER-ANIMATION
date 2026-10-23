"""Ghep file audio MEAD <-> ten clip trong phaseF_e2v (giong audio_groups notebook 6: theo THU TU trong nhom cam xuc, muc)."""
import os, tarfile
import numpy as np
G = r'G:\.shortcut-targets-by-id\1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j\MEAD'
E = r'G:\.shortcut-targets-by-id\1KaV-u_oFzL06sQpotcZhWayVLKmoq0ZS\DAT301m (1)\Data_student\Data_emotion_mead_task_b\phaseF_e2v'
_cache = {}


def audio_to_clip(sp):
    """-> {'audio/happy/level_3/030.m4a': (ten clip, co dac trung?)} cho speaker sp."""
    if sp in _cache:
        return _cache[sp]
    z = np.load(os.path.join(E, sp + '.npz'))
    names, has = [str(x) for x in z['names']], z['has']
    lm = {}
    for i, n in enumerate(names):
        q = n.split('__'); lm.setdefault((q[2], q[3][1:]), []).append((q[4], n, bool(has[i])))
    with tarfile.open(os.path.join(G, sp, 'audio.tar')) as tf:
        aud = [m.name for m in tf.getmembers() if m.name.endswith('.m4a')]
    au = {}
    for a in aud:
        p = a.split('/'); au.setdefault((p[-3], p[-2].replace('level_', '')), []).append(a)
    out = {}
    for k, items in lm.items():
        a = sorted(au.get(k, []), key=lambda s: s.split('/')[-1])
        if len(a) == len(items):
            for (sid, n, h), an in zip(sorted(items), a):
                out[an.split(':')[-1]] = (n, h)
    _cache[sp] = out
    return out


if __name__ == '__main__':
    import csv
    for c in csv.DictReader(open(r'C:\hien\mos\audio_list.csv')):
        m = audio_to_clip(c['speaker'])
        k = c['mead_source'].split(':')[1]
        print(c['file'], m.get(k, 'KHONG GHEP DUOC (nhom lech so luong)'))
