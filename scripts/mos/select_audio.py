"""Chon 12 cau MEAD (4 speaker test x neutral / happy / angry, muc cao nhat), 4-8 s, khong trung cau.
Doc audio.tar tren G: (CHI DOC), giai nen + doi sang wav 16 kHz mono vao C:\\hien\\mos\\audio\\, ghi audio_list.csv."""
import csv
import io
import os
import random
import subprocess
import tarfile
from pathlib import Path

G = Path(r'G:\.shortcut-targets-by-id\1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j\MEAD')
OUT = Path(r'C:\hien\mos\audio')
OUT.mkdir(parents=True, exist_ok=True)
FF = (__import__('os').environ.get('FFMPEG') or __import__('shutil').which('ffmpeg') or str(Path(__file__).parent / 'bin' / 'ffmpeg.exe'))
SPK = ['M032', 'M037', 'W018', 'W019']
EMO = [('neutral', ['level_1']), ('happy', ['level_3']), ('angry', ['level_3'])]   # muc cao nhat
import sys; sys.path.insert(0, str(Path(__file__).parent)); from e2v_map import audio_to_clip   # phai co dac trung emotion2vec+
rng = random.Random(2026)
used, rows = set(), []


def duration(path):
    r = subprocess.run([FF, '-i', str(path)], capture_output=True, text=True)
    t = [l for l in r.stderr.splitlines() if 'Duration' in l][0].split('Duration:')[1].split(',')[0].strip()
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def pick(slot_spk, emo, lvls, sources):
    # chon 1 cau cho o (slot_spk, emo): thu speaker cua o truoc, het thi lay speaker khac (cung muc cao nhat)
    for spk in sources:
      with tarfile.open(G / spk / 'audio.tar') as tf:
        mem = {m.name: m for m in tf.getmembers() if m.name.endswith('.m4a')}
        amap = audio_to_clip(spk)
        if True:
          for lvl in lvls:
            cands = sorted(n for n in mem if n.split('/')[-3] == emo and n.split('/')[-2] == lvl)
            rng.shuffle(cands)
            found = False
            for n in cands:
                if not amap.get(n, (None, False))[1]:                # khong co dac trung SpEx-KD -> bo
                    continue
                sid = n.split('/')[-1][:-4]
                key = sid                                # so cau khac nhau tren ca 12 cau (MEAD co the dung chung cau giua cac cam xuc)
                if key in used or sid in ('001', '002', '003'):   # 001-003: cau chung cho moi cam xuc -> bo
                    continue
                base = '%s_%s_%s_%s' % (spk, emo, lvl.replace('level_', 'L'), sid)
                m4a = OUT / (base + '.m4a')
                m4a.write_bytes(tf.extractfile(mem[n]).read())
                wav = OUT / (base + '.wav')
                subprocess.run([FF, '-y', '-loglevel', 'error', '-i', str(m4a), '-ar', '16000', '-ac', '1', str(wav)], check=True)
                d = duration(wav)
                if 4.0 <= d <= 8.0:
                    used.add(key)
                    rows.append(dict(file=wav.name, mead_source='%s/audio.tar:%s' % (spk, n), speaker=spk, emotion=emo,
                                     level=lvl.replace('level_', ''), duration_s='%.2f' % d, e2v_clip=amap[n][0],
                                     slot_speaker=slot_spk))
                    print(rows[-1])
                    found = True
                    break
                m4a.unlink(); wav.unlink()
            if found:
                return
    raise RuntimeError('khong du cau cho %s %s' % (slot_spk, emo))


for spk in SPK:
    for emo, lvls in EMO:
        others = [s_ for s_ in SPK if s_ != spk]
        rng.shuffle(others)
        pick(spk, emo, lvls, [spk] + others)

with open(OUT.parent / 'audio_list.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), 'cau ->', OUT.parent / 'audio_list.csv')
