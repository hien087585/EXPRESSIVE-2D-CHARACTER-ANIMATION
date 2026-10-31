"""Dung ZIP THAY THE <sp>_landmark_package.zip cho cac speaker mat zip goc trong stage_b_mead_out (W033, M032, M037).

Notebook 5 (Cell 5, audio_map) chi doc TEN cac entry '<sp>/clips/<sp>__front__<emo>__L<l>__<nnn>.npz' de ghep clip
landmark <-> audio .m4a theo thu tu trong nhom (cam xuc, muc). Zip thay the giu dung ten / cau truc do:
  * nguon 'ckpt'  : danh sach + noi dung cac clip front trong stage_b_mead_out/<sp>/ckpt/clips (chi doc, vd W033)
  * nguon 'video' : ten clip front quet tu header MEAD/<sp>/video.tar (nhu prepare_baseline_inputs.scan_video_tar,
                    dung lai cache _scan/<sp>/scan.json neu co); entry rong (0 byte) vi khong con npz landmark
Moi zip co them '<sp>/REBUILT.json' (ngoai '/clips/' -> audio_map bo qua) va REBUILT_README.txt ben canh.
Chi ghi vao --out; nguon chi doc. Khong ghi de zip da co.

  python scripts/rebuild_landmark_zips.py --drive <DRIVE> --mead <...\\MEAD> --out <out dir>\\replacement_zips \\
      --scan-cache <LOCAL_OUT>\\baselines\\_scan W033:ckpt M032:video M037:video
"""
import argparse, datetime, hashlib, json, sys, tarfile, zipfile
from pathlib import Path


def front_from_ckpt(stage, sp):
    d = stage / sp / 'ckpt' / 'clips'
    return {p.stem: p for p in sorted(d.glob('%s__front__*.npz' % sp))}, str(d)


def front_from_video(mead, sp, scan_cache):
    c = scan_cache / sp / 'scan.json' if scan_cache else None
    if c and c.exists():
        return {n: None for n in sorted(json.loads(c.read_text())['front'])}, '%s/video.tar (header scan, cache %s)' % (mead / sp, c)
    front = []
    with tarfile.open(mead / sp / 'video.tar', 'r:') as tf:          # doc het header: vai tar khong xep theo goc quay
        for m in iter(tf.next, None):
            if m.isfile() and m.name.startswith('video/front/') and m.name.endswith('.mp4'):
                q = Path(m.name).parts                                  # video/front/<emo>/level_<l>/<nnn>.mp4
                front.append('%s__front__%s__L%s__%s' % (sp, q[2], q[3].replace('level_', ''), Path(q[4]).stem))
    return {n: None for n in sorted(front)}, '%s/video.tar (header scan)' % (mead / sp)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', type=Path, required=True)
    ap.add_argument('--mead', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--scan-cache', type=Path, default=None)
    ap.add_argument('jobs', nargs='+', help='<speaker>:ckpt | <speaker>:video')
    a = ap.parse_args()
    stage = a.drive / 'Data_emotion_mead_task_b' / 'stage_b_mead_out'
    now = datetime.datetime.now().isoformat(timespec='seconds')
    for job in a.jobs:
        sp, src = job.split(':')
        clips, src_desc = front_from_ckpt(stage, sp) if src == 'ckpt' else front_from_video(a.mead, sp, a.scan_cache)
        assert clips, '%s: khong thay clip front (%s)' % (sp, src_desc)
        od = a.out / sp; od.mkdir(parents=True, exist_ok=True)
        zp = od / ('%s_landmark_package.zip' % sp)
        assert not zp.exists(), 'da co %s (khong ghi de)' % zp
        rc = stage / sp / 'receipt.json'
        receipt = json.loads(rc.read_text()) if rc.exists() else {}
        meta = dict(replacement=True, note='ZIP THAY THE, KHONG PHAI BAN GOC', speaker=sp, built_at=now, source=src_desc,
                    script='scripts/rebuild_landmark_zips.py', n_front_clips=len(clips),
                    clip_content='npz copy tu ckpt/clips' if src == 'ckpt' else 'entry rong (0 byte), chi giu ten',
                    original_receipt=receipt or None)
        with zipfile.ZipFile(zp, 'x', zipfile.ZIP_STORED) as z:
            for name, p in clips.items():
                arc = '%s/clips/%s.npz' % (sp, name)
                if p is None:
                    z.writestr(arc, b'')
                else:
                    z.write(p, arc)
            z.writestr('%s/REBUILT.json' % sp, json.dumps(meta, indent=1))
        h = sha256(zp)
        (od / 'REBUILT_README.txt').write_text('\n'.join([
            'ZIP THAY THE - KHONG PHAI BAN GOC',
            'speaker          : %s' % sp,
            'ngay dung        : %s' % now,
            'nguon            : %s' % src_desc,
            'noi dung clip    : %s' % meta['clip_content'],
            'so clip front    : %d' % len(clips),
            'script           : scripts/rebuild_landmark_zips.py %s' % ' '.join(sys.argv[1:]),
            'file             : %s' % zp.name,
            'sha256 moi       : %s' % h,
            'kich thuoc moi   : %d byte' % zp.stat().st_size,
            'sha256 goc       : %s' % (receipt.get('sha256') or 'khong co receipt'),
            'kich thuoc goc   : %s byte (receipt; n_clips moi goc %s)' % (receipt.get('n_bytes'), receipt.get('n_clips')),
            '',
            'Chi dung cho notebook 5 (Cell 5 audio_map: ghep ten clip front <-> audio.tar). Sha256 KHAC ban goc la binh thuong.',
            '']), encoding='utf-8')
        print('%s: %d clip front | %s | sha256 %s' % (sp, len(clips), zp, h))


if __name__ == '__main__':
    main()
