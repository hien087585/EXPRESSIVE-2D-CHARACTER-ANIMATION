"""Machine-specific paths <LOCAL_OUT>, <MOS_DIR>, <DRIVE> (no defaults in the code).

Read in order: (1) env var ELSA_<KEY>, (2) <REPO>/local_paths.json {"LOCAL_OUT": ..., "MOS_DIR": ..., "DRIVE": ...}
(git-ignored, one per machine), (3) otherwise stop with an error naming both.
  LOCAL_OUT  local output root (baselines/ is under it)
  MOS_DIR    local MOS folder (audio_list.csv, characters/, _check/ ...)
  DRIVE      Data_student folder on Google Drive (read only)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / 'local_paths.json'
KEYS = ('LOCAL_OUT', 'MOS_DIR', 'DRIVE')


def local_path(key):
    assert key in KEYS, key
    v = os.environ.get('ELSA_' + key)
    if not v and CFG.exists():
        v = json.loads(CFG.read_text(encoding='utf-8')).get(key)
    if not v:
        raise SystemExit('path %s is not set: set env var ELSA_%s or add "%s" to %s' % (key, key, key, CFG))
    return Path(v)
