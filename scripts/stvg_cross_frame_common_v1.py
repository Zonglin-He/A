"""Isolated final mechanism scope, immutable custody and no-label guard."""
from pathlib import Path
from scripts.stvg_instance_identity_common_v1 import read,write,sha,guard,row
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_cross_frame_validation_v1'
PRIOR=ROOT/'artifacts/stvg_instance_identity_update_v1'
OLD=ROOT/'artifacts/stvg_native_instance_spatial_opd_v1'
def verify(stage='student'):
 d=read(BASE/(stage.upper()+'_RUNTIME.json'))
 if d['status']!='locked' or d['GT_read'] is not False:raise RuntimeError('runtime not locked GT-free')
 for rel,h in d['pins'].items():
  if sha(ROOT/rel)!=h:raise RuntimeError('runtime pin changed: '+rel)
 return d
