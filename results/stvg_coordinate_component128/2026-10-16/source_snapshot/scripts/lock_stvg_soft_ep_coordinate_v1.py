"""Bind complete preparation source closure and exact reused old evidence."""
from pathlib import Path
import sys,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
from scripts.lock_stvg_ecld_runtime002 import closure
def main():
 assert not(BASE/'PREPARATION_RUNTIME.json').exists()
 for name in ['FROZEN_METHOD_LOCK.json','TEMPORAL_SELECTION_LOCK.json']:
  shutil.copy2(OLD/name,BASE/name)
 d=read(OLD/'confirmation/VARIANT_LOCK.json');d['frozen_method_sha256']=sha(BASE/'FROZEN_METHOD_LOCK.json');write(BASE/'confirmation/VARIANT_LOCK.json',d)
 scripts=['stvg_soft_ep_coordinate_common_v1.py','prepare_stvg_soft_ep_coordinate_v1.py','run_stvg_soft_ep_coordinate_source_v1.py','run_stvg_soft_ep_coordinate_inputs_v1.py','lock_stvg_soft_ep_coordinate_v1.py','launch_stvg_soft_ep_coordinate_v1.py','continue_stvg_soft_ep_coordinate_preparation_v1.py',
 'run_stvg_evidence_confirmation_temporal012.py','fit_stvg_evidence_confirmation012.py','prepare_stvg_evidence_confirmation012.py','run_stvg_evidence_teacher_v1.py','stvg_evidence_spatial_phase009.py','stvg_evidence_spatial_inputs009.py']
 src=closure([ROOT/'scripts'/n for n in scripts]);pins={str(p.relative_to(ROOT)):sha(p)for p in src}
 inherited=read(OLD/'SOURCE_RUNTIME.json');pins.update(inherited['pins'])
 for name in ['ROSTER_LOCK.json','SEARCH_DESIGN.json','FROZEN_METHOD_LOCK.json','TEMPORAL_SELECTION_LOCK.json','confirmation/VARIANT_LOCK.json']:
  pins[str((BASE/name).relative_to(ROOT))]=sha(BASE/name)
 for r in roster():
  for key in ['plan_path','subject_path']:pins[r[key]]=sha(ROOT/r[key])
  if 'cached_parent'in r:
   for p in [ROOT/r['source_case_path'],(ROOT/r['source_case_path']).with_suffix('.receipt.json'),ROOT/r['cached_temporal_path'],ROOT/r['cached_parent']/'REFERENCE_LOCK.json']:
    pins[str(p.relative_to(ROOT))]=sha(p)
 protected={p:sha(ROOT/p)for p in ['methods/CURRENT_METHOD.json','artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json']}
 d=dict(status='locked',scope='64 exact saved source/Temporal/Top1 reuse +64 additional dev inputs; no scoring or full task resumption',pins=pins,protected=protected,source_checkpoints=inherited['source_checkpoints'],GT_read=False)
 write(BASE/'SOURCE_RUNTIME.json',d)
 d['pins']=dict(pins,**{str((BASE/'SOURCE_RUNTIME.json').relative_to(ROOT)):sha(BASE/'SOURCE_RUNTIME.json')})
 write(BASE/'PREPARATION_RUNTIME.json',d);print('preparation locked',len(src),len(pins))
if __name__=='__main__':main()
