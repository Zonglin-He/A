"""Lock separate CPU scoring and no-update token before result inspection."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_routed_common_v1 import *
def run():
 names=['scripts/score_tastvg_routed_online_v1.py','scripts/tastvg_event_support_metrics_v1.py','scripts/diagnose_tastvg_pipeline_cpu_v1.py','scripts/score_tastvg_best_quick_v1.py','scripts/score_tastvg_schedule_j01_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','vg_tta/tastvg_paper_readouts_v1.py','vg_tta/tastvg_reference_selection_v1.py']
 write(BASE/'SCORING_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},diagnostic_labels={ds:dict(path=str((POOL/ds/'GT_LABELS_search.json').relative_to(ROOT)),sha256=sha(POOL/ds/'GT_LABELS_search.json')) for ds in DATASETS},GT_interpreted=False,time=time.time()))
 inputs={}
 for ds in DATASETS:
  p=read(QUAL/ds/'PLAN.json');inputs[str((QUAL/ds/'PLAN.json').relative_to(ROOT))]=sha(QUAL/ds/'PLAN.json')
  for c in p['cells']:
   for f in [ROOT/c['payload'],POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json']:
    inputs[str(f.relative_to(ROOT))]=sha(f)
   r=read(POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json');inputs[str((POOL/ds/r['cache']).relative_to(ROOT))]=r['sha256']
 names=['scripts/run_tastvg_native_token_p0_v1.py','vg_tta/tastvg_native_token_binding_v1.py','scripts/test_tastvg_native_token_p0_v1.py','protocols/tastvg_routed_online_token_v1.md','scripts/score_tastvg_native_token_p0_v1.py']
 write(BASE/'TOKEN_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},inputs=inputs,cells=60,candidates=540,GT_read=False,parameter_update=False,time=time.time()))
 print('SCORING and TOKEN locked, no GT interpretation')
if __name__=='__main__':run()
