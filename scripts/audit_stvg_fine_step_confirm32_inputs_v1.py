"""Postcapture cache/clock/dependency audit before any confirmation fit/GT."""
from pathlib import Path
import sys,json,hashlib,datetime,subprocess
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=ROOT/'artifacts/stvg_fine_step_confirm32_v1'
def main():
 import torch,numpy as np
 p=read(D/'PLAN_LOCK.json');bar=read(D/'FEATURE_BARRIER.json');assert bar['count']==32;checks={};models={};count=0
 for rr in bar['files']:assert sha(ROOT/rr['path'])==rr['sha256']
 for r in p['records']:
  x=load(D/'features'/(r['alias']+'.pt'));s=load(ROOT/r['source_path']);e=load(ROOT/r['prediction_path']);assert s['query_ordinal']==e['query_ordinal']==r['query_ordinal'];assert len(s['logits'])==len(s['records'])==2;assert len(e['frames'])==len(e['estimate']['selected']);ids=e['frame_ids'];assert x['frame_ids']==s['frame_ids']==ids and x['interval']==e['interval'];assert torch.equal(x['E'],torch.as_tensor(e['outputs']['E_DINO_support']));assert x['Source_and_time_logits_exact'];assert x['maps'].shape[0]==len(ids)and x['maps'].shape[1]==1024 and torch.isfinite(x['maps']).all()
  for off in [0,1]:assert s['records'][off]['offset']==off and s['records'][off]['frame_ids']==ids[off::2]
  pos=[];bb=[]
  for f,k in zip(e['frames'],e['estimate']['selected']):
   assert 0<=f['position']<len(ids)and f['position']in e['planned_positions'];assert k==-1 or 0<=k<len(f['boxes'])
   if k>=0:pos.append(f['position']);bb.append(torch.as_tensor(f['boxes'][k]).tolist())
  assert pos==x['positions']and(len(bb)==0 or np.array_equal(np.asarray(bb),np.asarray(x['boxes'])));models.setdefault(r['dataset'],set()).add(x['model_hash']);count+=1
 # Dependency audit is explicitly POSTcapture, not retroactively prelocked.
 deps=set(Path(f)for f in read(ROOT/'artifacts/stvg_dino_supported_source256_v1/RUNTIME_LOCK.json')['pins']if f.endswith('.py'))
 deps.update(Path('scripts')/f for f in ['run_decota_paper_main_v1.py','run_spatial_regression_alignment_v1.py','run_stvg_dino_supported_expanded256_recovery002.py'])
 deps.update(f.relative_to(ROOT)for f in (ROOT/'vg_tta').glob('stvg_opd_frozen*.py'));deps.update(f.relative_to(ROOT)for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py'));deps.update(f.relative_to(ROOT)for f in (ROOT/'external/TA-STVG/models').rglob('*.py'))
 journal=[json.loads(line)for line in subprocess.check_output(['journalctl','--user','-u','stvg-fine-step-confirm32-capture001.service','--no-pager','-o','json'],text=True).splitlines()];started=[j for j in journal if j.get('JOB_TYPE')=='start'and j.get('JOB_RESULT')=='done'];assert len(started)==1;start=int(started[0]['__REALTIME_TIMESTAMP'])/1e6;binding={str(f):dict(sha256=sha(ROOT/f),mtime=(ROOT/f).stat().st_mtime,existed_unchanged_by_mtime_before_launch=(ROOT/f).stat().st_mtime<=start)for f in sorted(deps)if (ROOT/f).is_file()};assert all(v['existed_unchanged_by_mtime_before_launch']for v in binding.values())
 out=dict(status='pass',count=count,two_offsets_cached_records_exact=True,selected_anchor_index_and_clock_valid=True,raw_layer3_finite=True,models={d:sorted(v)for d,v in models.items()},dependency_audit_timing='postcapture before fits, not retroactive prelock',dependencies=binding,initial_runtime_inputs_and_core_pinned=True,actual_forward_Source_logits_exact_asserted_during_capture=True,CPU_did_not_rerun_backbone=True,no_GT_read=True)
 put(D/'INPUT_DEPENDENCY_AUDIT.json',out);print('input/dependency audit',count,len(binding),flush=True)
if __name__=='__main__':main()
