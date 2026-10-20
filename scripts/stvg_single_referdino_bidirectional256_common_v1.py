"""Single-expert experiment metadata and sanitized Source/ReferDINO inputs."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
D=ROOT/'artifacts/stvg_single_referdino_bidirectional256_v1'
P=ROOT/'results/stvg_error_state/2026-10-20/single_referdino_bidirectional256'
OLD=ROOT/'artifacts/stvg_spatial_strong_bidirectional256_v1'
PRE=ROOT/'artifacts/stvg_referdino_condition_hc128_v1'
TUNE=ROOT/'artifacts/stvg_vid_joint_parameter_search128_v1'
ROWS=['Source','Spatial_only','Temporal_only','Full','Without_spatial_adaptation']
def read(f):return json.loads(Path(f).read_text())
def put(f,x):Path(f).parent.mkdir(parents=True,exist_ok=True);Path(f).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb')as g:
  for b in iter(lambda:g.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def load(f):
 import torch
 return torch.load(f,map_location='cpu',weights_only=False)
def save(f,x):
 import torch,os
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,f);return sha(f)
def verify():
 p=read(D/'PLAN_LOCK.json')
 for f,h in {**p['pins'],**p['protected']}.items():assert sha(ROOT/f)==h,f
 return p
def prepare():
 import torch,numpy as np
 from methods.stvg_single_referdino_opd_v1.initialization import initialize
 from methods.stvg_single_referdino_opd_v1.positions import observation_positions
 assert not(D/'PLAN_LOCK.json').exists();old=read(OLD/'PLAN_LOCK.json');manifest=read(ROOT/'methods/stvg_single_referdino_opd_v1/config.json');inputs={};sources={};reused=0
 for r in old['records']:
  ds,a=r['dataset'],r['alias'];z=load(OLD/f'predictions/{ds}/{a}.pt');sources[str((OLD/f'predictions/{ds}/{a}.pt').relative_to(ROOT))]=sha(OLD/f'predictions/{ds}/{a}.pt')
  if ds=='hc2':
   f=PRE/f'inputs/{a}.pt';base=load(f);rf=PRE/f'teacher0/{a}.pt';teacher=load(rf)['native'];indices=load(ROOT/f"artifacts/stvg_opd_paper_hc2_revision_v2/inputs/vidstg_to_hc2/clean/{r['query_ordinal']:05}.pt")['indices']
  else:
   f=TUNE/f'inputs/{a}.pt';base=load(f)if f.exists()else None;rf=OLD/f'traces/vidstg/{a}.pt';teacher=load(rf)['teacher'];legacy=load(ROOT/f"artifacts/stvg_opd_paper_hc2_revision_v2/inputs/hcstvg2_to_vidstg/clean/{r['query_ordinal']:05}.pt");indices=legacy['indices']
  positions=observation_positions(z['frame_ids'],indices);assert positions==list(z['planned_positions']),(a,positions,z['planned_positions'])
  rawref={k:v for k,v in teacher.items()if k in ['boxes','selected_query','native_query_scores','selected_frame_scores','mask_summary','text_sha256','input_shape','cost','native_same_track_all8','model_load_seconds','peak_GPU_bytes','process_wall_seconds']}
  src=torch.as_tensor(z['Source'],dtype=torch.float32);assert np.array_equal(src.double().numpy(),z['Source'])
  E,init=initialize(src.double().numpy(),positions,rawref['boxes'].numpy(),manifest['datasets'][ds]['E_calibration'])
  cache=dict(alias=a,dataset=ds,query_ordinal=r['query_ordinal'],Source=src,E=E,initialization=init,teacher_native=rawref,frame_ids=z['frame_ids'],planned_positions=positions,intervals=z['intervals'],posterior=z['posterior'],needs_source_capture=base is None,GT_read=False,
       provenance=dict(Source_prediction_sha256=sources[str((OLD/f'predictions/{ds}/{a}.pt').relative_to(ROOT))],ReferDINO_cache_sha256=sha(rf),eight_positions_reconstructed_without_detector=True,old_E_or_detector_observations_not_in_cache=True))
  if base is not None:
   assert np.array_equal(np.asarray(base['Source'],dtype=np.float64),z['Source']);cache['spatial_inputs']=base['spatial_inputs'];cache['provenance']['Source_features_sha256']=sha(f);reused+=1
  assert not any(k in cache for k in ['expert','observations','original_input']);dest=D/f'inputs/{ds}/{a}.pt';inputs[str(dest.relative_to(ROOT))]=save(dest,cache)
 own=['protocols/stvg_single_referdino_bidirectional256_v1.md','scripts/stvg_single_referdino_bidirectional256_common_v1.py','scripts/run_stvg_single_referdino_bidirectional256_v1.py']+[str(f.relative_to(ROOT))for f in (ROOT/'methods/stvg_single_referdino_opd_v1').glob('*')if f.is_file()]
 own += [str(f.relative_to(ROOT))for f in (ROOT/'methods/stvg_referdino_opd_v1').glob('*.py')]
 p=dict(status='locked_single_expert_before_new_OPD',records=old['records'],configs=manifest['datasets'],rows=ROWS,seeds=[41],expert_count=1,expert='ReferDINO',same_eight_evidence=True,new_initialization='robust global correction of native selected ReferDINO8 singleton references',Strong=old['Strong'],checkpoints=old['checkpoints'],source_features_reused=reused,source_features_missing=256-reused,
   all_source_and_time_frozen=True,prior_parameter_search='paused by user before new lr/tau trials',Vid30_selection='completed old-E nine-grid maximum; developer data, not confirmed optimum',
   pins={f:sha(ROOT/f)for f in own},protected={f:sha(ROOT/f)for f in ['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','methods/RESEARCH_METHOD.json']},original_source_files=sources,prepared_input_files=inputs,created_epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','prepared_input_files']});put(D/'STATE.json',dict(status='prepared',source_features_reused=reused,missing=256-reused,new_independent_detector_calls=0));print('PREPARED',reused,256-reused,flush=True)
if __name__=='__main__':prepare()
