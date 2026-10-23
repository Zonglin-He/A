import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,hashlib,time
from pathlib import Path
sys.path[:0]=['.','tools']
import torch,numpy as np
from full import read,sha,atomic,PROVIDER,RUN
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry,evidence_dp
R=Path.cwd();D=R/'outputs/cycle_reliability_ablation_v1';torch.set_num_threads(1)
job=read(RUN/'plan.json')['jobs']['hc2'];manifest=PROVIDER/job['manifest'];assert sha(manifest)==job['manifest_sha256']
key=lambda x:hashlib.sha256(json.dumps({k:v for k,v in x['metadata'].items()if k!='video_path'},sort_keys=True).encode()).hexdigest()
lookup={key(x):x for x in map(json.loads,manifest.read_text().splitlines())};dev=[x for x in map(json.loads,(PROVIDER/'data/evaluation/development.jsonl').read_text().splitlines())if x['dataset']=='hc2'];seal=read(RUN/'hc2/sealed.json');items=[];unmatched=[];started=time.perf_counter()
for r in dev:
 raw=PROVIDER/'outputs/layout_reward_dev/student_cache'/(r['id']+'.pt');mp=PROVIDER/'outputs/layout_reward_dev/research/native_masks'/(r['id']+'.pt')
 if not raw.exists()or not mp.exists()or key(r)not in lookup:unmatched.append(dict(alias=r['id'],exact_official_input=key(r)in lookup,raw_exists=raw.exists(),mask_exists=mp.exists()));continue
 row=lookup[key(r)];e=seal['files'][row['id']];cf=RUN/'hc2'/e['capture_path'];assert sha(cf)==e['capture_sha256'];cap=torch.load(cf,map_location='cpu',weights_only=False);old=torch.load(raw,map_location='cpu',weights_only=False);mask=torch.load(mp,map_location='cpu',weights_only=False)
 assert sha(raw)==mask['source_cache_sha256'] and not old['GT_read'] and not mask['GT_read']
 assert np.array_equal(old['Source'],cap['Source'])and old['frame_ids']==list(cap['frame_ids'])and old['planned_positions']==list(cap['positions8'])
 j=int(mask['selected_query']);teacher=mask['raw_out']['pred_boxes'][0,:,j];assert torch.equal(teacher,torch.as_tensor(cap['geometry']['native_boxes']).float())
 ev=SourceEvidence.from_capture(old['spatial_inputs'],cap['frame_ids'],cap['positions8'],mask['raw_out']['pred_masks'][0,:,j].float(),mask['input_shape'][-2:],selected_query=j)
 g=SpatialGeometry(ev,cap['E'],teacher);original=cap['geometry'];checked=[]
 for f,v in original.items():
  if torch.is_tensor(v):assert torch.equal(v,getattr(g,f)),(r['id'],f);checked.append(f)
 with np.load(RUN/'hc2'/e['path'],allow_pickle=False)as out:
  assert np.array_equal(evidence_dp(g.for_interval(out['Full_interval']))['final'].numpy(),out['Full'])
 items.append(dict(alias=r['id'],row=row,source_key=row['id'],raw_path=str(raw),raw_sha256=sha(raw),mask_path=str(mp),mask_sha256=sha(mp),capture_path=str(cf),capture_sha256=sha(cf),prediction_path=str(RUN/'hc2'/e['path']),prediction_sha256=e['sha256'],geometry_tensor_fields_exact=checked))
 if len(items)%16==0:print('matched_exact',len(items),flush=True)
assert len(items)==67 and not torch.cuda.is_initialized()
atomic(D/'source_only_preflight.json',dict(status='passed',queries=67,items=items,unmatched=unmatched,all_cached_geometry_tensor_fields_exact=True,current_Full_boxes_exact=True,no_GT=True,new_models=0,new_expert=0,new_RGB=0,seconds=time.perf_counter()-started))
print('passed',len(items),time.perf_counter()-started,flush=True)
