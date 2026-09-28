"""Geometry readback and fixed-state functional diagnostic; no fitting or online rule."""
import sys,json,time,argparse,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
PARENT=ROOT/'artifacts/c1_concept_local_v1'
OUT=ROOT/'artifacts/c1_concept_state_diagnostic_v1'
QUERY='spatial.query_residual'
def stem(r):return r['key'].replace(':','_')
def name(fam,cfg):return f'{fam}|rho{cfg["rho"]}|m{cfg["multiplier"]}'
def check_parent():
 m=read(PARENT/'RESULT_MANIFEST.json')
 for rel,e in m['files'].items():assert sha(PARENT/rel)==e['sha256'],rel
 for path,h in read(PARENT/'CODE_MANIFEST_FINAL.json')['files_sha256'].items():assert sha(path)==h,path
 return len(m['files'])
def prepare():
 import torch
 check_parent();p=read(PARENT/'LOCK.json');selection=read(PARENT/'SELECTION.json');rows=[]
 for r in p['rows']:
  f=PARENT/r['split']/(stem(r)+'.pt');x=load(f)
  cfg=selection[r['cohort']]['S']
  R=x['fits'][name('R',cfg)];S=x['fits'][name('S',cfg)]
  assert len(R['path'])==len(S['path'])==(11 if x['anchors'] else 1)
  for family in [R,S]:
   assert family['path'][-1]['step']==(10 if x['anchors'] else 0)
   assert all(torch.equal(v,family['initial_state'][k]) for k,v in R['initial_state'].items())
  rows.append({**r,'parent_file':str(f),'parent_sha':sha(f),'features_file':x['features_file'],'features_sha':x['features_sha'],'config':cfg,'actual_step':10 if x['anchors'] else 0})
 pins=dict(p['pins'])
 for rel in ['protocols/c1_concept_state_diagnostic_v1.md','scripts/c1_concept_state_diagnostic_v1.py','scripts/run_spatial_ssl_gpu_v1.py','scripts/run_spatial_regression_alignment_v1.py','scripts/spatial_consolidation_roles_v1.py']:
  pins[rel]=sha(ROOT/rel)
 write(OUT/'LOCK.json',dict(rows=rows,pins=pins,parent_result_manifest_sha=sha(PARENT/'RESULT_MANIFEST.json'),parent_text_lock_sha=sha(PARENT/'TEXT_LOCK.json'),parent_selection_sha=sha(PARENT/'SELECTION.json'),labels=p['labels'],labels_sha=p['labels_sha'],max_seconds=3600,max_bytes=2*1024**3,created=time.time(),GT_new_state_construction=False,GT_historical_exposure=True,new_fits=0,new_backwards=0,nominal_step=10))
 status(OUT/'STATUS.json',dict(status='prepared',sources=64))
def verify():
 p=read(OUT/'LOCK.json')
 for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
 assert sha(PARENT/'RESULT_MANIFEST.json')==p['parent_result_manifest_sha']
 assert sha(PARENT/'SELECTION.json')==p['parent_selection_sha']
 return p
def committed(f):
 if not f.with_suffix('.json').exists():return False
 rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];assert rc['lock_sha']==sha(OUT/'LOCK.json');return True
def commit(f,x):
 save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),lock_sha=sha(OUT/'LOCK.json')))
def distribution(v):
 import numpy as np
 a=np.asarray(v,dtype=float)
 if not len(a):return dict(n=0,mean=None,median=None,min=None,max=None,p10=None,p90=None)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))
def geometry():
 import torch,numpy as np
 from vg_tta.box_stability_diagnostics_v1 import overlap
 p=verify();texts=read(PARENT/'TEXT_LOCK.json');torch.set_num_threads(4);rows=[];matrices={}
 for r in p['rows']:
  assert sha(r['features_file'])==r['features_sha'];assert sha(r['parent_file'])==r['parent_sha']
  x=load(r['parent_file']);f=load(r['features_file']);z0=f['original']['z'];scale=float(z0.norm());rank=f['rank'];cfg=r['config'];radius=cfg['rho']*scale;groups={};mr={}
  tr=texts['rows'][r['key']]
  for tag,group,saved_spectrum in [('S','basis','singular_S'),('N','far_basis','singular_N')]:
   expr=f['groups'][group];D=torch.stack([v['z']-z0 for v in expr],1).double() if expr else torch.zeros(256,0,dtype=torch.double)
   u,s,vh=torch.linalg.svd(D,full_matrices=False);assert torch.allclose(s.float(),f[saved_spectrum],rtol=1e-6,atol=1e-8)
   numerical=int((s>max(1e-10,float(s[0])*1e-5)).sum()) if len(s) else 0
   retained=s[:rank];norms=D.norm(dim=0);positive=norms[norms>0]
   residual=x['fits'][name(tag,cfg)]['path'][-1]['state'][QUERY].double()
   coeff=u[:,:rank].T@residual
   boxes=[np.asarray(v['boxes']) for v in expr];before=np.asarray(x['before']);pos=[i for i in range(len(before)) if i not in x['planned']]
   disagreements=[float(1-overlap(v,before).mean()) for v in boxes]
   check_disagreements=[float(1-overlap(v[pos],before[pos]).mean()) for v in boxes]
   raw=norms.tolist();relative=(norms/scale).tolist()
   per=[dict(text=t,delta_norm=n,delta_relative=nr,box_disagreement_all=d,box_disagreement_nonplanned=dc,mean_abs_box_difference=float(np.abs(b-before).mean())) for t,n,nr,d,dc,b in zip(tr[group],raw,relative,disagreements,check_disagreements,boxes)]
   energy=float(s.square().sum())
   groups[tag]=dict(expressions=per,raw_relative=distribution(relative),singular_values=s.tolist(),singular_relative=(s/scale).tolist(),numerical_rank=numerical,actual_rank=rank,retained_energy=float(retained.square().sum()/energy) if energy else None,participation_rank=energy**2/float(s.pow(4).sum()) if energy else 0.,radius_over_median_raw=radius/float(positive.median()) if len(positive) else None,radius_over_max_raw=radius/float(positive.max()) if len(positive) else None,radius_over_each_raw=(radius/positive).tolist(),radius_over_singular=(radius/retained).tolist(),actual_residual_relative=float(residual.norm())/scale,actual_coefficients=coeff.tolist(),actual_coeff_over_singular=(coeff/retained).tolist(),linear_combination_min_norm=float((coeff/retained).norm()) if rank else 0.,box_disagreement_all=distribution(disagreements),box_disagreement_nonplanned=distribution(check_disagreements))
   mr[tag]=dict(D=D,U=u,S=s,Vh=vh)
  rows.append(dict(key=r['key'],cohort=r['cohort'],split=r['split'],source=r['source'],rank=rank,anchors=len(x['anchors']),scale=scale,radius=radius,config=cfg,actual_step=r['actual_step'],groups=groups))
  matrices[r['key']]=mr
 write(OUT/'GEOMETRY.json',rows);save(OUT/'DIFFERENCE_MATRICES.pt',matrices)
 write(OUT/'GEOMETRY_BARRIER.json',dict(files={n:sha(OUT/n) for n in ['GEOMETRY.json','DIFFERENCE_MATRICES.pt']},sources=64,GT_read=False))
 print('Geometry complete',len(rows),flush=True)
def cross(limit=0):
 import torch
 from scripts.spatial_consolidation_roles_v1 import setup
 from scripts.run_final_simplification_v1 import lease
 from scripts.run_spatial_regression_alignment_v1 import model_load
 from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
 from vg_tta.exact_frame_decode_audit_v2 import decode
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from methods.decota_final_simplified_v1.backbone import query_subject,full_prediction
 from methods.decota_final_simplified_v1.objectives import SpatialLoss
 p=verify();setup();guard=lease();start=time.time();done=0;model=None;current=None
 torch.cuda.reset_peak_memory_stats()
 try:
  for r in sorted(p['rows'],key=lambda v:(v['cohort'],v['split'],v['key'])):
   out=OUT/'cross'/(stem(r)+'.pt')
   if committed(out):done+=1;continue
   if limit and done>=limit:break
   assert time.time()-start<p['max_seconds']
   if current!=r['cohort']:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_load(r['cohort']);current=r['cohort'];mh=state_hash(model.state_dict())
   assert sha(r['parent_file'])==r['parent_sha'];assert sha(r['original'])==r['original_sha']
   x=load(r['parent_file']);original=load(r['original']);frames,ids=decode(original['input']);assert ids==x['frame_ids']
   batch,records,replay=frozen_forward(model,frames,original);assert torch.equal(replay.zero['boxes'].cpu(),x['native'])
   cfg=r['config'];R=x['fits'][name('R',cfg)]['path'][-1];S=x['fits'][name('S',cfg)]['path'][-1]
   assert R['step']==S['step']==r['actual_step']
   initial=x['fits'][name('R',cfg)]['initial_state'];replay.restore(detached(initial,'cuda'))
   with torch.no_grad():before=detached(replay.values())
   assert torch.equal(before['boxes'].cpu(),x['before']);lossfn=SpatialLoss(x['anchors'],before['boxes']);arms={}
   for ln,qr in [('R','R'),('R','S'),('S','R'),('S','S')]:
    src={'R':R['state'],'S':S['state']};state={k:(src[qr][k] if k==QUERY else src[ln][k]).clone() for k in src[ln]}
    replay.restore(detached(state,'cuda'))
    with torch.no_grad():expected=detached(replay.values());loss=float(lossfn(expected['boxes']))
    if ln==qr:assert torch.equal(expected['boxes'].cpu(),(R if ln=='R' else S)['boxes'])
    with query_subject(model,batch,r['subject']):full=full_prediction(model,batch,ids,records,detached(state,'cuda'),expected)
    arms[ln+qr]=dict(state=state,boxes=full['boxes'],reference_loss=loss,full_reinsertion=True)
   assert state_hash(model.state_dict())==mh
   payload=dict(key=r['key'],source=r['source'],split=r['split'],cohort=current,query=r['query'],config=cfg,nominal_step=10,actual_step=r['actual_step'],frame_ids=ids,indices=x['indices'],anchors=x['anchors'],observed=x['observed'],planned=x['planned'],before=x['before'],F10=x['fits']['F']['path'][-1]['boxes'],arms=arms,audit=dict(native_exact=True,Before_exact=True,diagonals_exact=True,full_reinsertions=4,source_restored=True,GT_access=False,new_fits=0),seconds_since_start=time.time()-start)
   commit(out,detached(payload,'cpu'));done+=1
   status(OUT/'STATUS.json',dict(status='cross_running',sources_done=done,total=64,seconds=time.time()-start))
   print('CROSS',done,r['key'],'step',r['actual_step'],'seconds',round(time.time()-start,2),flush=True)
   del frames,batch,records,replay,before,expected,x,original,arms,payload,full;gc.collect();torch.cuda.empty_cache()
  receipt=dict(done=done,seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_fits=0,new_backwards=0,GT_read=False)
  write(OUT/f'GPU_STAGE_{time.time_ns()}.json',receipt)
  if done==64:
   files={str(OUT/'cross'/(stem(r)+'.pt')):sha(OUT/'cross'/(stem(r)+'.pt')) for r in p['rows']}
   write(OUT/'CROSS_BARRIER.json',dict(files=files,sources=64,cross_readouts=256,GT_read=False))
   status(OUT/'STATUS.json',dict(status='cross_complete_unscored',**receipt))
 except BaseException as e:
  write(OUT/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),done=done,seconds=time.time()-start));raise
 finally:guard.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','geometry','cross']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 if a.phase=='prepare':prepare()
 elif a.phase=='geometry':geometry()
 else:cross(a.limit)
