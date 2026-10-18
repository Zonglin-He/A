"""CPU-only sealed exact decomposition of cached64 temporal projection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,time,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(4)
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
D=ROOT/'artifacts/stvg_projection_components64_v1';P=ROOT/'results/stvg_interval_decision/2026-10-18/components64';OLD=ROOT/'artifacts/stvg_interval_decision_v1'
ARMS=['Native','full_projection','length_only','fluctuation_only','native_MBR'];GEOMETRIES=['Source','Before','After'];SUPPORTS=['offset0','offset1','fused']
OWN=['scripts/run_stvg_projection_components64_v1.py','scripts/report_stvg_projection_components64_v1.py','protocols/stvg_projection_components64_v1.md']
def prepare():
 old=read(OLD/'PLAN_LOCK.json');b=read(OLD/'GLOBAL_READOUT_BARRIER.json');assert len(old['files'])==b['count']==64
 refs={Path(x['path']).stem:x for x in b['files']};files=[]
 for r in old['files']:
  assert sha(ROOT/r['path'])==r['sha256'];v=refs[r['alias']];assert sha(ROOT/v['path'])==v['sha256'];files.append(dict(**r,reference_readout=v))
 put(D/'PLAN_LOCK.json',dict(status='locked_before_components_and_GT',count=64,files=files,arms=ARMS,geometries=GEOMETRIES,primary_geometry='After',supports=SUPPORTS,prior_weight=.1,GT_before_readout_seal=False,new_model_calls=0,new_training=0,GPU_seconds=0,created_epoch=time.time(),deadline_epoch=time.time()+3600))
 names=['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in OWN+['methods/decota_final_simplified_v1/objectives.py','vg_tta/tastvg_oracle_event5_v1.py']},registries={f:sha(ROOT/f)for f in names},existing_feature_sha256=sha(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')))
 print('LOCKED64',flush=True)
def verify():
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch']
 for f,h in {**read(D/'RUNTIME_LOCK.json')['pins'],**read(D/'RUNTIME_LOCK.json')['registries']}.items():assert sha(ROOT/f)==h,f
 return p

def readouts():
 from types import SimpleNamespace
 from methods.decota_final_simplified_v1.objectives import project,native_view_indices
 files=[];checks=0;maxerr=0.;beg=time.time()
 for r in verify()['files']:
  assert sha(ROOT/r['path'])==r['sha256'];z=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);ref=load(ROOT/r['reference_readout']['path']);assert ref['input_source_sha256']==r['sha256']
  cfg=SimpleNamespace(**z['config']);assert cfg.prior_weight==.1;ev=project(z['capture']['zero']['logits'],z['capture']['zero']['actions'],z['capture']['records'],cfg);ivs={a:[]for a in ARMS};aud=[]
  for off,(logits,record,e) in enumerate(zip(z['capture']['zero']['logits'],z['capture']['records'],ev['offsets'])):
   u,w,ij,lp=e['standardized_logits'],e['omega'],e['ij'],e['logp0'];assert abs(float(w.sum())-1)<1e-14
   bar=(w*u).sum();massp=torch.cat((w.new_zeros(1),w.cumsum(0)));fp=torch.cat((u.new_zeros(1),(w*(u-bar)).cumsum(0)));mass=massp[ij[1]+1]-massp[ij[0]];L=bar*mass;F=fp[ij[1]+1]-fp[ij[0]];constant=(w*torch.nn.functional.logsigmoid(-u)).sum();prior=cfg.prior_weight*lp
   scores=dict(length_only=prior+L,fluctuation_only=prior+F);nv=native_view_indices(logits);ns=[record['frame_ids'][nv[0]],record['frame_ids'][nv[1]]+1]
   ivs['Native'].append(ns);ivs['full_projection'].append(e['interval']);ivs['native_MBR'].append(ref['offset_intervals']['native_MBR'][off])
   for a,s in scores.items():
    k=int(s.argmax());i,j=ij[:,k].tolist();ivs[a].append([record['frame_ids'][i],record['frame_ids'][j]+1])
   # Independent direct enumeration uses sliced numpy sums, never the prefix implementation.
   un,wn,pn=u.numpy(),w.numpy(),lp.numpy();bn=float(np.dot(wn,un));direct=[];ld=[];fd=[]
   for i,j in ij.T.tolist():
    l=bn*float(np.sum(wn[i:j+1]));f=float(np.sum(wn[i:j+1]*(un[i:j+1]-bn)));direct.append(float(constant)+.1*pn[len(direct)]+float(np.dot(wn[i:j+1],un[i:j+1])));ld.append(.1*pn[len(ld)]+l);fd.append(.1*pn[len(fd)]+f)
   errs=[float(np.max(abs(np.asarray(direct)-e['score'].numpy()))),float((constant+prior+L+F-e['score']).abs().max()),float(np.max(abs(np.asarray(ld)-scores['length_only'].numpy()))),float(np.max(abs(np.asarray(fd)-scores['fluctuation_only'].numpy())))];assert max(errs)<1e-8;maxerr=max(maxerr,*errs)
   for a,arr in [('length_only',ld),('fluctuation_only',fd)]:assert int(np.argmax(arr))==int(scores[a].argmax());checks+=len(arr)
   assert int(np.argmax(direct))==e['target'];checks+=len(direct)
   info={}
   for a in ARMS:
    iv=ivs[a][-1];i=record['frame_ids'].index(iv[0]);j=record['frame_ids'].index(iv[1]-1);wm=float(w[i:j+1].sum());physical=(iv[1]-iv[0])/(record['frame_ids'][-1]+1-record['frame_ids'][0]);info[a]=dict(weighted_cell_mass=wm,endpoint_span_fraction=physical,difference=wm-physical)
   aud.append(dict(offset=off,legal_intervals=ij.shape[1],bar_u=float(bar),constant=float(constant),max_decomposition_error=max(errs),direct_argmax_all_exact=True,original_target_indices=e['target_indices'],length_definitions=info))
  fused={a:[min(v[0]for v in ivs[a]),max(v[1]for v in ivs[a])]for a in ARMS}
  for a,oldname in [('Native','Native'),('full_projection','projection'),('native_MBR','native_MBR')]:assert ivs[a]==ref['offset_intervals'][oldname]and fused[a]==ref['fused_intervals'][oldname];checks+=3
  assert fused['full_projection']==z['predictions']['ActionnessProjection']['physical_interval'];assert fused['Native']==z['predictions']['Native']['physical_interval']
  boxes=dict(Source=z['capture']['zero']['boxes'],Before=z['original_spatial_before']['boxes'],After=z['predictions']['SpatialOPD']['boxes'])
  files.append(save(D/'readouts'/(r['alias']+'.pt'),dict(alias=r['alias'],dataset=r['dataset'],query_ordinal=z['query_ordinal'],source_sha256=r['sha256'],offset_intervals=ivs,fused_intervals=fused,boxes=boxes,frame_ids=z['frame_ids'],audit=aud,actual_config=z['config'],GT_read=False)));print('SEALED',r['alias'],flush=True)
 put(D/'GLOBAL_READOUT_BARRIER.json',dict(status='complete',count=64,files=files,GT_read=False,new_models=0,GPU_seconds=0,enumerated_checks=checks,maximum_decomposition_error=maxerr,CPU_wall_seconds=time.time()-beg));print('SEALED64',checks,maxerr,flush=True)

def avg(values):
 a=[float(v)for v in values if v is not None];return float(np.mean(a))if a else None

def ratio_bin(x):return '<0.5'if x<.5 else '0.5-1'if x<1 else '1-2'if x<2 else '2+'

def score():
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 p=verify();bar=read(D/'GLOBAL_READOUT_BARRIER.json');assert bar['count']==64 and not bar['GT_read'];zs=[]
 for f in bar['files']:zs.append(load(ROOT/f['path']))
 features=ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json';assert sha(features)==read(D/'RUNTIME_LOCK.json')['existing_feature_sha256'];fb={d:{r['query_ordinal']:r for r in v}for d,v in read(features)['rows'].items()};plans={d:read(ROOT/f'artifacts/decota_paper_experiments_v1/{d}/PLAN.json')['rows']for d in ['hc2','vidstg']};targets={d:truths(d,dict(parents=[z['query_ordinal']for z in zs if z['dataset']==d]))for d in ['hc2','vidstg']};rows=[];checks=0
 old=read(ROOT/'results/stvg_evidence_strata/2026-10-18/cached_full/TEMPORAL64.json');oldrows={(r['dataset'],r['query_alias'],r['geometry'],r['support']):r for r in old['rows']}
 for z in zs:
  ds,q=z['dataset'],z['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];feat=fb[ds][q];duration=row['frame_ids'][-1]+1-row['frame_ids'][0];physical_gt=feat['event_physical_frames']
  for geo in GEOMETRIES:
   boxes=z['boxes'][geo].numpy();tube=DenseTube(boxes,row,gt,span,clip=ds=='hc2')
   for sup in SUPPORTS:
    ivs=z['fused_intervals']if sup=='fused'else{a:v[int(sup[-1])]for a,v in z['offset_intervals'].items()};native=ivs['Native'];nm=(tube.fids>=native[0])&(tube.fids<native[1]);vals={}
    for a,iv in ivs.items():
     m=official(boxes,row,gt,span,iv,ds);assert max(abs(m[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10;checks+=3
     now=(tube.fids>=iv[0])&(tube.fids<iv[1]);add=now&~nm;lost=nm&~now;overlap=max(0,min(iv[1],native[1])-max(iv[0],native[0]));added=iv[1]-iv[0]-overlap;lostframes=native[1]-native[0]-overlap
     vals[a]=dict(**m,R03=float(m['v']>.3),R05=float(m['v']>.5),pred_to_GT_physical_length_ratio=(iv[1]-iv[0])/physical_gt,pred_to_GT_metric_span_ratio=(iv[1]-iv[0])/(span[1]-span[0]),pred_input_span_fraction=(iv[1]-iv[0])/duration,added_known_frames=int(add.sum()),added_known_IoU=avg(tube.iou[add]),added_unknown_frames=int(added-add.sum()),lost_known_frames=int(lost.sum()),lost_known_IoU=avg(tube.iou[lost]),lost_unknown_frames=int(lostframes-lost.sum()))
     assert added>=int(add.sum())and lostframes>=int(lost.sum())
     if a in ['Native','full_projection','native_MBR']:
      og='Frozen'if geo=='Source'else geo;oa='projection'if a=='full_projection'else a;ref=oldrows[(ds,feat['query_alias'],og,sup)]['values'][oa]
      for k in ['v','t','s']:assert abs(m[k]-ref[k])<1e-12;checks+=1
    assert len({v['s']for v in vals.values()})==1
    rows.append(dict(dataset=ds,alias=z['alias'],query_alias=feat['query_alias'],parent_alias=feat['parent_alias'],geometry=geo,support=sup,event_length_bin=feat['bins']['event_fraction'],event_start_bin=feat['bins']['event_start'],values=vals))
  print('SCORED',z['alias'],flush=True)
 put(D/'POSTSEAL_SCORED_PRIVATE.json',dict(rows=rows,GT_read=True,GT_after_barrier_sha256=sha(D/'GLOBAL_READOUT_BARRIER.json')))
 def summary(rr):
  out={}
  for a in ARMS:
   parent=collections.defaultdict(list)
   for r in rr:parent[r['parent_alias']].append(r['values'][a])
   keys=list(rr[0]['values'][a]);out[a]=dict(queries=len(rr),parents=len(parent),known_queries={k:sum(r['values'][a][k]is not None for r in rr)for k in keys},query_mean={k:avg([r['values'][a][k]for r in rr])for k in keys},parent_mean={k:avg([avg([v[k]for v in vs])for vs in parent.values()])for k in keys})
  return out
 pairs=[(a,'Native')for a in ARMS[1:]]+[('full_projection','length_only'),('full_projection','fluctuation_only'),('length_only','fluctuation_only')]
 def paired(rr):
  out={};ps=sorted({r['parent_alias']for r in rr});idx=np.random.default_rng(20261018).integers(len(ps),size=(10000,len(ps)))
  for a,b in pairs:
   out[a+'__minus__'+b]={}
   for k in ['v','t','s','R03','R05']:
    y=np.array([avg([r['values'][a][k]-r['values'][b][k]for r in rr if r['parent_alias']==p])for p in ps]);out[a+'__minus__'+b][k]=dict(mean=float(y.mean()),CI95=np.quantile(y[idx].mean(1),[.025,.975]).tolist(),parents=len(ps),positive=int((y>1e-10).sum()),negative=int((y< -1e-10).sum()),neutral=int((abs(y)<=1e-10).sum()),gross_gain=float(np.maximum(y,0).mean()),gross_loss=float(-np.minimum(y,0).mean()))
  return out
 groups={}
 for ds in ['hc2','vidstg']:
  groups[ds]={}
  for geo in GEOMETRIES:
   groups[ds][geo]={}
   for sup in SUPPORTS:
    rr=[r for r in rows if(r['dataset'],r['geometry'],r['support'])==(ds,geo,sup)];groups[ds][geo][sup]=dict(overall=summary(rr),paired=paired(rr),strata={key:{v:summary([r for r in rr if r[key]==v])for v in sorted({r[key]for r in rr})}for key in ['event_length_bin','event_start_bin']})
 public=[]
 for r in rows:
  v={}
  for a,m in r['values'].items():
   v[a]={k:x for k,x in m.items()if k not in ['pred_to_GT_physical_length_ratio','pred_to_GT_metric_span_ratio']};v[a]['pred_to_GT_physical_length_ratio_bin']=ratio_bin(m['pred_to_GT_physical_length_ratio']);v[a]['delta_v_vs_Native']=m['v']-r['values']['Native']['v'];v[a]['delta_t_vs_Native']=m['t']-r['values']['Native']['t']
  public.append(dict(**{k:x for k,x in r.items()if k!='values'},values=v))
 for ds in ['hc2','vidstg']:put(P/f'RESULTS_{ds}.json',dict(status='complete',count=32,primary_geometry='After',groups=groups[ds],rows=[r for r in public if r['dataset']==ds],historically_exposed=True,all_predictions_sealed_before_GT=True,exact_GT_lengths_and_coordinates_private=True))
 put(P/'VERIFICATION.json',dict(status='pass',count=64,independent_official_and_reference_checks=checks,enumerated_direct_score_checks=bar['enumerated_checks'],maximum_decomposition_error=bar['maximum_decomposition_error'],old_Native_projection_MBR_exact=True,sIoU_invariant=True,new_model_calls=0,GPU_seconds=0,no_parameter_fits=True))
 put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True,new_model_calls=0,GPU_seconds=0));print('SCORE_COMPLETE',checks,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
