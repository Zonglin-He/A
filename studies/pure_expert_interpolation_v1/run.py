"""Pure eight-expert interpolation: no Source/E input to the spatial estimator."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'))
import common_v1 as parent
import numpy as np,torch
read=parent.read;sha=parent.sha;atomic=parent.atomic;digest=parent.digest;identity=parent.identity
D=R/'outputs/pure_expert_interpolation_v1';P=R/'results/pure_expert_interpolation_v1';OLD=R/'outputs/review_strengthening_v1';RUN=parent.selected.RUN
ARMS=['Source','OriginalHybrid','PureExpert','Current'];PAIRS=[('Current','PureExpert'),('Current','OriginalHybrid'),('PureExpert','OriginalHybrid')]

def pure_interpolation(ids,positions,native_boxes):
 times=np.asarray(ids,dtype=np.float64);pos=np.asarray(positions,dtype=np.int64);boxes=np.asarray(native_boxes,dtype=np.float64)
 assert len(pos)==8 and len(set(pos))==8 and np.all(np.diff(pos)>0) and np.all(np.diff(times)>0)
 assert boxes.shape==(8,4) and np.isfinite(boxes).all() and (boxes[:,2:]>0).all()
 result=np.repeat(boxes[:1],len(times),axis=0);result[pos[-1]+1:]=boxes[-1]
 for j in range(7):
  a,b=int(pos[j]),int(pos[j+1]);w=((times[a:b+1]-times[a])/(times[b]-times[a]))[:,None]
  result[a:b+1,:2]=(1-w)*boxes[j,:2]+w*boxes[j+1,:2]
  result[a:b+1,2:]=np.exp((1-w)*np.log(boxes[j,2:])+w*np.log(boxes[j+1,2:]))
 # Preserve every actual observed expert box exactly, including log/exp endpoints.
 result[pos]=boxes
 assert np.array_equal(result[pos],boxes) and np.array_equal(result[:pos[0]],np.repeat(boxes[:1],pos[0],axis=0))
 assert np.array_equal(result[pos[-1]+1:],np.repeat(boxes[-1:],len(times)-pos[-1]-1,axis=0))
 return result

def rows(ds):return read(D/'selection.json')[ds]
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p

def prepare():
 assert not(D/'plan.json').exists();sel=R/'outputs/matched_pipeline_timing_v1/selection.json';atomic(D/'selection.json',read(sel));inputs=[sel,D/'selection.json',RUN/'plan.json',R/'outputs/reviewer_key_controls_v1/interpolation_definition_audit_20261103/read_only_fixed128_tail_audit.json'];inventory={}
 for ds in ['hc2','vidstg']:
  inputs += [RUN/ds/'sealed.json',OLD/'matched_cpu'/ds/'sealed.json',OLD/'matched_cpu'/ds/'metrics.json',OLD/'matched_cpu'/ds/'reporting_receipt.json']
  ss=read(RUN/ds/'sealed.json');ms=read(OLD/'matched_cpu'/ds/'sealed.json');inventory[ds]={};rr=rows(ds);assert len(rr)==128 and len({r['metadata']['original_video_id']for r in rr})==128
  for row in rr:
   key=row['id'];e=ss['files'][key];m=ms['files'][key];assert sha(RUN/ds/e['path'])==e['sha256'] and sha(OLD/m['path'])==m['sha256']
   with np.load(RUN/ds/e['path'])as z:assert z['native_boxes'].shape==(8,4) and np.isfinite(z['native_boxes']).all() and (z['native_boxes'][:,2:]>0).all()
   inventory[ds][key]=dict(parent=e,matched=m)
 atomic(D/'inventory.json',inventory);inputs += [D/'inventory.json']
 p=dict(schema='pure_expert_interpolation_fixed128_v1',authorization='2026-11-03 user rejects additions to pure expert interpolation; correction of baseline definition, unchanged prior fixed128/domain panel; no GPU or full expansion',
  baseline='Only original eight expert boxes, physical observation times and requested frame times. Linear center/log-size interpolation between consecutive anchors; outside anchor span keep nearest endpoint expert box. Overwrite anchors exactly. No Source/E/shared correction/features/DP/temporal refinement/model. All256 have eight valid boxes; no invalid-box fallback or skipped query.',
  temporal='Same existing audited Current interval for Source boxes, old hybrid, pure interpolation and Current. Temporal conditioning is common evaluation support, not a temporal update inside PureExpert. This is NOT a standalone native-interval pipeline ranking.',
  original='Old interpolation retained E outside anchor span. Old published full gaps +2.134753 HC/-0.106955 Vid concern that hybrid, not this pure baseline. Preserve all old source/plans/predictions/scores/negatives.',
  score='Whole128 seal before separate official CPU score of PureExpert only, reuse all audited matched-time Source/old-hybrid/Current scalars; separate all arrays/SHA/input/means audit. No new GT acquisition, no new model/Source/expert/optimizer/media. Fixed previously exposed panel, not unknown validation or outcome-selected subset.',
  counts={'hc2':128,'vidstg':128},new_GPU=0,new_Source=0,new_expert=0,new_optimizer=0,new_model=0,
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'controller.py',R/'studies/reviewer_key_controls_v1/common_v1.py',R/'stvg/paper_panel.py',R/'tools/full_score.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',p);print(p['fingerprint'],flush=True)

def derive(ds,row):
 key=row['id'];inv=read(D/'inventory.json')[ds][key];e,m=inv['parent'],inv['matched'];assert sha(RUN/ds/e['path'])==e['sha256'] and sha(OLD/m['path'])==m['sha256']
 with np.load(RUN/ds/e['path'])as z,np.load(OLD/m['path'])as old:
  ids=z['frame_ids'];pos=z['positions8'];nb=z['native_boxes'];pure=pure_interpolation(ids,pos,nb);iv=old['Ours_interval'].copy();assert np.array_equal(iv,old['ExpertInterpolation_interval'])
  arrays=dict(frame_ids=ids.copy(),positions8=pos.copy(),Source=old['SourceMatchedTime'].copy(),OriginalHybrid=old['ExpertInterpolation'].copy(),PureExpert=pure,Current=old['Ours'].copy(),expert_boxes=nb.copy())
  for arm in ARMS:arrays[arm+'_interval']=iv.copy()
  tail=(np.arange(len(ids))<pos[0])|(np.arange(len(ids))>pos[-1]);event=(ids>=iv[0])&(ids<iv[1]);diff=np.any(pure!=old['ExpertInterpolation'],axis=1)
  assert np.array_equal(old['ExpertInterpolation'][tail],z['SharedInit'][tail])
  diag=dict(nodes=len(ids),tail_nodes=int(tail.sum()),event_nodes=int(event.sum()),tail_event_nodes=int((tail&event).sum()),different_tail_nodes=int((diff&tail).sum()),different_tail_event_nodes=int((diff&tail&event).sum()),interior_maximum_roundoff=float(np.max(np.abs(pure[~tail]-old['ExpertInterpolation'][~tail]))),all8_expert_boxes_exact=True,no_Source_or_E_input_to_interpolator=True)
 return arrays,dict(parent_prediction_sha256=e['sha256'],matched_prediction_sha256=m['sha256'],input_SHA_checked=True),diag

def preflight():
 validate();parent.infer_guard();torch.set_num_threads(1);cases=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   a,p,d=derive(ds,row);b,q,e=derive(ds,row);assert set(a)==set(b)and all(np.array_equal(v,b[k])for k,v in a.items())and p==q and d==e;cases.append(dict(dataset=ds,all_arrays_repeat_exact=True,all8_anchor_boxes_exact=True,only_expert_time_inputs=True))
 atomic(D/'preflight.json',dict(status='passed_four_real_cached_cases',cases=cases,new_GPU=0,new_model=0,new_GT=0));print('preflight passed',flush=True)

def predict(ds):
 plan=validate();parent.infer_guard();torch.set_num_threads(1);assert read(D/'preflight.json')['status']=='passed_four_real_cached_cases';assert not(D/ds/'sealed.json').exists();files={};start=time.perf_counter()
 for i,row in enumerate(rows(ds),1):
  parent.resources(start);a,proof,diag=derive(ds,row);key=row['id'];out=D/ds/'predictions'/(key+'.npz');rp=D/ds/'receipts'/(key+'.json');out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('wb')as f:np.savez_compressed(f,**a)
  with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  e=dict(path=str(out.relative_to(D)),sha256=sha(out),proof=proof,diagnostics=diag,GT_read=False,worker=identity());atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));atomic(D/ds/'progress.json',dict(completed=i,total=128,GT_read=False))
 atomic(D/ds/'sealed.json',dict(queries=128,files=files,GT_read=False,plan_fingerprint=plan['fingerprint'],elapsed_wall_seconds=time.perf_counter()-start,worker=identity()));print(ds,'sealed',flush=True)

def score(ds):
 validate();torch.set_num_threads(1);seal=read(D/ds/'sealed.json');assert seal['queries']==128 and not seal['GT_read'];ack=read(OLD/'matched_cpu'/ds/'reporting_receipt.json');assert ack['all_temporal_IoUs_identical'] and ack['all_arrays_cold_rederived_exact'];prior={r['id']:r for r in read(OLD/'matched_cpu'/ds/'metrics.json')['private_rows']};truth=parent.selected.labels(ds);rr=[];mx=0.;checks=0
 from full_score import metric
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];old=prior[key];v=dict(Source=old['values']['SourceMatchedTime'],OriginalHybrid=old['values']['ExpertInterpolation'],Current=old['values']['Ours'])
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}};v['PureExpert'],err=metric(z['PureExpert'],inp,{int(k):b for k,b in g['boxes'].items()},g['interval'],z['PureExpert_interval'].tolist(),z['positions8']);mx=max(mx,err);checks+=1
  assert len({a['t']for a in v.values()})==1;rr.append(dict(id=key,parent=old['parent'],duration_group=old['duration_group'],values=v))
 assert len({r['parent']for r in rr})==128;atomic(D/ds/'metrics.json',dict(private_rows=rr));parent.selected.suite().parent_report(rr,ARMS,PAIRS,P/ds);parent.selected.suite().signed_tails(rr,PAIRS,P/ds)
 atomic(D/ds/'score_receipt.json',dict(status='scored_whole128_pure_only',new_official_checks=checks,max_discrepancy=mx,all_tIoU_equal=True,seal_sha256=sha(D/ds/'sealed.json'),metrics_sha256=sha(D/ds/'metrics.json'),worker=identity()));print(ds,'scored',flush=True)

def audit(ds):
 validate();parent.infer_guard();torch.set_num_threads(1);seal=read(D/ds/'sealed.json');sc=read(D/ds/'score_receipt.json');assert sc['seal_sha256']==sha(D/ds/'sealed.json')and sc['metrics_sha256']==sha(D/ds/'metrics.json')
 for row in rows(ds):
  a,p,d=derive(ds,row);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']and p==e['proof']and d==e['diagnostics']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
 rr=read(D/ds/'metrics.json')['private_rows'];rep=read(P/ds/'report.json')
 for arm,m in rep['means'].items():
  for k,v in m.items():
   vv=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert (v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 ack=dict(status='verified_pure_expert128_all_arrays_SHA_means',queries=128,parents=128,all_arrays_cold_exact=True,all_tIoU_equal=True,report_sha256=sha(P/ds/'report.json'),score_sha256=sha(D/ds/'score_receipt.json'),new_model=0,new_GPU=0,new_Source=0,new_expert=0,worker=identity());atomic(D/ds/'reporting_receipt.json',ack);atomic(P/ds/'reporting_receipt.json',ack);print(ds,'audited',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','predict','score','audit']);ap.add_argument('--dataset',choices=['hc2','vidstg']);x=ap.parse_args();globals()[x.stage](*([]if x.stage in ['prepare','preflight']else[x.dataset]))
