"""Frozen-source calibration and independent audit, no target selection.

Controls are cached feature interventions, sealed before this module reads any
source labels. The original acquisition and matching implementation stay frozen.
"""
from source_validation import *
K=C/'calibration'
ARMS=['Source','PureExpert']+[f'Config{i}'for i in range(4)]
CONTROL_NAMES=['UniformFeatures','FixedTimePermutation']
def prepare_calibration():
 assert not(K/'plan.json').exists();p=source_plan()
 z=dict(schema='source36_validation_controls_calibration_audit_v1',source_plan_fingerprint=p['fingerprint'],
  objective='Per direction, query-equal mean box IoU on GT-supported native unobserved frames of only that direction source-domain validation36. Missing support excluded and counted, never0. Max over the original4configs, exact ties original config order. Not source-training36; these videos were exposed as target in opposite direction. No averaging directions or reading current target scores.',
  controls='Uniform all-one vectors and SHA256 fixed temporal permutation of entire feature grids AND valid masks, all4configs; fixed masks/physical positions/boxes unchanged. Identity permutation rotated. Controls have no new hyperparameter and are NOT candidates. All72 controls seal before source labels. Report array-change and feature-effect, do not force superiority.',
  audit='Independent process rereads each input/native/pred/control SHA; cold recomputes all4configs+controls; separate normalized-coordinate IoU formula agrees with official pixel IoU on every scored frame. Reconstruct all means/selection, parentCI, support and empty counts. Root must inspect source functioning and explicitly accept candidate before target plan. No automatic target launch.',
  max_new_Source=0,max_new_expert=0,max_new_optimizer=0,
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'source_validation.py',S/'matching.py',S/'cpu.py',R/'stvg/metrics_base.py']},
  input_sha256={str(f.relative_to(R)):sha(f)for f in [C/'plan.json',D/'source_validation_cohort_v1.json',RUN/'plan.json']})
 z['fingerprint']=digest(z);atomic(K/'plan.json',z);atomic(P/'source_calibration_protocol.json',z);print(z['fingerprint'],flush=True)
def check_calibration():
 p=read(K/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def permutation(target,key,n):
 pp=sorted(range(n),key=lambda i:(hashlib.sha256(f'20261103|source_validation|{target}|{key}|{i}'.encode()).digest(),i))
 if n>1 and pp==list(range(n)):pp=pp[1:]+pp[:1]
 return pp
def control_arrays(target,key,v):
 out={};proof={};n=len(v['frame_ids']);perm=permutation(target,key,n)
 for name in CONTROL_NAMES:
  w=dict(v)
  if name=='UniformFeatures':w['features']=torch.ones_like(v['features'])
  else:w['features']=v['features'][perm].clone();w['valid']=v['valid'][perm].clone()
  for j,cfg in enumerate(CONFIGS):
   arm=f'{name}_Config{j}';out[arm],proof[arm]=matching.propagate(w['features'],w['valid'],w['positions8'],w['seeds'],w['native_boxes'],w['shape'],w['image_size'],**cfg)
 return out,dict(permutation=perm,diagnostics=proof,GT_read=False)
def seals():
 state=read(C/'state.json');assert state['status']=='all72_source_features_fourconfig_predictions_sealed_pending_source_only_calibration'
 out={}
 for target in COUNTS:
  out[target]=read(C/target/'sealed.json');assert out[target]['queries']==36 and not out[target]['GT_read']
 return out
def controls():
 plan=check_calibration();parent.infer_guard();torch.set_num_threads(1);ss=seals();assert not(K/'controls_sealed.json').exists();files={};start=time.perf_counter()
 for target,seal in ss.items():
  for key,e in seal['files'].items():
   parent.resources(start);assert sha(C/e['input_path'])==e['input_sha256'];v=torch.load(C/e['input_path'],map_location='cpu',weights_only=False,mmap=True);a,proof=control_arrays(target,key,v);again,_=control_arrays(target,key,v);assert all(np.array_equal(x,again[k])for k,x in a.items());out=K/'controls'/target/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
   with out.open('wb')as f:np.savez_compressed(f,**a)
   with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(x,z[k])for k,x in a.items())
   rp=out.with_suffix('.json');atomic(rp,proof);files[target+'/'+key]=dict(path=str(out.relative_to(K)),sha256=sha(out),receipt_path=str(rp.relative_to(K)),receipt_sha256=sha(rp),input_sha256=e['input_sha256'])
   atomic(K/'controls_progress.json',dict(completed=len(files),total=72,GT_read=False,worker=identity()))
 atomic(K/'controls_sealed.json',dict(files=files,queries=72,GT_read=False,plan_fingerprint=plan['fingerprint'],source_seal_sha256={t:sha(C/t/'sealed.json')for t in COUNTS},worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print('source72 controls sealed noGT',flush=True)
def ious_pixel(boxes,metadata,gt,ids,positions):
 from stvg.metrics_base import xyxy
 from stvg.metrics import box_iou
 pixels=xyxy(boxes,metadata['width'],metadata['height']);pixels=np.maximum(pixels,0)if metadata['dataset']=='hc2'else pixels;obs=set(map(int,positions));ind=[i for i,f in enumerate(ids)if int(f)in gt and i not in obs]
 return ind,np.array([box_iou(pixels[i],gt[int(ids[i])])for i in ind],dtype=float)
def ious_independent(boxes,metadata,gt,ids,positions):
 # Independent explicit center-size->corners, pixel-scale and intersection/union.
 wh=np.array([metadata['width'],metadata['height']],float);obs=set(map(int,positions));ind=[i for i,f in enumerate(ids)if int(f)in gt and i not in obs];ans=[]
 for i in ind:
  c=np.asarray(boxes[i,:2],float)*wh;s=np.asarray(boxes[i,2:],float)*wh;a=np.r_[c-s/2,c+s/2]
  if metadata['dataset']=='hc2':a=np.maximum(a,0)
  b=np.asarray(gt[int(ids[i])],float);ii=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();den=aa+bb-ii;ans.append(float(ii/den)if den>0 else 0.)
 return ind,np.asarray(ans)
def derive_scores(independent=False):
 check_calibration();ss=seals();cs=read(K/'controls_sealed.json');assert cs['queries']==72 and not cs['GT_read'];cohort=read(D/'source_validation_cohort_v1.json');rr=[];maxerr=0.;frames=0
 for target,seal in ss.items():
  ds=seal['source_domain'];truth=SEL.labels(ds);records={r['id']:r for r in cohort[target]['rows']};assert set(records)==set(seal['files'])
  for key,e in seal['files'].items():
   ce=cs['files'][target+'/'+key]
   for base,entry,pairs in [(C,e,[('path','sha256'),('input_path','input_sha256'),('native_path','native_sha256'),('receipt_path','receipt_sha256')]),(K,ce,[('path','sha256'),('receipt_path','receipt_sha256')])]:
    for f,h in pairs:assert sha(base/entry[f])==entry[h]
   row=records[key];meta=dict(row['metadata'],dataset=ds);g=truth[key];gt={int(f):b for f,b in g['boxes'].items()};vals={};changed={}
   with np.load(C/e['path'])as z,np.load(K/ce['path'])as ctr:
    if independent:
     v=torch.load(C/e['input_path'],map_location='cpu',weights_only=False,mmap=True);a,_=match_arrays(v);b,_=control_arrays(target,key,v)
     assert set(a)==set(z.files)and all(np.array_equal(x,z[k])for k,x in a.items());assert set(b)==set(ctr.files)and all(np.array_equal(x,ctr[k])for k,x in b.items())
    for arm in ARMS+list(ctr.files):
     boxes=z[arm]if arm in ARMS else ctr[arm];ids=z['frame_ids'];pos=z['positions8'];ind,vv=ious_pixel(boxes,meta,gt,ids,pos)
     if independent:
      jj,uu=ious_independent(boxes,meta,gt,ids,pos);assert ind==jj;err=float(np.max(np.abs(vv-uu)))if len(vv)else 0.;assert err<1e-12;maxerr=max(maxerr,err)
     frames+=len(vv);vals[arm]=dict(unobserved_IoU_percent=100*float(vv.mean())if len(vv)else None,annotated_unobserved_frames=len(ind))
     if arm not in ARMS:changed[arm]=int(np.any(boxes!=z[arm.split('_',1)[1]],axis=1).sum())
   rr.append(dict(target=target,source_domain=ds,id=key,parent=str(g['parent']),values=vals,control_changed_boxes=changed,empty_counts={f'Config{j}':e['diagnostics'][f'Config{j}']['empty_unobserved_masks']for j in range(4)}))
 return rr,dict(scored_frame_arm_checks=frames,max_independent_IoU_error=maxerr)
def report_for(rr):
 result={}
 for target in COUNTS:
  qq=[r for r in rr if r['target']==target];armnames=list(qq[0]['values']);means={a:stats([dict(parent=r['parent'],value=r['values'][a]['unobserved_IoU_percent'])for r in qq if r['values'][a]['unobserved_IoU_percent']is not None],['value'])for a in armnames};scores=[means[f'Config{i}']['statistics']['value']['mean']for i in range(4)];assert all(v is not None for v in scores);best=max(range(4),key=lambda i:(scores[i],-i));pairs={}
  for a,b in [(f'Config{best}','PureExpert'),(f'Config{best}','Source')]+[(f'Config{best}',f'{c}_Config{best}')for c in CONTROL_NAMES]:
   paired=[dict(parent=r['parent'],delta=r['values'][a]['unobserved_IoU_percent']-r['values'][b]['unobserved_IoU_percent'])for r in qq if r['values'][a]['unobserved_IoU_percent']is not None and r['values'][b]['unobserved_IoU_percent']is not None];pairs[a+'-'+b]=stats(paired,['delta'])
  result[target]=dict(source_domain=qq[0]['source_domain'],queries=len(qq),means=means,selected_candidate=f'Config{best}',config=CONFIGS[best],paired=pairs,empty_unobserved_masks={f'Config{j}':sum(r['empty_counts'][f'Config{j}']for r in qq)for j in range(4)},control_changed_frames={c:sum(r['control_changed_boxes'][c]for r in qq)for c in qq[0]['control_changed_boxes']},target_release=False)
 return result
def score():
 check_calibration();parent.infer_guard();torch.set_num_threads(1);assert not(K/'score_receipt.json').exists();rr,check=derive_scores();rep=report_for(rr);atomic(K/'private_rows.json',rr);atomic(P/'source_validation'/'report.json',rep);atomic(K/'score_receipt.json',dict(status='source_only_scores_config_candidates_pending_independent_audit_and_ROOT_review',controls_seal_sha256=sha(K/'controls_sealed.json'),private_rows_sha256=sha(K/'private_rows.json'),report_sha256=sha(P/'source_validation'/'report.json'),worker=identity(),checks=check));print('source calibration scored, target NOT released',flush=True)
def audit():
 plan=check_calibration();parent.infer_guard();torch.set_num_threads(1);sc=read(K/'score_receipt.json');assert sc['controls_seal_sha256']==sha(K/'controls_sealed.json')and sc['private_rows_sha256']==sha(K/'private_rows.json');rr,checks=derive_scores(True);assert rr==read(K/'private_rows.json');rep=report_for(rr);assert rep==read(P/'source_validation'/'report.json');ack=dict(status='independent_source72_allarrays_SHA_pixel_IoU_means_controls_selection_verified_PENDING_ROOT_acceptance',source_plan_fingerprint=plan['source_plan_fingerprint'],report_sha256=sha(P/'source_validation'/'report.json'),controls_seal_sha256=sha(K/'controls_sealed.json'),worker=identity(),checks=checks,target_release=False);atomic(K/'audit_receipt.json',ack);atomic(P/'source_validation'/'audit_receipt.json',ack);print('source calibration independent audit passed; ROOT decision required',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare_calibration','controls','score','audit']);x=ap.parse_args();globals()[x.stage]()
