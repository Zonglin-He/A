"""R53 configurable one-pass post-seal scorer; UNEXECUTED schema draft.

Only a root scoring grant plus independent prediction audit may unlock GT.
Original 256 source boxes are immutable. Existing source/R50-wideG task
scalars are joined from sealed old tables instead of being GT-scored again.
"""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, itertools, json, math, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PLATFORMS=('drone','quad')
IDENTITY_FIELDS=('imagepath','pointpath','metapath','caption','platform','scene')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def local(s):
 p=Path(s);assert not p.is_absolute() and '..' not in p.parts
 assert 'round46/final' not in s and 'FINAL384' not in s
 return ROOT/p

def identity(r):return tuple(r[k] for k in IDENTITY_FIELDS)
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def summarize(tasks,arms,refs,n):
 scopes=[('ALL'+str(n),tasks)]
 if n==5912:scopes += [(part,[r for r in tasks if r['part']==part]) for part in ('DEV336','NONDEV5576')]
 else:
  for split in sorted({r['split'] for r in tasks}):
   if split!='ALL336':scopes.append((split,[r for r in tasks if r['split']==split]))
 summary={};scenes=[]
 for name,rows in scopes:
  assert rows;ss={}
  for pl in PLATFORMS:
   rr=[r for r in rows if r['platform']==pl];assert rr
   ss[pl]={}
   for arm in arms:
    v=np.asarray([r['IoU'][arm] for r in rr],dtype=np.float64)
    record={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
    for ref in refs:
     d=v-np.asarray([r['IoU'][ref] for r in rr]);record.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improved_vs_{ref}':int((d>1e-10).sum()),f'worsened_vs_{ref}':int((d< -1e-10).sum()),f'query_changes_vs_{ref}':sum(r['queries'][arm]!=r['queries'][ref] for r in rr)})
    for status in ('unknown','fallback','no_parameter_update'):record[status+'_n']=sum(r['statuses'][arm][status] for r in rr)
    ss[pl][arm]=record
   for scene in sorted({r['scene_PRIVATE'] for r in rr}):
    sr=[r for r in rr if r['scene_PRIVATE']==scene]
    scenes.append({'scope':name,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{arm:float(np.mean([r['IoU'][arm] for r in sr])*100) for arm in arms}})
  ss['balanced_mIoU_percent']={arm:float(np.mean([ss[pl][arm]['mIoU_percent'] for pl in PLATFORMS])) for arm in arms}
  ss['micro_mIoU_percent']={arm:float(np.mean([r['IoU'][arm] for r in rows])*100) for arm in arms}
  summary[name]=ss
 return summary,scenes

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):ap.add_argument('--'+k,required=True)
 a=ap.parse_args();auth=read(a.authorization);plan=read(a.plan)
 assert auth['state']=='ROOT_AUTHORIZED_R53_SINGLE_POSTSEAL_TASK_SCORE'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(a.plan)
 assert plan['n'] in (336,5912) and plan['no_protected_FINAL_specific_artifacts'] is True
 pins={**plan['programs_sha256'],**plan['inputs_sha256']}
 for rel,h in pins.items():assert sha(local(rel))==h,('pin',rel)
 def get(rel):assert rel in pins;return read(local(rel))
 arms=plan['arm_names'];assert arms and len(arms)==len(set(arms));n=plan['n'];refs=plan['reference_arms'];assert set(refs)<=set(arms)
 reused=plan['reused_arms'];assert set(reused)<=set(arms)
 scored=[arm for arm in arms if arm not in reused];assert scored,'No new arms: use saved-scalar reaggregation, not this GT scorer'
 if n==5912:assert plan.get('calibration') is None,'Full cohort cannot retune calibration'
 calibration=plan.get('calibration')
 if calibration is not None:
  assert n==336 and calibration['factor_values']==[.5,1.,2.] and calibration['shared_across_platforms'] is True
  assert set(calibration['target_arm_by_factor'])=={'0.5','1','2'}
  assert set(calibration['target_arm_by_factor'].values())<=set(arms)
  assert sorted(calibration['tie_priority'])==[.5,1.,2.]
 pred=local(plan['prediction_directory']);seal=get(plan['prediction_seal'])
 assert local(plan['prediction_seal'])==pred/'SEALED.json'
 assert auth['prediction_seal_sha256']==sha(pred/'SEALED.json')
 assert seal['n']==n and seal['arm_names']==arms
 for rel,h in seal['files'].items():
  rp=Path(rel);assert not rp.is_absolute() and '..' not in rp.parts and sha(pred/rp)==h
 audit=get(plan['independent_prediction_audit'])
 assert auth['independent_prediction_audit_sha256']==sha(local(plan['independent_prediction_audit']))
 assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(pred/'SEALED.json')
 assert audit['n']==n and audit['arm_names']==arms and audit['canonical_source_256_boxes_and_readout_verified'] is True
 assert audit['all_unknown_fallback_and_zero_update_rows_retained'] is True
 out=local(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 start=time.perf_counter();tasks=[]
 try:
  # Cohort/P5/GT decoding starts only after all grant, seal and audit gates.
  rows=get(plan['cohort_registration']);assert len(rows)==n and len({r['id'] for r in rows})==n and len({identity(r) for r in rows})==n
  byid={r['id']:r for r in rows};counter=collections.Counter(r['platform'] for r in rows)
  assert counter==({'drone':168,'quad':168} if n==336 else {'drone':2984,'quad':2928})
  statusrows=get(plan['readout_status_table']);assert len(statusrows)==n and len({r['id'] for r in statusrows})==n
  status={r['id']:r['arm_statuses'] for r in statusrows};assert set(status)==set(byid)
  for ss in status.values():
   assert set(ss)==set(arms)
   for x in ss.values():assert set(x)=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in x.values())
  dev=get(plan['dev_registration']);assert len(dev)==336
  devkeys={identity(r) for r in dev};assert len(devkeys)==336
  if n==5912:assert devkeys<={identity(r) for r in rows}
  else:assert devkeys=={identity(r) for r in rows}
  splits=get(plan['dev_split_by_id']);assert set(splits)=={r['id'] for r in dev}
  oldid_by_key={identity(r):r['id'] for r in dev}
  cached={}
  for arm,spec in reused.items():
   reg=get(spec['registration']);old=get(spec['saved_rows']);oldbyid={r['id_PRIVATE']:r for r in old}
   assert len(oldbyid)==len(old);rr={identity(r):r for r in reg};assert len(rr)==len(reg)
   assert {identity(r) for r in rows}<=set(rr)
   cached[arm]={}
   for r in rows:
    oldr=oldbyid[rr[identity(r)]['id']];oldarm=spec['saved_arm']
    assert oldr['platform']==r['platform'] and oldr['scene_PRIVATE']==r['scene']
    val=oldr['IoU'][oldarm];q=oldr['queries'][oldarm]
    assert math.isfinite(val) and 0<=val<=1 and type(q) is int and 0<=q<256
    cached[arm][r['id']]=(val,q)
  # Pin original independent core and original P5 object-index metadata join.
  corepath=local(plan['independent_geometry_core']);assert plan['independent_geometry_core']=='research/round5/audit/independent_core_v2.py'
  assert plan['independent_geometry_core'] in pins and 'tools/obb_iou.py' in pins
  spec=importlib.util.spec_from_file_location('R53_PINNED_ORIGINAL_INDEPENDENT_CORE',corepath);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
  import ijson
  p5=local(plan['scoring_index']);pr=get(plan['scoring_index_preparation_receipt']);assert sha(p5)==pr['scoring_index_sha256']
  idx={}
  with p5.open('rb') as f:
   for ix in ijson.items(f,'item'):
    if ix['id'] not in byid:continue
    r=byid[ix['id']];assert ix['id'] not in idx and ix['platform']==r['platform'] and ix['scene']==r['scene']
    assert Path(ix['meta_path']).resolve()==Path(r['metapath']).resolve() and type(ix['object_index']) is int and ix['object_index']>=0
    idx[ix['id']]={k:ix[k] for k in ('meta_path','object_index','metadata_container_sha256')}
  assert set(idx)==set(byid);containers={}
  for i,r in enumerate(rows):
   rel=plan['sample_relative_template'].format(id=r['id']);assert rel in seal['files']
   z=load(pred/rel);assert z['arm_names'].tolist()==arms
   assert z['source_boxes'].shape==(256,6) and np.isfinite(z['source_boxes']).all()
   q=z['queries'];selected=z['selected_original_boxes']
   assert q.shape==(len(arms),) and np.issubdtype(q.dtype,np.integer) and ((0<=q)&(q<256)).all()
   assert selected.shape==(len(arms),6) and exact(selected,z['source_boxes'][q])
   values={};qmap=dict(zip(arms,map(int,q)))
   for arm in reused:
    value,oldq=cached[arm][r['id']];assert qmap[arm]==oldq,('reused_endpoint_query_changed',i,arm)
    values[arm]=value
   ix=idx[r['id']];mp=ix['meta_path']
   if mp not in containers:assert sha(mp)==ix['metadata_container_sha256'];containers[mp]=read(mp)
   obj=containers[mp]['ground_info'][ix['object_index']]
   assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption']
   gt=core.transform_gt(obj['bbox_3d'],r['platform']);boxes=selected[[arms.index(arm) for arm in scored]].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6)
   vv=core.candidate_ious(boxes,gt)[0]
   assert len(vv)==len(scored) and np.isfinite(vv).all() and ((0<=vv)&(vv<=1)).all()
   values.update(dict(zip(scored,map(float,vv))))
   part='DEV336' if identity(r) in devkeys else 'NONDEV5576'
   tasks.append({'row':f'R{i+1:05d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':part,'split':splits[oldid_by_key[identity(r)]] if part=='DEV336' else 'NONDEV5576','queries':qmap,'IoU':{arm:values[arm] for arm in arms},'statuses':status[r['id']]})
  assert len(tasks)==n;summary,scenes=summarize(tasks,arms,refs,n);neg=[]
  for r in tasks:
   for arm,ref in itertools.product(arms,repeat=2):
    d=r['IoU'][arm]-r['IoU'][ref]
    if arm!=ref and d< -1e-10:neg.append({'row':r['row'],'part':r['part'],'split':r['split'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
  selection=None
  if calibration is not None:
   m=summary['ALL336']['balanced_mIoU_percent'];targets=calibration['target_arm_by_factor'];key={.5:'0.5',1.:'1',2.:'2'}
   best=max(m[targets[key[k]]] for k in calibration['factor_values'])
   chosen=next(k for k in calibration['tie_priority'] if best-m[targets[key[k]]]<=1e-10)
   selection={'selected_shared_factor':chosen,'selected_arm':targets[key[chosen]],'metric':'ALL336_two_platform_equal_weight_mIoU_percent','numeric_tie_tolerance_pp':1e-10,'tie_priority':calibration['tie_priority'],'platform_specific_calibration_forbidden':True,'all_trials_retained':True,'development_labels_used_for_calibration':True}
  dump(out/'ALL_TASK_ROWS_PRIVATE.json',tasks);dump(out/'TASK_SUMMARY.json',summary);dump(out/'ALL_SCENE_SUMMARIES_PRIVATE.json',scenes);dump(out/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg)
  dump(out/'SHARED_CALIBRATION_SELECTION.json',selection)
  receipt={'state':'PASS_R53_SINGLE_POSTSEAL_CONFIGURED_ORIGINAL_BOX_TASK_SCORE','n':n,'arm_names':arms,'task_values':n*len(arms),'new_GT_scored_task_values':n*len(scored),'reused_saved_task_values':n*len(reused),'ordered_negative_comparisons':len(neg),'single_GT_scoring_pass':True,'unknown_fallback_and_no_update_rows_remain_full_denominator':True,'source_256_boxes_not_changed':True,'balanced_platforms_not_micro_selection':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'independent_prediction_audit_sha256':sha(local(plan['independent_prediction_audit'])),'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'seconds':time.perf_counter()-start,'no_protected_FINAL_specific_artifacts':True}
  dump(out/'SCORING_RECEIPT.json',receipt);dump(out/'SEALED.json',{'state':receipt['state'],'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
  print(json.dumps(receipt),flush=True)
 except BaseException as exc:
  dump(out/'PARTIAL_ROWS_PRIVATE.json',tasks);dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':len(tasks),'no_automatic_GT_rerun_or_method_change':True});raise

if __name__=='__main__':main()
