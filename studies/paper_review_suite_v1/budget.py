"""Isolated actual joint-ReferDINO observation-budget study; no production edits."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,gc,subprocess,types,inspect,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import RUN,PROVIDER,read,sha,atomic,check,identity
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
CONFIGS=[(f'{schedule}{k}',schedule,k)for k in [2,4,8,16,32]for schedule in ['Hybrid','Uniform']]+[('Random8','Random',8),('SourceOnly8','SourceOnly',8),('DenseNative','Dense',None)]

def variable_geometry():
 import stvg.spatial as m
 source=inspect.getsource(m.SpatialGeometry);changed=source.replace('native.shape == (8,4)','native.shape == (len(self.positions),4)').replace('sorted(range(8),','sorted(range(len(self.positions)),')
 assert source!=changed and changed.replace('native.shape == (len(self.positions),4)','native.shape == (8,4)').replace('sorted(range(len(self.positions)),','sorted(range(8),')==source
 scope=dict(vars(m));exec(compile(changed,__file__,'exec'),scope);return scope['SpatialGeometry']

def positions(ids,extent,k,schedule,seed):
 import numpy as np
 from stvg.positions import observation_positions
 n=len(ids);k=min(n,k);a,b=extent;ids=np.asarray(ids,float)
 if schedule=='Hybrid'and k==8:return observation_positions(ids,extent)
 if schedule=='Random':return sorted(np.random.default_rng(seed).choice(n,k,replace=False).tolist())
 if schedule=='Uniform':pool=list(range(n));m=k
 else:pool=list(range(a,b+1));m=min(k if schedule=='SourceOnly'else max(1,k//2),len(pool))
 edges=np.linspace(ids[pool[0]],ids[pool[-1]]+1,m+1);out=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  eligible=[i for i in pool if lo<=ids[i]<hi and i not in out]
  if eligible:out.append(min(eligible,key=lambda i:(abs(ids[i]-(lo+hi)/2),i)))
 while len(out)<m:out.append(max((i for i in pool if i not in out),key=lambda i:(min(abs(ids[i]-ids[j])for j in out),-i)))
 while len(out)<k:
  remaining=[i for i in range(n)if i not in out]
  if schedule=='SourceOnly':out.append(min(remaining,key=lambda i:(min(abs(ids[i]-ids[a]),abs(ids[i]-ids[b])),i)))
  else:out.append(max(remaining,key=lambda i:(min(abs(ids[i]-ids[j])for j in out),-ids[i])))
 return sorted(out)

def evidence(features,valid,ids,pos,mask,shape,image_shape):
 import torch
 import torch.nn.functional as F
 mask=torch.as_tensor(mask,dtype=torch.float32).cpu();full=F.interpolate(mask[:,None],size=tuple(image_shape),mode='bilinear',align_corners=False).sigmoid();seeds=F.interpolate(full,size=tuple(shape),mode='area').squeeze(1).double();seeds*=valid[pos].reshape_as(seeds);obs=torch.zeros(len(ids),dtype=torch.bool);obs[pos]=True
 return types.SimpleNamespace(features=features,valid=valid,frame_ids=torch.tensor(ids,dtype=torch.long),positions=list(pos),height=shape[0],width=shape[1],seeds=seeds,observed=obs)

def frozen():
 p=read(D/'budget_plan.json');assert sha(Path(__file__))==p['code_sha256']and sha(Path(__file__).with_name('consensus_exact.py'))==p['consensus_sha256'];assert sha(R/'stvg/spatial.py')==p['geometry_sha256'];check();return p

def predict(stage):
 import numpy as np,torch
 from stvg.runtime import setup,lease
 from stvg.teacher_bridge import ResidentTeacher
 from stvg.decode import decode
 from stvg.spatial import evidence_dp
 from stvg.consensus import encode,apply
 from stvg.predictor import output_writeback
 from stvg.representation import extract
 from stvg.prototype import one_step
 from stvg.temporal import final_mbr
 from stvg.readout import native_view_indices
 from consensus_exact import estimate
 p=frozen();setup();gpu=lease();atomic(D/('budget_'+stage+'_worker.json'),identity());klass=variable_geometry();selection=read(D/'selection.json');supports=read(D/'supports_sealed.json');cfg=read(R/'configs/default.json');teacher=ResidentTeacher(cache_dir=D/'budget_teacher_cache',fingerprint=p['teacher_runtime_fingerprint']);files={};failed=[];started=time.perf_counter();seen=0
 try:
  for ds,groups in selection.items():
   files[ds]={};oldseal=read(RUN/ds/'sealed.json')
   for row in groups[stage]:
    key=row['id'];entry=supports['files'][ds][stage][key];assert sha(D/entry['shared_path'])==entry['shared_sha256'];shared=torch.load(D/entry['shared_path'],map_location='cpu',weights_only=False);cap=torch.load(shared['parent_capture'],map_location='cpu',weights_only=False);assert sha(shared['parent_capture'])==shared['parent_capture_sha256'];frames,ids=decode(row['metadata'],ds);raw=[]
    for rec,z in zip(cap['records'],cap['native_logits']):
     a,b=native_view_indices(z);raw.append([rec['frame_ids'][a],rec['frame_ids'][b]])
    extent=(ids.index(min(a for a,b in raw)),ids.index(max(b for a,b in raw)));files[ds][key]={}
    for name,schedule,k in CONFIGS:
     if stage=='confirm'and name not in read(D/'budget_qualification.json')['eligible']:continue
     dest=D/'budget'/stage/ds;dest.mkdir(parents=True,exist_ok=True);out=dest/(key+'.'+name+'.npz');rp=dest/(key+'.'+name+'.json')
     if rp.exists():
      rr=read(rp);assert rr['fingerprint']==p['fingerprint']and sha(out)==rr['sha256'];files[ds][key][name]=rr;continue
     assert not out.exists();assert time.perf_counter()-started<43200;seed=int(hashlib.sha256(f"20261024:{ds}:{key}:{name}".encode()).hexdigest()[:16],16)%2**32;pos=list(range(len(ids)))if schedule=='Dense'else positions(ids,extent,k,schedule,seed);tick=time.perf_counter()
     try:
      # Every schedule/K has its own complete RGB context; NEVER subset the old8 outputs.
      native=teacher(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,dataset=ds)
     except Exception as exc:
      fail=dict(dataset=ds,stage=stage,id=key,config=name,actual_K=len(pos),T=len(ids),reason=str(exc),seconds=time.perf_counter()-tick,no_prediction=True,GT_read=False);failed.append(fail);atomic(dest/(key+'.'+name+'.failed.json'),fail)
      if teacher.log:
       teacher.log.flush();lp=Path(teacher.log.name)
       if lp.exists():(dest/(key+'.'+name+'.teacher.log')).write_bytes(lp.read_bytes())
      teacher.close();teacher=ResidentTeacher(cache_dir=D/'budget_teacher_cache',fingerprint=p['teacher_runtime_fingerprint']);continue
     ref=np.asarray(native['boxes'],np.float64);source=np.asarray(cap['Source'],np.float64);cost=native.get('evidence_original_cost',native['cost']);diagnostics={}
     if schedule=='Dense':boxes=ref;iv=cap['native_interval'];diagnostics['definition']='Native joint all-sampled-frame ReferDINO raw boxes, fixed native Source MBR time. No correction/propagation/prototype.'
     else:
      calibration=cfg['datasets'][ds]['E_calibration'];points=encode(ref)-encode(source)[pos];est=estimate(points,calibration['scale'],calibration['truncation']);E=output_writeback(source,apply(source,est['delta']),cfg['datasets'][ds]['initialization_gamma']);ev=evidence(shared['spatial_features'],shared['spatial_valid'],ids,pos,native['native_mask_logits'],shared['spatial_grid_shape'],native['input_shape'][-2:]);geometry=klass(ev,E,ref);aux=evidence_dp(geometry.for_interval([ids[0],ids[-1]+1]))['final'].numpy();f,_=extract(dict(frame_ids=ids,native_local_grid=shared['native_local_grid']),aux);q,_=one_step(f,cap['native_prior']);iv=final_mbr(ids,q)['interval'];boxes=evidence_dp(geometry.for_interval(iv))['final'].numpy();diagnostics=dict(consensus_nodes=est['nodes'],exact_consensus=True,source_geometry_recomputed=True,Native2D_repool=True)
      if name=='Hybrid8':
       assert pos==list(map(int,cap['positions8']))and np.array_equal(ref,np.asarray(cap['geometry']['native_boxes'])),'K8 native teacher bridge failed'
       assert np.array_equal(E,cap['E'])and np.array_equal(f,cap['content_features']),'K8 geometry/pool bridge failed'
       with np.load(RUN/ds/oldseal['files'][key]['path'])as old:assert np.array_equal(boxes,old['Full'])and np.array_equal(iv,old['Full_interval'])
       diagnostics['K8_full_bitwise']=True
     np.savez_compressed(out,boxes=boxes,interval=np.asarray(iv),positions=np.asarray(pos),frame_ids=np.asarray(ids));rr=dict(path=str(out.relative_to(D)),sha256=sha(out),fingerprint=p['fingerprint'],actual_K=len(pos),T=len(ids),K_over_T=len(pos)/len(ids),native_joint_context=True,cache_hit=native['cache_hit'],new_expert_forward=native['query_forwards'],expert_GPU_forward_seconds=cost['forward_seconds'],expert_prepost_seconds=cost['preprocess_seconds']+cost['postprocess_seconds'],model_load_seconds=native['model_load_seconds'],pipeline_seconds=time.perf_counter()-tick,GT_read=False,diagnostics=diagnostics);atomic(rp,rr);files[ds][key][name]=rr
    seen+=1;atomic(D/'budget_progress.json',dict(stage=stage,dataset=ds,queries=seen,configs=len(CONFIGS),failures=len(failed),seconds=time.perf_counter()-started,**identity(),GT_read=False));print(json.dumps(dict(stage=stage,dataset=ds,query=seen,failures=len(failed))),flush=True);del shared,cap,frames;gc.collect()
  atomic(D/('budget_'+stage+'_sealed.json'),dict(files=files,failures=failed,fingerprint=p['fingerprint'],GT_read=False,queries=seen,seconds=time.perf_counter()-started,scope='All requested cases retained, failed config has no fabricated prediction; partial coverage reported, never silently filtered.'))
 finally:teacher.close();gpu.close()

def score(stage):
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import numpy as np,torch
 from full_score import metric,cluster
 torch.set_num_threads(1);p=frozen();seal=read(D/('budget_'+stage+'_sealed.json'));assert seal['GT_read']is False;checks=0;maxerr=0.;selection=read(D/'selection.json')
 for ds,groups in selection.items():
  j=read(RUN/'plan.json')['jobs'][ds];f=PROVIDER/j['labels'];assert sha(f)==j['labels_sha256'];gt={x['id']:x for x in map(json.loads,f.read_text().splitlines())if x['dataset']==ds};old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']};reports={};signed=[]
  for name,_,_ in CONFIGS:
   rows=[];cost=[];missing=[]
   for i,r in enumerate(groups[stage],1):
    key=r['id'];e=seal['files'][ds][key].get(name)
    if e is None:missing.append(i);continue
    assert sha(D/e['path'])==e['sha256'];g=gt[key]
    with np.load(D/e['path'])as z:v,err=metric(z['boxes'],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z['interval'].tolist(),z['positions']);checks+=1;maxerr=max(maxerr,err)
    vals={a:old[key]['values'][a]for a in ['Source','Full','Interpolation']};vals[name]=v;rows.append(dict(parent=str(g['parent']),values=vals));cost.append(e);signed.append([f'Q{i:03d}',name,v['t'],v['v'],e['actual_K'],e['T'],e['expert_GPU_forward_seconds']])
   reports[name]=dict(status='complete'if not missing else'incomplete_no_full_cohort_claim',queries=len(rows),requested=len(groups[stage]),missing_queries=missing,means={a:{f:float(np.mean([x['values'][a][f]for x in rows]))for f in ['v','t']}for a in ['Source','Full',name]}if rows else None,paired=cluster(rows,name,'Source')if rows else None,actual_K=[min(x['actual_K']for x in cost),max(x['actual_K']for x in cost)]if cost else None,K_over_T_quantiles=np.quantile([x['K_over_T']for x in cost],[0,.25,.5,.75,1]).tolist()if cost else None,mean_actual_GPU_forward_seconds=float(np.mean([x['expert_GPU_forward_seconds']for x in cost]))if cost else None,new_expert_forwards=sum(x['new_expert_forward']for x in cost),cachehits=sum(x['cache_hit']for x in cost),all_requested_or_failed_preserved=True)
  pub=P/'budget'/ds/stage;atomic(pub/'report.json',dict(dataset=ds,stage=stage,configs=reports,fingerprint=p['fingerprint'],definition=p['description'],failure_count=sum(x['dataset']==ds for x in seal['failures'])));
  with(pub/'signed.tsv').open('w')as out:w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['query','configuration','t','v','actual_K','T','actual_GPU_forward_seconds']);w.writerows(signed)
 if stage=='screen':
  eligible=[];reasons={}
  reports_by_domain={ds:read(P/'budget'/ds/'screen/report.json')['configs']for ds in selection}
  for name,_,_ in CONFIGS:
   valid=all(reports_by_domain[ds][name]['status']=='complete'for ds in selection)
   affordable=valid and all(reports_by_domain[ds][name]['mean_actual_GPU_forward_seconds']<=10*reports_by_domain[ds]['Hybrid8']['mean_actual_GPU_forward_seconds']for ds in selection)
   if affordable:eligible.append(name)
   reasons[name]=dict(implementation_complete_both_domains=valid,within10xK8_GPUcost=affordable,accuracy_used_for_selection=False)
  atomic(D/'budget_qualification.json',dict(eligible=eligible,reasons=reasons,policy='Advance every implementation-complete candidate whose actual mean expert GPU forward <=10xK8 each domain; NO metric-based ranking or tuning. Failed candidates and partial coverage retained.'))
 atomic(D/('budget_'+stage+'_score_receipt.json'),dict(checks=checks,max_error=maxerr,seal_sha256=sha(D/('budget_'+stage+'_sealed.json')),all_stage_predictions_sealed_before_GT=True))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['predict','score']);a.add_argument('--stage',choices=['screen','confirm'],required=True);args=a.parse_args();predict(args.stage)if args.command=='predict'else score(args.stage)
