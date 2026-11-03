"""Full pure interpolation and two fixed offline diagnostics; no GPU/model calls."""
from pathlib import Path
import os,sys,json,time,hashlib,csv,argparse
from functools import lru_cache
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'studies/pure_expert_interpolation_v1'))
import run as pure
parent=pure.parent;read=pure.read;sha=pure.sha;atomic=pure.atomic;digest=pure.digest;identity=pure.identity
import numpy as np,torch
sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'))
from statistics_v1 import stats
D=R/'outputs/spatial_temporal_six_v1';P=R/'results/spatial_temporal_six_v1';OLD=pure.OLD;RUN=pure.RUN;SEL=parent.selected
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['Source','SourceMatched','PureNative','PureMatched','Current','HybridMatched']
PAIRS=[('Current','PureMatched'),('Current','PureNative'),('Current','Source'),('Current','SourceMatched'),('PureMatched','HybridMatched')];EPS=1e-10
fixed=lru_cache(None)(read)
def rows(ds):return SEL.records(ds)
def validate():
 p=read(D/'cpu_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for g in ['code_sha256','input_sha256']:
  for f,h in p[g].items():assert sha(R/f)==h,f
 return p
def prepare():
 assert not(D/'cpu_plan.json').exists();inputs=[R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',RUN/'plan.json',pure.D/'complete_published_v1.json',pure.D/'plan.json']
 for ds in COUNTS:
  inputs += [RUN/ds/'sealed.json',OLD/'matched_cpu'/ds/'sealed.json',OLD/'matched_cpu'/ds/'metrics.json',OLD/'matched_cpu'/ds/'reporting_receipt.json',SEL.D/f'components_{ds}_sealed.json',SEL.D/'components'/ds/'metrics.json',SEL.D/f'components_{ds}_reporting_receipt.json',parent.D/'temporal'/ds/'metrics.json',parent.D/'temporal'/ds/'reporting_receipt.json',OLD/'offline_current'/ds/'derived.json',OLD/'offline_current'/ds/'audit_receipt.json',pure.D/ds/'sealed.json',pure.D/ds/'metrics.json',pure.D/ds/'reporting_receipt.json']
 p=dict(schema='six_experiments_CPU_full_pure_temporal_quality_expert_quality_v1',authorization='2026-11-03 user: 六个实验都做; includes explicit full pure interpolation expansion, not old fullgamma/CG/corruption resumption',counts=COUNTS,
  pure='Unchanged proven pure_interpolation, raw eight boxes/time only, nearest endpoint holding; common Current interval for spatial comparison, native Source interval for own-interval pure pipeline. Reuse128 already audited matched-time pure predictions/scores exactly. No E-tail in pure; retain old hybrid only as separately named reference.',
  temporal_groups='Native Source tIoU [0,.3),[.3,.5),[.5,1]. Two t-only contrasts: Source→Current and SourceJointMAP q0→ExpertInitTR visual q1, no readout confound in latter. No vIoU attribution mixing spatial paths. Unchanged absolute tolerance1e-10 pp; decline magnitudes positive, median/p90 linear quantile among losses only.',
  quality='Existing independently audited mean expert IoU on GT-supported anchors; fixed quarters plus missing, never missing=0. Own-interval Source/PureNative/Current and common-Current-time SourceMatched/PureMatched/Current. Offline labels only; no target selection.',
  score='Whole domain prediction seal then separate official score only new PureNative and missing PureMatched; reuse audited Source/current/hybrid and128pure scalar by identity. Separate allarray/input/predreceiptSHA/reusedscore/means audit. No new Source/expert/model/optimizer/GPU/media. Parent10000pointwiseCI, allsigned/tails/shortlong.',
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'cpu_controller.py',R/'studies/pure_expert_interpolation_v1/run.py',R/'studies/reviewer_key_controls_v1/common_v1.py',R/'studies/reviewer_key_controls_v1/statistics_v1.py',R/'tools/full_score.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'cpu_plan.json',p);atomic(P/'cpu_protocol.json',p);print(p['fingerprint'],flush=True)
def derive(ds,row):
 key=row['id'];e=fixed(RUN/ds/'sealed.json')['files'][key];m=fixed(OLD/'matched_cpu'/ds/'sealed.json')['files'][key];ce=fixed(SEL.D/f'components_{ds}_sealed.json')['files'][key]
 for base,x in [(RUN/ds,e),(OLD,m),(SEL.D,ce)]:assert sha(base/x['path'])==x['sha256']
 with np.load(RUN/ds/e['path'])as z,np.load(OLD/m['path'])as old,np.load(SEL.D/ce['path'])as cur:
  ids=z['frame_ids'];pos=z['positions8'];assert np.array_equal(ids,old['frame_ids'])and np.array_equal(pos,old['positions8'])
  p=pure.pure_interpolation(ids,pos,z['native_boxes']);a=dict(frame_ids=ids.copy(),positions8=pos.copy(),Source=cur['Source'].copy(),SourceMatched=old['SourceMatchedTime'].copy(),PureNative=p.copy(),PureMatched=p.copy(),Current=old['Ours'].copy(),HybridMatched=old['ExpertInterpolation'].copy())
  for arm in ARMS:a[arm+'_interval']=(cur['Source_interval']if arm in ['Source','PureNative']else old['Ours_interval']).copy()
  ps=fixed(pure.D/ds/'sealed.json')['files'];reuse=key in ps
  if reuse:
   pe=ps[key];assert sha(pure.D/pe['path'])==pe['sha256']
   with np.load(pure.D/pe['path'])as q:assert np.array_equal(p,q['PureExpert'])and np.array_equal(a['PureMatched_interval'],q['PureExpert_interval'])
 return a,dict(parent_prediction_sha256=e['sha256'],matched_prediction_sha256=m['sha256'],current_prediction_sha256=ce['sha256'],prior128_pure_exact_reuse=reuse)
def preflight():
 validate();parent.infer_guard();torch.set_num_threads(1);cases=[]
 for ds in COUNTS:
  for row in rows(ds)[:2]:
   a,p=derive(ds,row);b,q=derive(ds,row);assert p==q and all(np.array_equal(v,b[k])for k,v in a.items());cases.append(dict(dataset=ds,all_arrays_repeat_exact=True,pure_no_Source_E_input=True))
 atomic(D/'cpu_preflight.json',dict(status='passed_four_real_cache_cases',cases=cases,new_model=0,new_GPU=0,new_Source=0,new_expert=0));print('CPU preflight passed',flush=True)
def predict(ds):
 plan=validate();parent.infer_guard();torch.set_num_threads(1);assert read(D/'cpu_preflight.json')['status']=='passed_four_real_cache_cases';dest=D/'pure_full'/ds;assert not(dest/'sealed.json').exists();start=time.perf_counter();files={}
 for i,row in enumerate(rows(ds),1):
  parent.resources(start);a,proof=derive(ds,row);key=row['id'];out=dest/'predictions'/(key+'.npz');rp=dest/'receipts'/(key+'.json');out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('wb')as f:np.savez_compressed(f,**a)
  with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),proof=proof,GT_read=False);atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  if i%25==0 or i==COUNTS[ds]:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(queries=len(files),files=files,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print(ds,'pure full sealed',flush=True)
def score(ds):
 validate();torch.set_num_threads(1);dest=D/'pure_full'/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=SEL.labels(ds)
 old={r['id']:r for r in read(OLD/'matched_cpu'/ds/'metrics.json')['private_rows']};main={r['id']:r for r in read(SEL.D/'components'/ds/'metrics.json')['private_rows']};prior={r['id']:r for r in read(pure.D/ds/'metrics.json')['private_rows']};rr=[];checks=0;mx=0
 from full_score import metric
 for row in rows(ds):
  key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];v=dict(Source=main[key]['values']['Source'],SourceMatched=old[key]['values']['SourceMatchedTime'],Current=old[key]['values']['Ours'],HybridMatched=old[key]['values']['ExpertInterpolation'])
  with np.load(D/e['path'])as z:
   inp={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ['PureNative','PureMatched']:
    if arm=='PureMatched'and key in prior:v[arm]=prior[key]['values']['PureExpert'];continue
    v[arm],err=metric(z[arm],inp,{int(k):b for k,b in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;mx=max(mx,err)
  assert v['PureNative']['t']==v['Source']['t']and v['PureMatched']['t']==v['Current']['t'];rr.append(dict(id=key,parent=old[key]['parent'],duration_group=old[key]['duration_group'],values=v))
 atomic(dest/'metrics.json',dict(private_rows=rr));SEL.suite().parent_report(rr,ARMS,PAIRS,P/'pure_full'/ds);SEL.suite().signed_tails(rr,PAIRS,P/'pure_full'/ds)
 atomic(dest/'score_receipt.json',dict(status='scored_full_pure_both_temporal_conventions',queries=len(rr),new_official_checks=checks,max_discrepancy=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),worker=identity()));print(ds,'pure full scored',flush=True)
def audit(ds):
 validate();parent.infer_guard();torch.set_num_threads(1);dest=D/'pure_full'/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['seal_sha256']==sha(dest/'sealed.json')and sc['metrics_sha256']==sha(dest/'metrics.json')
 for row in rows(ds):
  a,p=derive(ds,row);e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']and p==e['proof']
  with np.load(D/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
 rr=read(dest/'metrics.json')['private_rows'];rep=read(P/'pure_full'/ds/'report.json')
 for a,m in rep['means'].items():
  for k,v in m.items():
   vv=[r['values'][a][k]for r in rr if r['values'][a][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 ack=dict(status='verified_full_pure_all_arrays_SHA_means',queries=len(rr),all_arrays_cold_exact=True,report_sha256=sha(P/'pure_full'/ds/'report.json'),metrics_sha256=sha(dest/'metrics.json'),worker=identity(),new_GPU=0,new_Source=0,new_expert=0);atomic(dest/'reporting_receipt.json',ack);atomic(P/'pure_full'/ds/'reporting_receipt.json',ack);print(ds,'full audit passed',flush=True)
def temporal_rows(ds):
 ack=read(parent.D/'temporal'/ds/'reporting_receipt.json');assert ack['metrics_sha256']==sha(parent.D/'temporal'/ds/'metrics.json');out=[]
 for r in read(parent.D/'temporal'/ds/'metrics.json')['private_rows']:
  v=r['values'];src=v['Source']['t'];group='Source_t_0_03'if src<.3 else'Source_t_03_05'if src<.5 else'Source_t_05_1'
  for name,a,b in [('whole_temporal_branch','Ours','Source'),('visual_update_same_jointMAP','ExpertInitTR','SourceJointMAP')]:
   before=v[b]['t'];after=v[a]['t'];dt=100*(after-before);out.append(dict(id=r['id'],parent=r['parent'],group=group,comparison=name,before_tIoU_percent=100*before,after_tIoU_percent=100*after,delta_t_pp=dt,improved_percent=100.*(dt>EPS),unchanged_percent=100.*(abs(dt)<=EPS),harmed_percent=100.*(dt< -EPS)))
 return out
def temporal_report(rr):
 report={}
 for name in ['whole_temporal_branch','visual_update_same_jointMAP']:
  for group in ['all','Source_t_0_03','Source_t_03_05','Source_t_05_1']:
   qq=[r for r in rr if r['comparison']==name and(group=='all'or r['group']==group)];loss=[-r['delta_t_pp']for r in qq if r['delta_t_pp']< -EPS];rep=stats(qq,['before_tIoU_percent','after_tIoU_percent','delta_t_pp','improved_percent','unchanged_percent','harmed_percent']);rep['decline_magnitude_pp_among_harmed']=dict(queries=len(loss),median=float(np.quantile(loss,.5))if loss else None,p90=float(np.quantile(loss,.9))if loss else None);report[name+'/'+group]=rep
 return report
def quality_report(ds,rr):
 qrows=read(OLD/'offline_current'/ds/'derived.json')['rows'];quality={r['id']:r['quality']for r in qrows};out={}
 for support,arms in [('own_intervals',['Source','PureNative','Current']),('common_Current_interval',['SourceMatched','PureMatched','Current'])]:
  for group in ['all','q0_25','q25_50','q50_75','q75_100','missing']:
   def belongs(r):
    q=quality[r['id']];g='missing'if q is None else'q0_25'if q<.25 else'q25_50'if q<.5 else'q50_75'if q<.75 else'q75_100';return group=='all'or g==group
   qq=[r for r in rr if belongs(r)];z=[]
   for r in qq:
    vals=r['values'];a,b=arms[:2];dsource=100*(vals['Current']['v']-vals[a]['v']);dpure=100*(vals['Current']['v']-vals[b]['v']);z.append(dict(parent=r['parent'],Source_v_percent=100*vals[a]['v'],Pure_v_percent=100*vals[b]['v'],Current_v_percent=100*vals['Current']['v'],Current_minus_Source_v_pp=dsource,Current_minus_Pure_v_pp=dpure,improved_vs_Source_percent=100.*(dsource>EPS),harmed_vs_Source_percent=100.*(dsource< -EPS),improved_vs_Pure_percent=100.*(dpure>EPS),harmed_vs_Pure_percent=100.*(dpure< -EPS)))
   out[support+'/'+group]=stats(z,list(z[0])[1:]if z else[])
 return out
def diagnostics(ds):
 validate();torch.set_num_threads(1);ack=read(D/'pure_full'/ds/'reporting_receipt.json');assert ack['all_arrays_cold_exact'];tr=temporal_rows(ds);temporal=temporal_report(tr);rr=read(D/'pure_full'/ds/'metrics.json')['private_rows'];quality=quality_report(ds,rr)
 atomic(D/'diagnostics'/ds/'temporal_private_rows.json',tr);atomic(P/'diagnostics'/ds/'temporal_quality_groups.json',temporal);atomic(P/'diagnostics'/ds/'expert_quality_groups.json',quality)
 with(P/'diagnostics'/ds/'all_temporal_signed.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');keys=['group','comparison','before_tIoU_percent','after_tIoU_percent','delta_t_pp','improved_percent','unchanged_percent','harmed_percent'];w.writerow(['query',*keys]);ids={k:f'Q{i:05}'for i,k in enumerate(dict.fromkeys(r['id']for r in tr),1)};w.writerows([[ids[r['id']],*[r[k]for k in keys]]for r in tr])
 atomic(D/'diagnostics'/ds/'derive_receipt.json',dict(status='derived_three_temporal_bins_two_readouts_and_quality_quarters',queries=COUNTS[ds],new_prediction=0,new_model=0,new_GPU=0,worker=identity()));print(ds,'diagnostics derived',flush=True)
def diagnostic_audit(ds):
 validate();torch.set_num_threads(1);rr=temporal_rows(ds);want=temporal_report(rr);got=read(P/'diagnostics'/ds/'temporal_quality_groups.json');assert want==got
 scored=read(D/'pure_full'/ds/'metrics.json')['private_rows'];assert quality_report(ds,scored)==read(P/'diagnostics'/ds/'expert_quality_groups.json')
 assert sum(got['whole_temporal_branch/'+g]['queries']for g in ['Source_t_0_03','Source_t_03_05','Source_t_05_1'])==COUNTS[ds]
 ack=dict(status='independently_verified_all_temporal_groups_losses_quantiles_quality_quarters',queries=COUNTS[ds],tolerance_pp=EPS,temporal_report_sha256=sha(P/'diagnostics'/ds/'temporal_quality_groups.json'),quality_report_sha256=sha(P/'diagnostics'/ds/'expert_quality_groups.json'),all_missing_groups_retained=True,worker=identity(),new_model=0,new_GPU=0);atomic(D/'diagnostics'/ds/'audit_receipt.json',ack);atomic(P/'diagnostics'/ds/'audit_receipt.json',ack);print(ds,'diagnostics audit passed',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','predict','score','audit','diagnostics','diagnostic_audit']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([]if x.stage in ['prepare','preflight']else[x.dataset]))
