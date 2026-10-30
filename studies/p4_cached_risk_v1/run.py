"""P4 only: descriptive risk statistics from previously audited scalar metrics."""
from pathlib import Path
import os,sys,json,hashlib,time,csv,argparse,collections,math
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
D=R/'outputs/p4_cached_risk_v1';P=R/'results/p4_cached_risk_v1'
COUNTS={'hc2':1902,'vidstg':10303};EPS=1e-10
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def identity():return dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
  x=os.fsdecode(args[0]);suffix=Path(x).suffix.lower()
  if x.startswith('/dev/nvidia')or suffix in ['.mp4','.avi','.mov','.pt','.pth','.npz']or x.endswith('_labels.jsonl'):raise PermissionError('Scalar-only P4: no GPU/media/model/raw prediction/GT-label reads')
def validate():
 p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_sha256'].items():assert sha(R/f)==h
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 return p
def prepare():
 assert not(D/'plan_v1.json').exists();inputs=[]
 for ds in COUNTS:
  root=R/'outputs/temporal_feedback_risk_v1'/ds;quality=R/'outputs/review_strengthening_v1/offline_current'/ds
  inputs += [root/x for x in ['metrics.json','reporting_receipt.json','score_receipt.json','sealed.json']]
  inputs += [quality/x for x in ['derived.json','audit_receipt.json','sealed.json']]
  inputs += [R/'results/temporal_feedback_risk_v1'/ds/'report.json',R/'data/evaluation'/f'{ds}.jsonl']
 inputs += [R/'data/annotations/vidstg_test.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 reuse={}
 for ds in COUNTS:
  reuse[ds]={}
  for label,rel in [('P1',f'results/review_strengthening_v1/matched_cpu/{ds}'),('P2_retained192',f'results/review_response_fast_v1/roi/{ds}')]:
   base=R/rel;paths=[base/'report.json',base/('ANALYSIS.md'if label=='P1'else'ROOT_FINDINGS_v1.md'),base/('root_export_receipt.json'if label=='P1'else'reporting_receipt.json')]
   assert all(f.is_file()for f in paths);inputs+=paths;reuse[ds][label]={str(f.relative_to(R)):sha(f)for f in paths}
 p=dict(schema='P4_predeclared_cached_scalar_risk_v1',authorization='2026-10-30 user: reuse existing results; CPU P4 only, P2 full deferred, no GPU, no manuscript edits.',counts=COUNTS,comparison='Current selected FullMAP minus untouched native Source, vIoU difference in percentage points.',equal_tolerance_pp=EPS,negative='d < -1e-10 pp; abs(d)<=1e-10 equal; d>1e-10 positive',severe='d <= -10 pp inclusive, exact descriptive threshold; no threshold search',q05='numpy.quantile(d,.05,method=linear), descriptive query-level 5th percentile',bootstrap=dict(replicates=10000,seed=20261022,unit='parent-cluster',aggregation='Each replicate resamples parent sources; query-summed deltas divided by query count of resampled clusters, never simple mean of parent means',interval='pointwise percentile95%, not familywise',statistics=['mean delta','negative proportion','severe proportion']),groups=['each complete domain','Vid official declarative/interrogative metadata','existing expert IoU quarter bins plus unscorable; missing anchor IoU never zero'],no_new_GT_scoring=True,no_new_model=True,no_new_predictions=True,no_GPU=True,resources='one sequential CPU process14-15 nice19 idleIO BLAS1, no service/process signals',P1_P2_reused=reuse,P2_full_status='deferred_by_user',P3_P5_writing_status='excluded_by_user',code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'audit_v1.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan_v1.json',p);atomic(P/'protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'})
 atomic(P/'reuse_index_v1.json',dict(P1_full_reused=True,P2_retained192_reused=True,P2_full_deferred=True,bindings=reuse));print(p['fingerprint'])
def load_inputs(ds):
 root=R/'outputs/temporal_feedback_risk_v1'/ds;aud=read(root/'reporting_receipt.json');assert aud['all_arrays_cold_rederived_exact']and aud['all_means_recomputed'];assert sha(root/'metrics.json')==aud['metrics_sha256']and sha(root/'score_receipt.json')==aud['score_sha256']and sha(root/'sealed.json')==aud['seal_sha256']
 rr=read(root/'metrics.json')['private_rows'];qroot=R/'outputs/review_strengthening_v1/offline_current'/ds;qa=read(qroot/'audit_receipt.json');assert sha(qroot/'derived.json')==qa['derived_sha256']and sha(qroot/'sealed.json')==qa['seal_sha256']and qa['all_GT_supported_anchors_distances_IoUs_exact']
 quality=read(qroot/'derived.json')['rows'];byq={r['id']:r for r in quality};assert len(byq)==len(quality)==len(rr)==COUNTS[ds]
 manifest={r['id']:r for r in map(json.loads,(R/'data/evaluation'/f'{ds}.jsonl').read_text().splitlines())};assert set(manifest)=={r['id']for r in rr}==set(byq)
 forms={}
 if ds=='vidstg':
  catalog=read(R/'data/annotations/vidstg_test.json');lookup=collections.defaultdict(set)
  for video in catalog:
   for field,form in [('captions','declarative'),('questions','interrogative')]:
    for z in video[field]:lookup[(str(video['vid']),z['description'])].add(form)
  for key,row in manifest.items():
   m=row['metadata'];x=lookup[(str(m['original_video_id']),m['caption'])];assert len(x)==1;forms[key]=next(iter(x))
  assert collections.Counter(forms.values())=={'declarative':4610,'interrogative':5693}
 out=[]
 for r in rr:
  q=byq[r['id']];assert q['parent']==r['parent']
  for a,b in [('Ours','FullMAP'),('Source','Source')]:
   assert q['main'][a]['v']==r['values'][b]['v']and q['main'][a]['t']==r['values'][b]['t']
  x,y=r['values']['FullMAP']['v'],r['values']['Source']['v'];assert all(math.isfinite(z)and 0<=z<=1 for z in [x,y]);assert(q['quality']is None)==(q['quality_group']=='unscorable_no_GT_anchor')
  out.append(dict(id=r['id'],parent=r['parent'],source_v=y,current_v=x,delta_pp=100*(x-y),quality_group=q['quality_group'],quality=q['quality'],query_form=forms.get(r['id'],'not_applicable')))
 assert len({r['id']for r in out})==COUNTS[ds];return out
def groups(rows,ds):
 z={'overall':rows}
 if ds=='vidstg':
  for name in ['declarative','interrogative']:z['query_form/'+name]=[r for r in rows if r['query_form']==name]
 for name in ['IoU_0_25','IoU_25_50','IoU_50_75','IoU_75_100','unscorable_no_GT_anchor']:z['expert_quality/'+name]=[r for r in rows if r['quality_group']==name]
 return z
def summary(rows):
 import numpy as np
 d=np.array([r['delta_pp']for r in rows]);parents=sorted({r['parent']for r in rows});ix={v:i for i,v in enumerate(parents)};counts=np.zeros(len(parents));sums=np.zeros((len(parents),3))
 for r in rows:
  i=ix[r['parent']];counts[i]+=1;sums[i]+=[r['delta_pp'],r['delta_pp']< -EPS,r['delta_pp']<=-10]
 w=np.random.default_rng(20261022).multinomial(len(parents),[1/len(parents)]*len(parents),size=10000);boot=(w@sums)/(w@counts)[:,None];ci=np.quantile(boot,[.025,.975],axis=0)
 return dict(queries=len(rows),parents=len(parents),source_vIoU_percent=float(100*np.mean([r['source_v']for r in rows])),current_vIoU_percent=float(100*np.mean([r['current_v']for r in rows])),mean_delta_pp=float(d.mean()),mean_delta_CI95_pp=ci[:,0].tolist(),negative_count=int((d< -EPS).sum()),negative_percent=float(100*(d< -EPS).mean()),negative_percent_CI95=(100*ci[:,1]).tolist(),equal_count=int((np.abs(d)<=EPS).sum()),positive_count=int((d>EPS).sum()),severe_le_minus10_count=int((d<=-10).sum()),severe_le_minus10_percent=float(100*(d<=-10).mean()),severe_percent_CI95=(100*ci[:,2]).tolist(),delta_q05_pp=float(np.quantile(d,.05,method='linear')),minimum_delta_pp=float(d.min()),maximum_delta_pp=float(d.max()),negative_not_severe_count=int(((d< -EPS)&(d> -10)).sum()))
def derive(ds):
 import numpy as np
 plan=validate();assert not(D/ds/'derived_receipt.json').exists();start=time.perf_counter();rows=load_inputs(ds);report={g:summary(rr)for g,rr in groups(rows,ds).items()}
 old=read(R/'results/temporal_feedback_risk_v1'/ds/'report.json')['paired']['FullMAP-Source']['v'];assert abs(report['overall']['mean_delta_pp']-old['mean'])<1e-12 and np.allclose(report['overall']['mean_delta_CI95_pp'],old['CI95'],rtol=0,atol=1e-12)
 atomic(D/ds/'private_scalar_rows.json',dict(rows=rows));atomic(P/ds/'report.json',dict(dataset=ds,groups=report,plan_fingerprint=plan['fingerprint']))
 with(P/ds/'all_signed_rows.tsv').open('w')as f:
  fields=['query','parent','query_form','expert_quality_group','mean_annotated_anchor_IoU','Source_vIoU_percent','Current_vIoU_percent','delta_pp','negative','equal','positive','severe_le_minus10']
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(fields);pm={k:f'P{i:04d}'for i,k in enumerate(sorted({r['parent']for r in rows}),1)}
  for i,r in enumerate(rows,1):
   d=r['delta_pp'];w.writerow([f'Q{i:05d}',pm[r['parent']],r['query_form'],r['quality_group'],r['quality'],100*r['source_v'],100*r['current_v'],d,d< -EPS,abs(d)<=EPS,d>EPS,d<=-10])
 with(P/ds/'risk_table.tsv').open('w')as f:
  fields=['queries','parents','mean_delta_pp','mean_delta_CI95_pp','negative_count','negative_percent','severe_le_minus10_count','severe_le_minus10_percent','delta_q05_pp','equal_count','positive_count'];w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['group',*fields]);w.writerows([[g,*[v[k]for k in fields]]for g,v in report.items()])
 rec=dict(status='derived_whole_domain_P4_from_audited_scalars_pending_independent_audit',dataset=ds,queries=len(rows),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,private_scalar_sha256=sha(D/ds/'private_scalar_rows.json'),report_sha256=sha(P/ds/'report.json'),signed_sha256=sha(P/ds/'all_signed_rows.tsv'),table_sha256=sha(P/ds/'risk_table.tsv'),plan_fingerprint=plan['fingerprint'],old_full_source_mean_and_CI_reproduced=True,new_model=0,new_Source=0,new_expert=0,new_predictions=0,new_GT_scoring=0,GPU=False)
 atomic(D/ds/'derived_receipt.json',rec);atomic(P/ds/'derived_receipt.json',rec);print(json.dumps(dict(dataset=ds,overall=report['overall'])))
if __name__=='__main__':
 sys.addaudithook(guard);p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','derive']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args();prepare()if a.command=='prepare'else derive(a.dataset)
