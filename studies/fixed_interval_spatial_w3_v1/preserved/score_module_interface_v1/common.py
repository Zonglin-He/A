"""Fixed-Full-interval spatial controls; no acquisition or temporal update."""
from pathlib import Path
import os,sys,time,json,functools,shutil,csv
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
D=R/'outputs/fixed_interval_spatial_w3_v1';P=R/'results/fixed_interval_spatial_w3_v1'
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as old
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
read=old.read;sha=old.sha;atomic=old.atomic;digest=old.digest;identity=old.identity
fixed=functools.lru_cache(None)(read)
RECHECK=R/'outputs/spatial_table_recheck_v1';TEMP=R/'outputs/reviewer_key_controls_v1';COMP=old.selected.D
COUNTS={'hc2':1902,'vidstg':10303}
NEW=['PastOnlyFixed','OneWayMatchFixed']
ARMS=['PureLog','PureLinear','SharedCorrection','Independent',*NEW,'Full']
PAIRS=[('Full',x)for x in ARMS if x!='Full']
@functools.lru_cache(None)
def rows(ds):return old.selected.records(ds)
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def refnpz(root,e):
 assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256']
 with np.load(root/e['path'])as z:return {k:z[k].copy()for k in z.files}
def references(ds,row):
 key=row['id'];a=fixed(RECHECK/ds/'sealed.json')['files'][key];b=fixed(TEMP/'temporal'/ds/'sealed.json')['files'][key];e=fixed(COMP/f'components_{ds}_sealed.json')['files'][key]
 x=refnpz(RECHECK,a);y=refnpz(TEMP,b);z=refnpz(COMP,e)
 ids=x['frame_ids'];pos=x['positions8'];iv=x['Full_interval']
 for n in [y,z]:assert np.array_equal(ids,n['frame_ids'])and np.array_equal(pos,n['positions8'])
 for v in [y['Ours_interval'],y['ExpertInitTR_interval'],z['FullMAP_interval'],z['FinalIndependent_interval']]:assert np.array_equal(iv,v)
 assert np.array_equal(x['Full'],y['Ours'])and np.array_equal(x['Full'],z['FullMAP'])
 a_=dict(frame_ids=ids,positions8=pos,interval=iv,PureLog=x['PureLogFixed'],PureLinear=x['PureLinearFixed'],SharedCorrection=y['ExpertInitTR'],Independent=z['FinalIndependent'],Full=x['Full'])
 proof=dict(recheck_prediction_sha256=a['sha256'],temporal_prediction_sha256=b['sha256'],components_prediction_sha256=e['sha256'],same_Full_interval_exact=True,same_physical_grid_positions_exact=True,unchanged_Full_exact_three_references=True)
 return a_,proof
def numpy_dp(bank):
 """Independent NumPy recurrence, same explicit first-index tie rule."""
 u=bank.score_unary.numpy();p=bank.score_pair.numpy();v=bank.valid.numpy().copy();v[~bank.active.numpy(),1:]=False
 value=np.where(v[0],u[0],-np.inf);back=[]
 for t in range(1,len(u)):
  scores=(value[:,None]+p[t-1])+u[t,None,:];previous=np.argmax(scores,axis=0);value=np.where(v[t],scores[previous,np.arange(3)],-np.inf);back.append(previous)
 act=int(np.argmax(value));acts=[act]
 for previous in reversed(back):act=int(previous[act]);acts.append(act)
 acts=np.asarray(acts[::-1]);return bank.boxes.numpy()[np.arange(len(acts)),acts],acts
def derive(ds,row,independent=False,preflight=False):
 ref,pr=references(ds,row);c,proof=old.source(ds,row);assert np.array_equal(c['ids'],ref['frame_ids'])and np.array_equal(c['pos'],ref['positions8'])and np.array_equal(c['E'],ref['SharedCorrection'])
 iv=ref['interval'];main=SpatialGeometry.from_cached_geometry(c['geometries']['main'])
 if preflight:
  got=evidence_dp(main.for_interval(iv));assert np.array_equal(got['final'].numpy(),ref['Full'])
  bank=main.for_interval(iv);act=bank.score_unary.masked_fill(~bank.valid,-torch.inf).argmax(1);assert np.array_equal(bank.boxes[torch.arange(bank.frames),act].numpy(),ref['Independent'])
 arrays={k:ref[k]for k in ['frame_ids','positions8','interval']}
 for arm,gkey in [('PastOnlyFixed','past'),('OneWayMatchFixed','oneway')]:
  g=SpatialGeometry.from_cached_geometry(c['geometries'][gkey])
  assert list(g.positions)==list(main.positions)
  for field in ['E','native_boxes','frame_ids']:
   assert torch.equal(getattr(g,field),getattr(main,field)),(arm,field)
  if arm=='PastOnlyFixed':
   assert not g.side_available[:,1].any()and(g.source_anchors[:,1]==-1).all()
   assert torch.equal(g.boxes[:,0:2],main.boxes[:,0:2])and torch.equal(g.side_available[:,0],main.side_available[:,0])
  else:assert torch.equal(g.source_anchors,main.source_anchors)
  bank=g.for_interval(iv);arrays[arm]=(numpy_dp(bank)[0]if independent else evidence_dp(bank)['final'].numpy())
  assert np.isfinite(arrays[arm]).all()
 return arrays,dict(**pr,parent_payload=proof,GT_read=False,temporal_update_executed=False,Source_expert_GPU_model_optimizer_calls=0,spatial_geometry_fitted_again=False,Full_method_unchanged=True)
def reused_metrics(ds):
 aa=fixed(RECHECK/ds/'metrics.json')['private_rows'];bb=fixed(TEMP/'temporal'/ds/'metrics.json')['private_rows'];cc=fixed(COMP/'components'/ds/'metrics.json')['private_rows']
 a={x['id']:x for x in aa};b={x['id']:x for x in bb};c={x['id']:x for x in cc};keys={r['id']for r in rows(ds)};assert set(a)==set(b)==set(c)==keys
 return {k:dict(parent=str(a[k]['parent']),duration_group=a[k]['duration_group'],values=dict(PureLog=a[k]['values']['PureLogFixed'],PureLinear=a[k]['values']['PureLinearFixed'],Full=a[k]['values']['Full'],SharedCorrection=b[k]['values']['ExpertInitTR'],Independent=c[k]['values']['FinalIndependent']))for k in a}
def report(rr,folder):
 from full_score import cluster
 folder.mkdir(parents=True,exist_ok=True)
 def summary(items):
  means={a:{f:float(np.mean(v))if(v:=[r['values'][a][f]for r in items if r['values'][a][f]is not None])else None for f in ['t','v','s','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds']}for a in ARMS}
  paired={a+'-'+b:cluster(items,a,b)for a,b in PAIRS}
  return dict(queries=len(items),parents=len({r['parent']for r in items}),means=means,paired=paired)
 z=summary(rr);z['time_groups']={k:summary([r for r in rr if r['duration_group']==k])for k in ['short_le5s','long_gt5s']};z['scope']='same original Full interval/8expert inputs/full query roster; queryequal; 10000pairedparentCI seed20261022; pointwise exposed cohorts; unchanged soft-anchor implementation'
 atomic(folder/'report.json',z);old.selected.suite().signed_tails(rr,PAIRS,folder)
 with(folder/'all_query_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['query','timegroup','arm','tIoU','vIoU','R03','R05','observedIoU','unobservedIoU'])
  for i,r in enumerate(rr,1):
   for a in ARMS:m=r['values'][a];w.writerow([f'Q{i:05}',r['duration_group'],a,*[m.get(k)for k in ['t','v','vIoU_ge_03','vIoU_ge_05','observed_actual_sIoU','unobserved_actual_sIoU']]])
 return z
