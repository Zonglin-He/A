"""Cache-only MAP paper components and real per-K budget reconstruction."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys,json,time,argparse,hashlib,subprocess,signal,gc,traceback,csv,shutil
from pathlib import Path
from functools import lru_cache
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/map_full_v1'),str(R/'studies/paper_review_suite_v1'),str(R/'studies/paper_followup_v1')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
import map_core
from common import records,labels,report
D=R/'outputs/map_components_v1';P=R/'results/map_components_v1';OLD=R/'outputs/paper_review_suite_v1';MAP=R/'outputs/map_full_now_v3';CG=R/'outputs/cgstvg_full_v1'
COUNTS=dict(hc2=1902,vidstg=10303)
COMPONENTS=['Source','EightReplacement','SharedCorrection','SpatialInference','FullMAP','SourceJointMAP','FinalIndependent']
CONFIGS=[f'{s}{k}'for k in [2,4,8,16]for s in ['Hybrid','Uniform']]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def exited(w):
 try:s=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return s[0]=='Z'or int(s[19])!=w['startticks']
 except FileNotFoundError:return True
def mem():return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
def group_signal(w,s):
 if not exited(w):assert os.getpgid(w['pid'])==w['pid'];os.killpg(w['pid'],s)
@lru_cache(maxsize=None)
def fixed(path):return read(path)
def teacher_signature(c):return digest({k:c[k]for k in ['dataset','caption','positions','frame_ids']})
def prepare():
 assert not(D/'plan.json').exists();check();index={}
 for path in sorted((OLD/'budget_teacher_cache').glob('*.json')):
  meta=read(path);assert path.stem==meta['key'];sig=teacher_signature(meta['context']);index.setdefault(sig,[]).append(dict(receipt=str(path),receipt_sha256=sha(path),payload=str(path.with_suffix('.pt')),payload_sha256=meta['sha256']))
 atomic(D/'teacher_index.json',index)
 code=[Path(__file__),R/'studies/map_full_v1/map_core.py',R/'studies/paper_review_suite_v1/budget_v2.py',R/'studies/paper_review_suite_v1/consensus_exact.py',R/'studies/paper_review_suite_v1/supports.py',R/'studies/paper_followup_v1/common.py',R/'tools/full_score.py',*sorted((R/'stvg').glob('*.py'))]
 inputs=[RUN/'plan.json',OLD/'selection.json',OLD/'supports_sealed.json',OLD/'budget_v2_plan.json',OLD/'budget_v2_confirm_sealed.json',D/'teacher_index.json',*[RUN/ds/'sealed.json'for ds in COUNTS],*[MAP/f'TA_{ds}_sealed.json'for ds in COUNTS]]
 p=dict(schema='MAP_paper_components_v1',authorization='Direct user requests existing local experiments recomputed from caches on CPU now, without interrupting CG-Vid. Table3 then fixed K2/4/8/16 budget and final-DP control; no unnecessary historical reruns.',
  components=COMPONENTS,budget_configs=CONFIGS,cohorts=dict(components=COUNTS,budget='Prior fixed confirm128parents/domain; original selection and all per-K teacher acquisitions, no resampling or accuracy selection.'),
  definitions=dict(Source='Original per-offset nativeMAP physical hull + Source boxes.',EightReplacement='Source boxes, replace only observed8 positions with exact same native teacher boxes; q0 jointMAP interval.',SharedCorrection='Existing globally corrected E; q0 jointMAP interval, no propagation or temporal revision.',SpatialInference='Existing geometry exactDP on q0 jointMAP interval, no temporal evidence revision.',FullMAP='Existing fixed auxiliary trajectory and pooled features -> same q1 -> jointMAP and actual final interval-conditioned DP. Must be bitwise current complete FullMAP.',SourceJointMAP='Unchanged Source boxes and q0 jointMAP; explicit readout control.',FinalIndependent='Same final q1 MAP interval/candidate bank/validity/unary as FullMAP; per-frame valid unary argmax, no link term or path optimization. Auxiliary trajectory and posterior unchanged.'),
  budget='Hybrid/Uniform2/4/8/16 use their OWN exact joint teacher observations/masks, cached original postfusion and pretext native grids, rebuild shared correction/geometry/auxDP, genuinely repool, same q0/k64/beta1/context1.5 -> q1 MAP -> finalDP. Every old MBR budget prediction must replay exactly before using the corresponding new MAP output. Never subset eight observations to impersonate another K.',
  science=dict(kappa=64,beta=1,context=1.5,observed=.5,unobserved=.5,link=.25,rounds=1,MAP='numpy.argmax joint physical interval, C-order earliest ties'),
  limits=dict(new_Source=0,new_prefix=0,new_teacher=0,new_GPU=0,optimizer=0,max_compute_processes=2,threads_each=1,CPU=[14,15],nice=19,io='idle class3',minimum_RAM_bytes=8*2**30,resume_RAM_bytes=10*2**30,max_seconds_each=259200,minimum_disk_free_bytes=60*2**30),
  not_available=dict(corruptions='Existing shift outputs contain boxes and intervals, not posterior/geometric candidate caches. Cannot derive complete new MAP path from those files alone; no new acquisition authorized in this CPU run.',TubeDETR='No compatible current-method captures/posteriors/geometry located in maintained checkout or baseline workspace. No fabricated MAP Ours.',deployment_cost='Cached CPU replay is conditional postprocessing/research cost, not complete source+expert deployment latency. Prior expert costs retained separately.'),
  scheduling='Original CG inference unchanged. Own CPU groups may pause on RAM<8GiB or original rolling pipeline>1.35x initial baseline with25new queries; resume after60s and RAM>10GiB. Own CG-MAP consumer guard taken over while obsolete automatic MBR strict dispatch is disabled with preserved state. No old source/hash edits. No automatic GPU reconstruction or old strict-deletion launch from this study.',
  evaluation='Whole task/domain seals before independent official labels; public v/t/s/recall.3/.5, all signed positive/negative, severe t/v union counted once/rescues, shortlong parent-cluster CIs and actual worker CPU segments. Full datasets historically exposed; no tuning or validation-set claim.',
  code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
 for f in code:
  target=D/'source_snapshot'/f.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
 atomic(D/'state.json',dict(status='prepared_CPU_cache_only',fingerprint=p['fingerprint']));print(p['fingerprint'])
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(f)==h,f
 check();return p
def rows(task,ds):return records(ds)if task=='components'else read(OLD/'selection.json')[ds]['confirm']
def load_parent(ds,row):
 e=fixed(RUN/ds/'sealed.json')['files'][row['id']];cp=RUN/ds/e['capture_path'];pp=RUN/ds/e['path'];rp=RUN/ds/e['receipt_path']
 assert sha(cp)==e['capture_sha256']and sha(pp)==e['sha256']and sha(rp)==e['receipt_sha256']
 import torch
 c=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read']
 return c,dict(capture=str(cp),capture_sha256=e['capture_sha256'],prediction=str(pp),prediction_sha256=e['sha256'],receipt=str(rp),receipt_sha256=e['receipt_sha256'])
def component_arrays(ds,row):
 import torch,numpy as np
 from stvg.spatial import SpatialGeometry,evidence_dp
 from supports import greedy
 c,parent=load_parent(ds,row);a=map_core.derive('TA',ds,c);g=SpatialGeometry.from_cached_geometry(c['geometry']);iv=a['SourceJointMAP_interval'].tolist();rep=np.asarray(c['Source']).copy();rep[c['positions8']]=g.native_boxes.numpy()
 arrays=dict(Source=a['Source'],Source_interval=a['Source_interval'],SourceJointMAP=a['SourceJointMAP'],SourceJointMAP_interval=a['SourceJointMAP_interval'],EightReplacement=rep,EightReplacement_interval=np.asarray(iv),SharedCorrection=np.asarray(c['E']),SharedCorrection_interval=np.asarray(iv),SpatialInference=evidence_dp(g.for_interval(iv))['final'].numpy(),SpatialInference_interval=np.asarray(iv),FullMAP=a['FullMAP'],FullMAP_interval=a['FullMAP_interval'],FinalIndependent=greedy(g.for_interval(a['FullMAP_interval'].tolist())),FinalIndependent_interval=a['FullMAP_interval'],frame_ids=a['frame_ids'],positions8=a['positions8'])
 me=fixed(MAP/f'TA_{ds}_sealed.json')['files'][row['id']];mp=MAP/me['path'];assert sha(mp)==me['sha256']
 with np.load(mp)as z:assert np.array_equal(arrays['FullMAP'],z['FullMAP'])and np.array_equal(arrays['FullMAP_interval'],z['FullMAP_interval'])
 assert np.array_equal(rep[np.setdiff1d(np.arange(len(rep)),c['positions8'])],np.asarray(c['Source'])[np.setdiff1d(np.arange(len(rep)),c['positions8'])])
 parent.update(current_MAP_path=str(mp),current_MAP_sha256=me['sha256']);return arrays,dict(parent=parent,FullMAP_bitwise=True,upstream_and_q1_fixed_for_final_control=True)
def budget_arrays(ds,row):
 import torch,numpy as np
 from stvg.spatial import evidence_dp
 from stvg.consensus import encode,apply
 from stvg.predictor import output_writeback
 from stvg.representation import extract
 from stvg.prototype import one_step
 from stvg.temporal import final_mbr
 from budget_v2 import variable_geometry,evidence
 from consensus_exact import estimate
 c,parent=load_parent(ds,row);e=fixed(OLD/'supports_sealed.json')['files'][ds]['confirm'][row['id']];sp=OLD/e['shared_path'];assert sha(sp)==e['shared_sha256'];s=torch.load(sp,map_location='cpu',weights_only=False,mmap=True)
 assert s['parent_capture_sha256']==parent['capture_sha256']and s['frame_ids']==c['frame_ids']and not s['GT_read']
 cfg=read(R/'configs/default.json')['datasets'][ds];klass=variable_geometry();index=fixed(D/'teacher_index.json');oldseal=fixed(OLD/'budget_v2_confirm_sealed.json')['files'][ds][row['id']]
 source=np.asarray(c['Source']);q0=c['native_prior'];ids=c['frame_ids'];a=dict(Source=source,Source_interval=np.asarray(map_core.derive('TA',ds,c)['Source_interval']),frame_ids=np.asarray(ids),positions8=np.asarray(c['positions8']));proof={}
 for name in CONFIGS:
  oe=oldseal[name];op=OLD/oe['path'];assert sha(op)==oe['sha256']
  with np.load(op)as z:pos=z['positions'].tolist();oldboxes=z['boxes'].copy();oldiv=z['interval'].tolist();assert np.array_equal(z['frame_ids'],ids)
  sig=teacher_signature(dict(dataset=ds,caption=row['metadata']['caption'],positions=pos,frame_ids=ids));candidates=index[sig]
  # Require identical full RGB/context identity if redundant cache receipts exist.
  contexts=[read(x['receipt'])['context']for x in candidates];assert all(v==contexts[0]for v in contexts)
  te=candidates[0];assert sha(te['receipt'])==te['receipt_sha256']and sha(te['payload'])==te['payload_sha256'];native=torch.load(te['payload'],map_location='cpu',weights_only=False,mmap=True)
  assert len(pos)==oe['actual_K']and contexts[0]['positions']==pos and contexts[0]['frame_ids']==ids
  ref=np.asarray(native['boxes'],np.float64);cal=cfg['E_calibration'];est=estimate(encode(ref)-encode(source)[pos],cal['scale'],cal['truncation']);E=output_writeback(source,apply(source,est['delta']),cfg['initialization_gamma'])
  ev=evidence(s['spatial_features'],s['spatial_valid'],ids,pos,native['native_mask_logits'],s['spatial_grid_shape'],native['input_shape'][-2:]);g=klass(ev,E,ref);aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();f,_=extract(dict(frame_ids=ids,native_local_grid=s['native_local_grid']),aux);q,_=one_step(f,q0)
  oldtime=final_mbr(ids,q)['interval'];oldout=evidence_dp(g.for_interval(oldtime))['final'].numpy();assert oldtime==oldiv and np.array_equal(oldout,oldboxes),(ds,row['id'],name,'Original per-K MBR replay mismatch')
  iv=map_core.map_interval(ids,q);a[name]=evidence_dp(g.for_interval(iv))['final'].numpy();a[name+'_interval']=np.asarray(iv)
  proof[name]=dict(original_prediction=str(op),original_prediction_sha256=oe['sha256'],teacher=te,original_cost=oe,old_MBR_all_arrays_bitwise=True,own_K_geometry_aux_repool_q1_recomputed=True)
  if name=='Hybrid8':
   me=fixed(MAP/f'TA_{ds}_sealed.json')['files'][row['id']];mp=MAP/me['path'];assert sha(mp)==me['sha256']
   with np.load(mp)as z:assert np.array_equal(a[name],z['FullMAP'])and np.array_equal(a[name+'_interval'],z['FullMAP_interval'])
 return a,dict(parent=parent,shared_path=str(sp),shared_sha256=e['shared_sha256'],configs=proof)
def derive(task,ds,row):return component_arrays(ds,row)if task=='components'else budget_arrays(ds,row)
def preflight():
 p=validate();import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);proof=[]
 for task in ['components','budget']:
  for ds in COUNTS:
   row=rows(task,ds)[0];a,meta=derive(task,ds,row);b,_=derive(task,ds,row);assert all(np.array_equal(a[k],b[k])for k in a);proof.append(dict(task=task,dataset=ds,arrays_exact=True,original_MAP_or_per_K_MBR_bitwise=True))
 assert not torch.cuda.is_initialized();atomic(D/'preflight_receipt.json',dict(status='passed',queries=4,repeated_exact=True,proof=proof,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint']));atomic(P/'preflight_receipt.json',read(D/'preflight_receipt.json'))
def predict(task,ds):
 p=validate();import torch,numpy as np
 from stvg.runtime import guard
 sys.addaudithook(guard);torch.set_num_threads(1);started=time.perf_counter();who=identity();dest=D/task/ds;files={};rr=rows(task,ds);atomic(D/f'{task}_{ds}_worker.json',who)
 for i,row in enumerate(rr,1):
  assert time.perf_counter()-started<p['limits']['max_seconds_each']and shutil.disk_usage(R).free>p['limits']['minimum_disk_free_bytes']
  key=row['id'];out=dest/'predictions'/(key+'.npz');rp=dest/'receipts'/(key+'.json');assert not out.exists()and not rp.exists(),'No blind overwrite/retry'
  t=time.perf_counter();a,meta=derive(task,ds,row);out.parent.mkdir(parents=True,exist_ok=True)
  with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
  os.replace(out.with_suffix('.tmp'),out)
  with np.load(out)as z:assert all(np.array_equal(z[k],a[k])for k in a)
  rec=dict(path=str(out.relative_to(D)),sha256=sha(out),provenance=meta,worker=who,conditional_CPU_wall_seconds=time.perf_counter()-t,GT_read=False,new_Source=0,new_expert=0,new_GPU=0,fingerprint=p['fingerprint']);atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
  atomic(D/f'{task}_{ds}_progress.json',dict(completed=i,total=len(rr),worker=who,seconds=time.perf_counter()-started,GT_read=False));print(task,ds,i,flush=True);del a,meta;gc.collect()
 assert not torch.cuda.is_initialized();atomic(D/f'{task}_{ds}_sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=who,seconds=time.perf_counter()-started,fingerprint=p['fingerprint']))
def score(task,ds):
 p=validate();import torch,numpy as np
 from full_score import metric
 torch.set_num_threads(1);sp=D/f'{task}_{ds}_sealed.json';seal=read(sp);rr=rows(task,ds);assert seal['queries']==len(rr)and not seal['GT_read'];truth=labels(ds);arms=COMPONENTS if task=='components'else['Source',*CONFIGS];outrows=[];maxerr=0.;checks=0
 for row in rr:
  e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(D/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for a in arms:
    pos=z['positions8']
    if task=='budget'and a!='Source':pos=read(e['provenance']['configs'][a]['teacher']['receipt'])['context']['positions']
    vals[a],err=metric(z[a],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),pos);maxerr=max(maxerr,err);checks+=1
  outrows.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
 pairs=([('EightReplacement','Source'),('SharedCorrection','EightReplacement'),('SpatialInference','SharedCorrection'),('FullMAP','SpatialInference'),('FullMAP','Source'),('FullMAP','SourceJointMAP'),('FullMAP','FinalIndependent')]if task=='components'else[(a,'Source')for a in CONFIGS]+[(f'Hybrid{k}','Hybrid8')for k in [2,4,16]])
 res=report(outrows,arms,pairs,P/task/ds);atomic(D/task/ds/'metrics.json',dict(private_rows=outrows,report=res));receipt=dict(status='scored',queries=len(rr),checks=checks,max_error=maxerr,seal_sha256=sha(sp),whole_domain_sealed_before_GT=True,fingerprint=p['fingerprint']);atomic(D/f'{task}_{ds}_score_receipt.json',receipt);atomic(P/task/ds/'score_receipt.json',receipt)
def reporting(task,ds):
 p=validate();import torch,numpy as np
 torch.set_num_threads(1);seal=read(D/f'{task}_{ds}_sealed.json');score=read(D/f'{task}_{ds}_score_receipt.json');assert score['seal_sha256']==sha(D/f'{task}_{ds}_sealed.json');rr=read(D/task/ds/'metrics.json')['private_rows'];res=read(P/task/ds/'report.json');segments={};signed=[];checks=0
 lookup={r['id']:r for r in rows(task,ds)}
 for i,r in enumerate(rr,1):
  e=seal['files'][r['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];checks+=2
  a,_=derive(task,ds,lookup[r['id']])
  with np.load(D/e['path'])as z:assert all(np.array_equal(z[k],a[k])for k in a)
  w=e['worker'];s=segments.setdefault(f"{w['pid']}:{w['startticks']}",dict(queries=0,conditional_CPU_wall_seconds=0.));s['queries']+=1;s['conditional_CPU_wall_seconds']+=e['conditional_CPU_wall_seconds']
  for a in r['values']:
   if a=='Source':continue
   t=100*(r['values'][a]['t']-r['values']['Source']['t']);v=100*(r['values'][a]['v']-r['values']['Source']['v']);signed.append([f'Q{i:05d}',r['duration_group'],a,t,v,t< -5,v< -5,t< -5 or v< -5,t>5 or v>5])
 for a,mm in res['means'].items():
  for f,v in mm.items():
   vv=[r['values'][a][f]for r in rr if r['values'][a][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 folder=P/task/ds
 with(folder/'all_signed_severe_timegroups.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','timegroup','arm','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union_once']);w.writerows(signed)
 if task=='budget':
  costs={a:dict(mean_original_expert_forward_seconds=float(np.mean([e['provenance']['configs'][a]['original_cost']['expert_GPU_forward_seconds']for e in seal['files'].values()])),actual_K=[e['provenance']['configs'][a]['original_cost']['actual_K']for e in seal['files'].values()],original_acquisition_not_repeated=True)for a in CONFIGS};atomic(folder/'original_expert_cost.json',costs)
  with(folder/'budget_curve.tsv').open('w')as f:
   w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['schedule','nominal_K','queries','tIoU_percent','vIoU_percent','R_03_percent','R_05_percent','mean_original_expert_forward_seconds']);w.writerows([[a.rstrip('0123456789'),int(a[len(a.rstrip('0123456789')):]),len(rr),*[100*res['means'][a][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']],costs[a]['mean_original_expert_forward_seconds']]for a in CONFIGS])
 receipt=dict(status='verified_complete_CPU_MAP_domain',queries=len(rr),prediction_receipt_SHA_checks=checks,all_arrays_cold_replayed_exact=True,all_means_recomputed=True,worker_segments=segments,new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,score_sha256=sha(D/f'{task}_{ds}_score_receipt.json'),fingerprint=p['fingerprint']);atomic(D/f'{task}_{ds}_reporting_receipt.json',receipt);atomic(folder/'reporting_receipt.json',receipt)
def domain(task,ds):
 validate()
 try:
  for command in ['predict','score','report']:
   atomic(D/f'{task}_{ds}_state.json',dict(status=command,worker=identity()))
   with(D/f'{task}_{ds}_{command}.log').open('a')as log:
    c=subprocess.Popen([sys.executable,'-B',__file__,command,'--task',task,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{task}_{ds}_{command}_child.json',identity(c.pid));code=c.wait();atomic(D/f'{task}_{ds}_{command}_exit.json',dict(code=code));assert code==0,'Preserved failure; no automatic retry/fallback'
  atomic(D/f'{task}_{ds}_state.json',dict(status='complete_scored_reported',worker=identity()))
 except BaseException as e:atomic(D/f'{task}_{ds}_state.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc()));raise
def rate():
 rr=[read(p)for p in sorted((CG/'vidstg/receipts').glob('*.json'),key=lambda p:p.stat().st_mtime)[-50:]];vv=[x['pipeline_seconds']for x in rr if not x['acquisition']['acquisition_reused']and x['acquisition']['Source']['Source_forwards']==1];return dict(completed=read(CG/'vidstg_progress.json')['completed'],mean=sum(vv)/len(vv))
def queue():
 p=validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity());baseline=rate();atomic(D/'Vid_baseline.json',baseline);paused=False;pause_at=0.;last=baseline['completed'];live=[]
 try:
  with(D/'preflight.log').open('a')as log:
   c=subprocess.Popen([sys.executable,'-B',__file__,'preflight'],stdout=log,stderr=subprocess.STDOUT);atomic(D/'preflight_child.json',identity(c.pid));assert c.wait()==0,'Preflight failure preserved; no full work'
  for task in ['components','budget']:
   live=[]
   for ds in COUNTS:
    assert mem()>p['limits']['minimum_RAM_bytes'];log=(D/f'{task}_{ds}_domain.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'domain','--task',task,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);w=identity(c.pid);atomic(D/f'{task}_{ds}_domain_child.json',w);live.append((c,log,w))
   atomic(D/'state.json',dict(status=task+'_CPU_running',fingerprint=p['fingerprint'],worker=identity()))
   while any(c.poll()is None for c,l,w in live):
    current=rate();slow=current['completed']-last>=25 and current['mean']>1.35*baseline['mean'];low=mem()<p['limits']['minimum_RAM_bytes'];groups=[w for c,l,w in live]+[read(MAP/'CG_vidstg_domain_child.json')]
    if not paused and(low or slow):
     for w in groups:group_signal(w,signal.SIGSTOP)
     paused=True;pause_at=time.time();last=current['completed'];atomic(D/'resource_guard.json',dict(status='own_CPU_groups_paused',reason='RAM'if low else'Vid_rate_precaution',current=current,time=pause_at))
    elif paused and mem()>p['limits']['resume_RAM_bytes']and time.time()-pause_at>60:
     for w in groups:group_signal(w,signal.SIGCONT)
     paused=False;last=current['completed'];atomic(D/'resource_guard.json',dict(status='own_CPU_groups_resumed',current=current,time=time.time()))
    if not slow and current['completed']-last>=25:last=current['completed']
    time.sleep(5)
   for c,l,w in live:l.close();assert c.returncode==0,'Preserve failed domain, no retry'
  atomic(D/'state.json',dict(status='complete_scored_reported_pending_publication',fingerprint=p['fingerprint']))
 except BaseException as e:
  if paused:
   for w in [w for c,l,w in live]+[read(MAP/'CG_vidstg_domain_child.json')]:group_signal(w,signal.SIGCONT)
  atomic(D/'state.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint']));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','preflight','domain','predict','score','report']);a.add_argument('--task',choices=['components','budget']);a.add_argument('--dataset',choices=list(COUNTS));v=a.parse_args();dict(prepare=prepare,queue=queue,preflight=preflight,domain=lambda:domain(v.task,v.dataset),predict=lambda:predict(v.task,v.dataset),score=lambda:score(v.task,v.dataset),report=lambda:reporting(v.task,v.dataset))[v.command]()
