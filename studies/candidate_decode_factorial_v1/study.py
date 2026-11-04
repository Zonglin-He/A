"""Fixed-interval candidate inclusion x global decoding; cached CPU only."""
from pathlib import Path
import os,sys,json,time,copy,csv,argparse,functools,ast,hashlib,subprocess,traceback,concurrent.futures
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/candidate_decode_factorial_v1';P=R/'results/candidate_decode_factorial_v1';F=R/'outputs/shared_time_spatial_recalc_v2'
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as old
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
read=old.read;sha=old.sha;atomic=old.atomic;digest=old.digest;identity=old.identity;fixed=functools.lru_cache(None)(read)
COMP=old.selected.D;COUNTS={'hc2':1902,'vidstg':10303};ARMS=['E','AnchorsIndependent','AnchorsDP','TransportIndependent','Full'];NEW=['AnchorsIndependent','AnchorsDP']
PAIRS=[(a,'E')for a in ARMS if a!='E']+[('AnchorsDP','AnchorsIndependent'),('TransportIndependent','AnchorsIndependent'),('Full','TransportIndependent'),('Full','AnchorsDP')]
FIELDS=['v','t','event_unobserved_IoU','event_observed_IoU'];DISTARMS=['E','TransportIndependent','Full'];DISTNAMES=['0_0.5s','0.5_1s','1_2s','2_4s','ge4s']
@functools.lru_cache(None)
def rows(ds):return old.selected.records(ds)
def resources(start):
 old.resources(start)
def validate():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for g in ['code_sha256','input_sha256']:
  for f,h in p[g].items():assert sha(R/f)==h,f
 return p
def npz(root,e):
 assert sha(root/e['path'])==e['sha256']and sha(root/e['receipt_path'])==e['receipt_sha256']
 with np.load(root/e['path'])as z:return {k:z[k].copy()for k in z.files}
def references(ds,key):
 a=fixed(F/ds/'sealed.json')['files'][key];b=fixed(COMP/f'components_{ds}_sealed.json')['files'][key];x=npz(F,a);y=npz(COMP,b)
 assert np.array_equal(x['frame_ids'],y['frame_ids'])and np.array_equal(x['positions8'],y['positions8'])and np.array_equal(x['shared_interval'],y['FullMAP_interval'])and np.array_equal(x['shared_interval'],y['FinalIndependent_interval'])and np.array_equal(x['Full'],y['FullMAP'])
 return dict(frame_ids=x['frame_ids'],positions8=x['positions8'],shared_interval=x['shared_interval'],E=x['SharedCorrection'],TransportIndependent=y['FinalIndependent'],Full=x['Full']),dict(Fresh_reference_sha256=a['sha256'],components_reference_sha256=b['sha256'])
def numpy_dp(bank):
 u=bank.score_unary.numpy();pair=bank.score_pair.numpy();valid=bank.valid.numpy().copy();valid[~bank.active.numpy(),1:]=False;v=np.where(valid[0],u[0],-np.inf);back=[]
 for t in range(1,len(u)):
  q=v[:,None]+pair[t-1]+u[t,None,:];previous=np.argmax(q,axis=0);v=np.where(valid[t],q[previous,np.arange(3)],-np.inf);back.append(previous)
 a=int(v.argmax());acts=[a]
 for prev in reversed(back):a=int(prev[a]);acts.append(a)
 acts=np.array(acts[::-1]);return bank.boxes.numpy()[np.arange(len(acts)),acts],acts
def derive(ds,row,reference=False,preflight=False):
 ref,proof=references(ds,row['id']);c,cp=old.source(ds,row);g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);assert np.array_equal(c['ids'],ref['frame_ids'])and np.array_equal(c['pos'],ref['positions8'])and np.array_equal(c['E'],ref['E']);bank=g.for_interval(ref['shared_interval']);nt=copy.copy(bank);nt.valid=bank.valid.clone();nt.valid[~g.observed,1:]=False
 # Candidate deletion only: retain identical node/link tables, all denominators,
 # original support/geometry validity, observed E/expert and first-index tie rule.
 assert torch.equal(nt.score_unary,bank.score_unary)and torch.equal(nt.score_pair,bank.score_pair)
 assert torch.equal(nt.valid[g.observed],bank.valid[g.observed])and not nt.valid[~g.observed,1:].any()
 mask=nt.valid.numpy();u=nt.score_unary.numpy();acts=np.where(mask,u,-np.inf).argmax(1);ind=nt.boxes.numpy()[np.arange(nt.frames),acts]
 dp,actions=numpy_dp(nt)if reference else (lambda x:(x['final'].numpy(),x['actions'].numpy()))(evidence_dp(nt))
 a={**ref,'AnchorsIndependent':ind,'AnchorsDP':dp,'AnchorsIndependent_actions':acts,'AnchorsDP_actions':actions}
 assert np.array_equal(ind[~g.observed.numpy()],ref['E'][~g.observed.numpy()])and np.array_equal(dp[~g.observed.numpy()],ref['E'][~g.observed.numpy()])
 if preflight or reference:
  z=numpy_dp(bank)[0]if reference else evidence_dp(bank)['final'].numpy();ii=bank.score_unary.masked_fill(~bank.valid,-torch.inf).argmax(1);ind0=bank.boxes[torch.arange(bank.frames),ii].numpy();assert np.array_equal(z,ref['Full'])and np.array_equal(ind0,ref['TransportIndependent'])
 proof.update(parent_payload=cp,GT_read=False,shared_interval_exact=True,node_edge_score_tables_and_weights_unchanged=True,no_transport_only_candidate_valid_mask_changed=True,original_soft_anchor_choices_retained=True,all_native_unobserved_no_transport_boxes_exact_E=True)
 return a,proof
def prepare():
 assert not(D/'plan.json').exists();torch.set_num_threads(1);cases=[];files=[R/'configs/default.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 for ds in COUNTS:
  aa=read(F/ds/'audit_receipt.json');assert aa['metrics_sha256']==sha(F/ds/'metrics.json')
  ca=read(COMP/f'components_{ds}_reporting_receipt.json');assert ca['all_arrays_cold_rederived_exact']and ca['all_means_recomputed']
  files += [F/ds/'sealed.json',F/ds/'metrics.json',F/ds/'audit_receipt.json',COMP/f'components_{ds}_sealed.json',COMP/'components'/ds/'metrics.json',COMP/f'components_{ds}_reporting_receipt.json',old.selected.PREV/'spatial'/f'{ds}_sealed.json']
  job=read(old.selected.RUN/'plan.json')['jobs'][ds];files += [old.selected.PROVIDER/job['manifest'],old.selected.PROVIDER/job['labels']]
  for row in rows(ds)[:2]:
   a,p=derive(ds,row,preflight=True);b,q=derive(ds,row,reference=True);assert p==q and all(np.array_equal(v,b[k])for k,v in a.items());cases.append(dict(dataset=ds,main_Full_and_independent_bitwise_existing=True,new_anchors_independent_and_DP_numpy_repeat_exact=True,unobserved_native_no_transport_E_exact=True))
 code=[Path(__file__),R/'stvg/spatial.py',R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'stvg/metrics_hc.py',R/'studies/shared_time_spatial_recalc_v2/score.py',R/'studies/reviewer_key_controls_v1/common_v1.py',R/'studies/shared_corrected_roi_v1/shared_common.py']
 p=dict(schema='full_fixed_time_candidate_inclusion_x_decoder_v1',authorization='2026-11-04 user requests five-row candidate generation x trajectory decoding plus distance supplement',counts=COUNTS,arms=ARMS,new_arms=NEW,pairs=PAIRS,Full_interval='Exact immutable current Full interval per query; E/experts/features/eightpositions/geometry/prototype/parameters unchanged; no temporal inference or new features.',no_transport='Remove only valid transported candidate1/2 at unobserved native nodes. Keep E everywhere and original legal expert candidates at observed nodes. Preserve common original node/edge scores, support and normalization. Candidate-inclusion experiment, not end-to-end deletion/refit of all geometry/appearance computation.',independent='Masked per-node original score_unary argmax, first-index ties. Not interpolation.',DP='Original exact global recurrence with unchanged score_unary/score_pair/validity/weights; no hard anchor clamp.',reuse='E/Full all original official and dense-conditional metrics reused by exact arrays/SHA; TransportIndependent original official v/t/recalls reused, new common-event dense unobserved/observed IoUs computed. Only two new anchor controls require official tube scoring.',metric='vIoU over whole exact full query list; event_unobserved_IoU conditional querymean over common predicted/GT interval annotated dense frames excluding exact8physicalobservations. Same support all5. Empty conditional support null, primary v retains zeros/allqueries. Not original128 all-video native-grid IoU.',distance='Same conditional dense unobserved support bins0-.5/.5-1/1-2/2-4/>=4seconds for E/TransportIndependent/Full; exactphysicaltime nearest8. Perquery thenparentCI, bin empty null. Official densification means changing observed native boxes can affect nearby unobserved dense frames even without transported candidates.',statistics='All signed/severe tvuniononce>5pp/rescues/short<=5s-long; 10000parentcluster bootstrap20261022 pointwise/exposed/not tuning; include2x2interaction.',resources='max2CPU14-15/nice19/ionice3/one thread/CUDAhidden; RAM8GiB/disk80GiB/private160GiB/domain72h',phase='Whole domain new prediction seal before independent officialCPU score, separate allarray/inputSHA/independent DP+normalizedIoU/means audit; ROOT then reporting',new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0,old_frozen_or_method_or_manuscript_changed=False,preflight=cases,code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R))if f.is_relative_to(R)else str(f):sha(f)for f in files})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'],flush=True)
def predict(ds):
 p=validate();old.infer_guard();torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;assert not(dest/'sealed.json').exists();files={}
 for i,row in enumerate(rows(ds),1):
  resources(start);tick=time.perf_counter();a,proof=derive(ds,row);f=dest/'predictions'/(row['id']+'.npz');f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists()
  with f.open('wb')as out:np.savez_compressed(out,**a)
  with np.load(f)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(f.relative_to(D)),sha256=sha(f),proof=proof,worker=identity(),GT_read=False,conditional_cached_query_seconds=time.perf_counter()-tick);rp=dest/'receipts'/(row['id']+'.json');atomic(rp,rec);files[row['id']]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  if i%100==0:atomic(dest/'predict_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 atomic(dest/'sealed.json',dict(queries=len(files),files=files,GT_read=False,worker=identity(),plan_fingerprint=p['fingerprint'],elapsed_seconds=time.perf_counter()-start));print(ds,'whole prediction seal',flush=True)
def xyxy(b,w,h):
 return np.column_stack([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2])*[w,h,w,h]
def metric_functions():
 path=R/'studies/shared_time_spatial_recalc_v2/score.py';tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['pixels_at_gt','measure']];assert len(nodes)==2;scope=dict(np=np,xyxy=xyxy);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope['measure'],scope['pixels_at_gt']
def distance_values(z,row,g,reference=False):
 _,pixels=metric_functions();gt={int(k):b for k,b in g['boxes'].items()};f=np.array(sorted(gt));truth=np.array([gt[int(x)]for x in f],float);m=row['metadata'];a,b=z['shared_interval'];c,d=g['interval'];event=(f>=max(a,c))&(f<min(b,d));obs=np.isin(f,z['frame_ids'][z['positions8']]);dist=np.abs(f[:,None]-z['frame_ids'][z['positions8']][None,:]).min(1)/m['fps'];ix=np.searchsorted([0,.5,1,2,4,np.inf],dist,side='right')-1;result={name:dict(count=0,values={})for name in DISTNAMES}
 for arm in DISTARMS:
  pred,valid=pixels(z,arm,f,m,row['dataset'],reference=reference);target=truth/np.array([m['width'],m['height']]*2)if reference else truth;assert valid[event].all();inter=np.maximum(0,np.minimum(pred[:,2:],target[:,2:])-np.maximum(pred[:,:2],target[:,:2])).prod(1);den=np.maximum(pred[:,2:]-pred[:,:2],0).prod(1)+np.maximum(target[:,2:]-target[:,:2],0).prod(1)-inter;iou=np.divide(inter,den,out=np.zeros(len(f)),where=den>0)
  for j,name in enumerate(DISTNAMES):mask=event&~obs&(ix==j);result[name]['count']=int(mask.sum());result[name]['values'][arm]=dict(IoU=float(iou[mask].mean())if mask.any()else None)
 return result
def score(ds):
 validate();torch.set_num_threads(1);sys.addaudithook(old.deny_gpu_media);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=old.selected.labels(ds);fr={r['id']:r for r in fixed(F/ds/'metrics.json')['private_rows']};cr={r['id']:r for r in fixed(COMP/'components'/ds/'metrics.json')['private_rows']};rr=[];dd={b:[]for b in DISTNAMES};checks=0;mx=0.;measure,_=metric_functions()
 from stvg.metrics import official
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};values={}
  with np.load(D/e['path'])as z:
   for arm in ARMS:
    v=measure(z,arm,row['metadata'],gt,g['interval'],ds)
    if arm in ['E','Full']:
     ref=fr[key]['values']['SharedCorrection'if arm=='E'else'Full'];assert set(v)==set(ref)and all((x is None and ref[k]is None)or(x is not None and abs(x-ref[k])<2e-10)for k,x in v.items());v=ref
    elif arm=='TransportIndependent':assert all(abs(v[k]-cr[key]['values']['FinalIndependent'][k])<2e-10 for k in ['v','t'])
    else:
     gold=official(z[arm],dict(input=row['metadata'],frame_ids=z['frame_ids'].tolist()),gt,g['interval'],z['shared_interval'],ds);err=max(abs(v[k]-gold[k])for k in ['t','v']);mx=max(mx,err);assert err<2e-10;checks+=1
    values[arm]=v
   assert len({v['t']for v in values.values()})==1
   for f in ['event_GT_frames','event_observed_frames','event_unobserved_frames']:assert len({v[f]for v in values.values()})==1
   bins=distance_values(z,row,g)
   for b,v in bins.items():dd[b].append(dict(id=key,parent=str(g['parent']),values=v['values'],annotated_frames=v['count']))
  rr.append(dict(id=key,parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=values))
  if i%100==0:atomic(dest/'score_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 atomic(dest/'metrics.json',dict(private_rows=rr));atomic(dest/'distance_rows.json',dd);atomic(dest/'score_receipt.json',dict(status='whole_domain_scored_pending_independent_audit',queries=len(rr),new_official_checks=checks,reused_official_arm_queries=3*len(rr),max_official_error=mx,seal_sha256=sha(dest/'sealed.json'),metrics_sha256=sha(dest/'metrics.json'),distance_rows_sha256=sha(dest/'distance_rows.json'),worker=identity(),elapsed_seconds=time.perf_counter()-start));print(ds,'score complete',checks,flush=True)
def audit(ds):
 validate();torch.set_num_threads(1);sys.addaudithook(old.deny_gpu_media);start=time.perf_counter();dest=D/ds;seal=read(dest/'sealed.json');sc=read(dest/'score_receipt.json');assert sc['metrics_sha256']==sha(dest/'metrics.json')and sc['distance_rows_sha256']==sha(dest/'distance_rows.json')and sc['seal_sha256']==sha(dest/'sealed.json');metrics={r['id']:r for r in read(dest/'metrics.json')['private_rows']};distance=read(dest/'distance_rows.json');truth=old.selected.labels(ds);measure,_=metric_functions();arrays=scalars=0;mx=0.
 for i,row in enumerate(rows(ds),1):
  resources(start);key=row['id'];a,proof=derive(ds,row,reference=True);e=seal['files'][key];assert proof==e['proof']and sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=truth[key];gt={int(k):v for k,v in g['boxes'].items()};assert metrics[key]['parent']==str(g['parent'])
  with np.load(D/e['path'])as z:
   assert set(a)==set(z.files)
   for k,v in a.items():assert np.array_equal(v,z[k]),(key,k);arrays+=1
   for arm in ARMS:
    values=measure(z,arm,row['metadata'],gt,g['interval'],ds,reference=True)
    for k,x in values.items():
     y=metrics[key]['values'][arm][k];assert(x is None)==(y is None)
     if x is not None:err=abs(x-y);mx=max(mx,err);assert err<2e-10;scalars+=1
   bins=distance_values(z,row,g,reference=True)
   for b,v in bins.items():
    ref=distance[b][i-1];assert ref['id']==key and ref['parent']==str(g['parent'])and ref['annotated_frames']==v['count']
    for arm in DISTARMS:
     x,y=v['values'][arm]['IoU'],ref['values'][arm]['IoU'];assert(x is None)==(y is None)
     if x is not None:assert abs(x-y)<2e-10;scalars+=1
  if i%100==0:atomic(dest/'audit_progress.json',dict(completed=i,total=COUNTS[ds],worker=identity()))
 atomic(dest/'audit_receipt.json',dict(status='passed_all_candidate_masks_SHA_arrays_numpyDP_independent_normalized_denseIoU_distance_parent_support',queries=len(metrics),array_checks=arrays,scalar_checks=scalars,max_scalar_error=mx,score_sha256=sha(dest/'score_receipt.json'),metrics_sha256=sha(dest/'metrics.json'),distance_sha256=sha(dest/'distance_rows.json'),seal_sha256=sha(dest/'sealed.json'),worker=identity(),elapsed_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_GPU=0));print(ds,'audit complete',flush=True)
def cluster(rr,a,b,field):
 parents=sorted({r['parent']for r in rr});n=len(parents);index={p:i for i,p in enumerate(parents)};s=np.zeros(n);c=np.zeros(n);d=[]
 for r in rr:
  x,y=r['values'][a][field],r['values'][b][field]
  if x is None or y is None:continue
  v=100*(x-y);d.append(v);j=index[r['parent']];s[j]+=v;c[j]+=1
 if not d:return dict(mean=None,CI95=None,queries=0,parents=0,improved=0,harmed=0,unchanged=0)
 w=np.random.default_rng(20261022).multinomial(n,[1/n]*n,size=10000);den=w@c;z=(w@s)[den>0]/den[den>0];v=np.asarray(d)
 return dict(mean=float(v.mean()),CI95=np.quantile(z,[.025,.975]).tolist(),queries=len(d),parents=int((c>0).sum()),improved=int((v>1e-10).sum()),harmed=int((v< -1e-10).sum()),unchanged=int((abs(v)<=1e-10).sum()),severe_decline_gt5pp=int((v< -5).sum()),rescue_gt5pp=int((v>5).sum()))
def report(ds):
 validate();dest=D/ds;aud=read(dest/'audit_receipt.json');assert aud['metrics_sha256']==sha(dest/'metrics.json');rr=read(dest/'metrics.json')['private_rows'];dd=read(dest/'distance_rows.json');folder=P/ds;means={a:{f:float(np.mean(v))if(v:=[r['values'][a][f]for r in rr if r['values'][a][f]is not None])else None for f in rr[0]['values'][a]}for a in ARMS};paired={a+'-'+b:{f:cluster(rr,a,b,f)for f in FIELDS}for a,b in PAIRS};groups={g:{a+'-'+b:{f:cluster(qq,a,b,f)for f in FIELDS}for a,b in PAIRS}for g in ['short_le5s','long_gt5s']if(qq:=[r for r in rr if r['duration_group']==g])};distance={}
 for name,qq in dd.items():distance[name]=dict(annotated_frames=sum(r['annotated_frames']for r in qq),scorable_queries=sum(r['annotated_frames']>0 for r in qq),means={a:float(np.mean(v))if(v:=[r['values'][a]['IoU']for r in qq if r['values'][a]['IoU']is not None])else None for a in DISTARMS},paired={a+'-E':cluster(qq,a,'E','IoU')for a in ['TransportIndependent','Full']})
 interaction=[]
 for r in rr:
  v=r['values'];values={}
  for f in FIELDS:
   z=[v[a][f]for a in ['Full','TransportIndependent','AnchorsDP','AnchorsIndependent']];values[f]=None if any(x is None for x in z)else z[0]-z[1]-z[2]+z[3]
  interaction.append(dict(parent=r['parent'],values=dict(interaction=values,zero={f:0. if x is not None else None for f,x in values.items()})))
 inter={f:cluster(interaction,'interaction','zero',f)for f in FIELDS};result=dict(queries=len(rr),parents=len({r['parent']for r in rr}),means=means,paired=paired,timegroups=groups,distance=distance,transport_DP_interaction=inter,score_support='same dense annotated unobserved frames inside common predicted/GT event; empty support null, whole primary queries retained',source_audit_sha256=sha(dest/'audit_receipt.json'));atomic(folder/'report.json',result)
 with (folder/'all_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['query','group','arm','tIoU','vIoU','event_unobserved_IoU','event_observed_IoU','event_unobserved_frames','delta_v_vs_E_pp'])
  for i,r in enumerate(rr,1):
   for a in ARMS:v=r['values'][a];w.writerow([f'Q{i:05}',r['duration_group'],a,v['t'],v['v'],v['event_unobserved_IoU'],v['event_observed_IoU'],v['event_unobserved_frames'],100*(v['v']-r['values']['E']['v'])])
 atomic(dest/'reporting_receipt.json',dict(status='whole_domain_scored_audited_reported_pending_ROOT',queries=len(rr),report_sha256=sha(folder/'report.json'),audit_sha256=sha(dest/'audit_receipt.json'),worker=identity()));print(ds,'report complete',flush=True)
def domain(ds):
 for stage in ['predict','score','audit','report']:
  start=time.perf_counter();log=D/'logs'/(ds+'_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('wb')as f:
   p=subprocess.Popen([str(R/'.conda/tubedetr/bin/python'),'-B',str(Path(__file__)) ,stage,'--dataset',ds],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy());w=identity(p.pid);atomic(D/'children'/(ds+'_'+stage+'.json'),w);atomic(D/(ds+'_state.json'),dict(status='running_'+stage,worker=w));code=p.wait()
  atomic(D/'exits'/(ds+'_'+stage+'.json'),dict(stage=stage,worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
 atomic(D/(ds+'_state.json'),dict(status='whole_domain_complete_PENDING_ROOT'))
def queue():
 validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   futures=[pool.submit(domain,ds)for ds in COUNTS]
   for f in futures:f.result()
  atomic(D/'state.json',dict(status='both_full_candidate_decode_factorial_scored_audited_pending_ROOT',new_Source=0,new_expert=0,new_GPU=0))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),all_old_and_new_outputs_preserved=True));raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','predict','score','audit','report','queue']);parser.add_argument('--dataset',choices=COUNTS);args=parser.parse_args();globals()[args.stage](*([]if args.stage in ['prepare','queue']else[args.dataset]))
