"""Shared frozen native grids for ROI, first/final DP and later budget studies."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,gc,hashlib,argparse,fcntl,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,PROVIDER,read,sha,atomic,check,identity,binding
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
ARMS=['SourceROI','FullFrame','GreedyGreedy','DPGreedy','GreedyDP','DPDP']

def prepare():
 assert not(D/'supports_plan.json').exists();check(True)
 p=dict(schema='paper_review_supports_v1',code_sha256=sha(Path(__file__)),selection_sha256=sha(D/'selection.json'),suite_plan_sha256=sha(D/'plan.json'),source_plan_sha256=sha(RUN/'plan.json'),seed=20260920,arms=ARMS,kappa=64,beta=1,rounds=1,context=1.5,new_expert=0,trainable_parameters=0,optimizer_steps=0,
 capture='One original Source224 full native query; pretext projected appearance grid + existing postfusion spatial appearance grid + actual seven TimeDecoder kwargs saved once. Old eight evidence/geometry reused; no teacher invocation. Do not use old pooled vectors for changed auxiliary support.',
 no_op='Every query Source boxes/native final FP32 logits/offset physical frames exact; repooled fulltimeDP features exact current Full; nativeP0 reused only after logit/frame equality; first2query/domain additionally recompute native hull posterior bitwise. DPDP posterior interval and finalboxes exact original.',
 arms_definition=dict(SourceROI='Pool pretext features on native Source boxes, no shared correction. TimeMBR, final SimplePathDP.',FullFrame='Pool all valid pretext visual cells (not enlarged object ROI); timeMBR, final SimplePathDP.',GreedyGreedy='Appearance greedy fulltime auxiliary, REPOOL, fixed native prior+prototype, MBR, final appearancegreedy.',DPGreedy='Original fulltime DP pool/time; final appearancegreedy.',GreedyDP='Greedy fulltime aux REPOOL/time; final exactDP.',DPDP='Original main, exact bitwise no-op.'),
 outcome='Equalquery official t/v/s+observed/unobserved, Source-fixed spatial temporal causal table separately. Screen64parents/domain, fixedconfirm128disjointparents/domain. No tuning/selection. These lists are from previously scored complete datasets, not unseen-data claims.',
 budget=dict(source_queries=384,extra_expert=0,source_wall_limit_seconds=14400,max_shared_cache_bytes=100*2**30),GT_in_inference=False,all_predictions_seal_before_independent_CPU_score=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'supports_plan.json',p);atomic(P/'supports_protocol.json',p);target=D/'source_snapshot/supports.py';target.write_bytes(Path(__file__).read_bytes());print(p['fingerprint'],flush=True)

def frozen():
 p=read(D/'supports_plan.json');assert sha(Path(__file__))==p['code_sha256']and sha(D/'selection.json')==p['selection_sha256'];assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest();check();return p

def greedy(bank):
 import torch
 valid=bank.valid.clone();valid[~bank.active,1:]=False;actions=bank.score_unary.masked_fill(~valid,-torch.inf).argmax(1);return bank.boxes[torch.arange(bank.frames),actions].numpy()

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.query import QuerySubjectParser
 from stvg.representation import observe_native_grid,extract
 from stvg.evidence import extract_visual_grids
 from stvg.temporal_hidden import cache_temporal
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.prototype import one_step
 from stvg.temporal import final_mbr,mass_table,probabilities
 p=frozen();setup();gpu=lease();atomic(D/'supports_predictor.json',identity());selection=read(D/'selection.json');parser=QuerySubjectParser(R/'checkpoints/stanza');cfg=read(R/'configs/default.json');started=time.perf_counter();files={};seen=0
 try:
  for ds,groups in selection.items():
   model=load_source(cfg['datasets'][ds]);files[ds]={};sealed=read(RUN/ds/'sealed.json');domain_i=0
   for stage,rows in groups.items():
    files[ds][stage]={}
    for row in rows:
     assert time.perf_counter()-started<p['budget']['source_wall_limit_seconds'];key=row['id'];dest=D/'supports'/ds;dest.mkdir(parents=True,exist_ok=True);rp=dest/(key+'.json');out=dest/(key+'.npz');shared=dest/(key+'.shared.pt')
     if rp.exists():
      rec=read(rp);assert rec['fingerprint']==p['fingerprint']and sha(out)==rec['sha256']and sha(shared)==rec['shared_sha256'];files[ds][stage][key]=rec;continue
     assert not out.exists()and not shared.exists(),'Preserve unbound partial outputs';tick=time.perf_counter();old=sealed['files'][key];cp=RUN/ds/old['capture_path'];assert sha(cp)==old['capture_sha256']and sha(RUN/ds/old['path'])==old['sha256'];cap=torch.load(cp,map_location='cpu',weights_only=False);assert not cap['GT_read'];receipt=read(RUN/ds/old['receipt_path']);assert receipt['input_binding']==binding(row,read(RUN/'plan.json')['provider_runtime_fingerprint'])
     frames,ids=decode(row['metadata'],ds);assert ids==list(cap['frame_ids']);torch.manual_seed(p['seed']);np.random.seed(p['seed']);model.cuda()
     with observe_native_grid(model)as grids:batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=dict(subject=parser(row['metadata']['caption'])['subject'])))
     assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);assert all(torch.equal(z.cpu(),zz)for z,zz in zip(base.zero['logits'],cap['native_logits']));assert [x['frame_ids']for x in recs]==[x['frame_ids']for x in cap['records']]
     q0=cap['native_prior'];q0recompute=False
     if domain_i<2:
      uu,qq=mass_table(probabilities(recs,base.zero['logits']));assert uu==ids and np.array_equal(qq,q0);q0recompute=True
     spatial,valid,_,shape,spmeta=extract_visual_grids(base.spatial_inputs,ids);temporal=cache_temporal(model,base,recs,ids);model.cpu();del batch,base,frames;gc.collect();torch.cuda.empty_cache()
     geometry=SpatialGeometry.from_cached_geometry(cap['geometry']);bank=geometry.for_interval([ids[0],ids[-1]+1]);auxdp=evidence_dp(bank)['final'].numpy();auxgreedy=greedy(bank);gridcap=dict(frame_ids=ids,native_local_grid=grids);fdp,_=extract(gridcap,auxdp);fgreedy,_=extract(gridcap,auxgreedy);fsource,_=extract(gridcap,cap['Source']);assert np.array_equal(fdp,cap['content_features'])
     fframe=np.empty_like(fdp)
     for off,(a,v)in enumerate(grids):
      for j in range(len(a)):
       aa=a[j].permute(1,2,0).numpy().astype(np.float64);vv=v[j].numpy();fframe[off+2*j]=aa[vv].mean(0)
     feat=dict(DP=fdp,Greedy=fgreedy,SourceROI=fsource,FullFrame=fframe);intervals={};posteriors={};evidence={}
     for name,f in feat.items():
      q,stats=one_step(f,q0);intervals[name]=final_mbr(ids,q)['interval'];posteriors[name]=q;evidence[name]=stats['L']
     arrays=dict(frame_ids=np.asarray(ids),positions8=np.asarray(cap['positions8']),Source=np.asarray(cap['Source']),Source_interval=np.asarray(cap['native_interval']))
     for arm,support,final in [('SourceROI','SourceROI','DP'),('FullFrame','FullFrame','DP'),('GreedyGreedy','Greedy','Greedy'),('DPGreedy','DP','Greedy'),('GreedyDP','Greedy','DP'),('DPDP','DP','DP')]:
      iv=intervals[support];bb=geometry.for_interval(iv);arrays[arm]=evidence_dp(bb)['final'].numpy()if final=='DP'else greedy(bb);arrays[arm+'_interval']=np.asarray(iv)
     with np.load(RUN/ds/old['path'])as z:
      assert np.array_equal(arrays['DPDP'],z['Full'])and np.array_equal(arrays['DPDP_interval'],z['Full_interval']);assert np.array_equal(arrays['DPGreedy'],z['Independent']);arrays['OursS']=z['OursS'].copy()
     shared_payload=dict(schema='review_shared_native_grids_v1',frame_ids=ids,native_local_grid=grids,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape,spatial_grid_metadata=spmeta,temporal=temporal,pooled_features=feat,evidence=evidence,posteriors=posteriors,aux_dp=auxdp,aux_greedy=auxgreedy,parent_capture=str(cp),parent_capture_sha256=old['capture_sha256'],GT_read=False)
     torch.save(shared_payload,shared);np.savez_compressed(out,**arrays);rec=dict(path=str(out.relative_to(D)),sha256=sha(out),shared_path=str(shared.relative_to(D)),shared_sha256=sha(shared),parent_prediction_sha256=old['sha256'],fingerprint=p['fingerprint'],GT_read=False,Source_bitwise=True,native_logit_frames_bitwise=True,q0_recomputed_bitwise=q0recompute,Full_pool_interval_boxes_bitwise=True,source_forwards=1,new_expert=0,seconds=time.perf_counter()-tick,worker=identity());atomic(rp,rec);files[ds][stage][key]=rec;seen+=1;domain_i+=1;atomic(D/'supports_progress.json',dict(dataset=ds,stage=stage,completed=seen,total=384,domain_completed=domain_i,seconds=time.perf_counter()-started,**identity(),GT_read=False));print(json.dumps(dict(dataset=ds,stage=stage,completed=domain_i,seconds=rec['seconds'],native_no_op=True)),flush=True);del shared_payload,cap,grids,gridcap,spatial,temporal,geometry;gc.collect()
    atomic(D/'supports'/ds/(stage+'_sealed.json'),dict(files=files[ds][stage],queries=len(rows),GT_read=False,fingerprint=p['fingerprint']))
   del model;gc.collect();torch.cuda.empty_cache()
  assert sum(len(v)for ds in files.values()for v in ds.values())==384;atomic(D/'supports_sealed.json',dict(status='sealed_before_labels',files=files,queries=384,GT_read=False,fingerprint=p['fingerprint'],seconds=time.perf_counter()-started));print('Shared supports384 sealed',flush=True)
 finally:gpu.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','predict']);a=ap.parse_args();prepare()if a.command=='prepare'else predict()
