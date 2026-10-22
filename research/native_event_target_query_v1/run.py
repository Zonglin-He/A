"""Same historical query optimizer, newly measured target-semantic feedback."""
import os,sys,time,json,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put
D=R/'outputs/native_event_target_query_v1';P=R/'results/native_event_target_query_v1';PARENT=R/'outputs/native_event_target_semantic_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','TargetQuery','DirectTarget','ReflectedQuery','DirectReflected','GlobalSemanticQuery','DirectGlobal','NoUpdate']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(PARENT/'sealed.json')==p['parent_seal_sha256']and sha(PARENT/'score_receipt.json')==p['parent_score_receipt_sha256']
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PARENT/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/event_query_v1/core.py','research/event_query_v1/test_core.py','research/event_query_v1/legacy_event.py','research/native_event_semantic_recheck_v1/score.py','research/native_event_projection_v1/run.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_target_query_v1',primary='TargetQuery',arms=ARMS,count=128,items=parent['items'],labels=parent['labels'],development_labels_sha256=parent['development_labels_sha256'],code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],parent_files=read(PARENT/'sealed.json')['files'],parent_plan_sha256=sha(PARENT/'plan.json'),parent_seal_sha256=sha(PARENT/'sealed.json'),parent_score_receipt_sha256=sha(PARENT/'score_receipt.json'),seed=20260920,bootstrap_seed=20261022,maximum_GPU_seconds=1200,stop_epoch=1792708800,authorization='User explicitly accepts adaptingquery inovernighteventresearch; fixedspace/same8expertobservations/noaddedexpert',rationale='MeasuredtrueTargetSemantic directq improvest+4.315/v+1.898HC64 with36win14harm2severe andSource short+2.231, stronger/lessharmthanoldpositiveclauses; fixedVid32meanpositivebutshortnegativeand9win16harm. Test oldnativequeryfamilyprojection withsame10stepbudget, notmoresteps/newinterface/mereReferDINOscore. EarlieroldRawQuery/ClauseQuerynegativepreserved.',fixed=dict(parameter='EXACToldfreshdisposableu256 sharedoffsets;delta=meanoriginalqueryRMS*u;normalizedRMSballradius.1. AllSourceweights eval/frozen, no sourcefinetuning.',optimizer='EXACToldAdam lr.01/10steps/radius.1,laststate,no beststep/qualitygate,all128executeALL3armsandALL10steps. Resetu/Adamperquery/arm.',target='FixedsealedposteriorTargetSemanticLoop2. Crossentropyteacherq vsdifferentiableexactSIXlayer TWOoffset oldhull posterior. q0 alreadyusedinsemanticteacher; notindependentteacherverification.',controls='DirectTarget sameexactsealedposterior/information; ReflectedQuery sameoptimizer onReflectedSemanticLoop2 andDirectReflected; GlobalSemanticQuery sameoptimizer onParentGlobalSemanticLoop2 andDirectGlobal; originalzero/native/NoUpdate. No extra teacherrefresh/newvisual/text.',scope='OnlyfeedbackchangedfromoldClauseQuery, interface/hyperparametersunchanged. TeacherSource wordcoordinatesdensearenotcalibratedeventprobabilities. NotOPD/weightTTA/learnedheadtraining. Nativefamilymayfitincorrecttargets; requireactualtask/controlbenefit, notlossdecrease.'),qualification='EVERYHC64/HC32/Vid32Source/G6actualt/fullvpositive,Source shortANDlongpositive,win>harm,beyondReflectedQuery/GlobalSemanticQuery ANDsameinformationDirectTargett/fullv;no fallbackpromotion/no steps/lr/radius scan.',new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_ASA_head=0,new_expert=0,new_observations=0,new_network_training=False,query_optimization=True,GT_in_inference=False,all128_seal_before_GT=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','code','paper_code','provider_sources','assets','parent_files']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal_hidden import verify_decoder
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.event_query_v1.core import restore_inputs,torch_posterior
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU_query_fit',fingerprint=p['fingerprint'],**who()));assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items());files={};maxerr=0.;counts=dict(time_decoder_calls=0,optimizer_steps=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_ASA_head=0,new_expert=0,new_observations=0,new_network_training=0)
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();decoder=model.ground_decoder.time_decoder;head=model.temp_embed;verify_decoder(decoder)
   for item in [x for x in p['items']if x['dataset']==ds]:
    verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=item['alias'];e=p['parent_files'][k];assert sha(PARENT/e['path'])==e['sha256']and sha(PARENT/e['trace_path'])==e['trace_sha256'];old=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');cf=PROVIDER/item['capture']['path'];assert sha(cf)==item['capture']['sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];offsets=cap['temporal_cache']['offsets'];kw=[restore_inputs(x,'cuda')for x in offsets];recs=cap['records'];tick=time.perf_counter();logs=[]
    with torch.no_grad():
     for inp,saved in zip(kw,offsets):
      hidden=decoder(**inp);counts['time_decoder_calls']+=1;assert torch.equal(hidden.cpu(),saved['hidden']),'Nativehiddenbitwise';z=head(hidden);assert torch.equal(z[-1].cpu(),saved['native_logits']);logs.append(z)
    q0=posterior(recs,[z.cpu()for z in logs],ids);assert np.array_equal(q0,old['q']['NativeG6']);qt=torch_posterior(recs,logs,ids);err=float(abs(qt.cpu().numpy()-q0).max());assert err<2e-12;maxerr=max(maxerr,err);targets=dict(TargetQuery=old['q']['TargetSemanticLoop2'],ReflectedQuery=old['q']['ReflectedSemanticLoop2'],GlobalSemanticQuery=old['q']['ParentGlobalSemanticLoop2']);qs=dict(NativeG6=q0,DirectTarget=targets['TargetQuery'],DirectReflected=targets['ReflectedQuery'],DirectGlobal=targets['GlobalSemanticQuery'],NoUpdate=q0.copy());paths={};scale=float(torch.stack([x['query_tgt'][0,0].square().mean().sqrt()for x in kw]).mean());assert scale>0
    for arm,t in targets.items():
     u=torch.zeros(256,device='cuda',requires_grad=True);opt=torch.optim.Adam([u],lr=.01);tt=torch.as_tensor(t,dtype=torch.float64,device='cuda');path=[]
     for step in range(11):
      zz=[]
      for x in kw:
       new=dict(x);new['query_tgt']=(x['query_tgt'][:1]+scale*u[None,None,:]).expand_as(x['query_tgt']);zz.append(head(decoder(**new)));counts['time_decoder_calls']+=1
      q=torch_posterior(recs,zz,ids);loss=-(tt*torch.log(q.clamp_min(1e-300))).sum();assert torch.isfinite(loss)
      if step==0:assert float(abs(q.detach().cpu().numpy()-q0).max())<2e-12,'ZeroquerystateNativeexactmath'
      rec=dict(step=step,loss=float(loss.detach()),normalized_delta_RMS=float(u.detach().norm()/16))
      if step<10:
       opt.zero_grad();loss.backward();rec['gradient_norm']=float(u.grad.norm());assert np.isfinite(rec['gradient_norm']);opt.step();counts['optimizer_steps']+=1
       with torch.no_grad():u.mul_(min(1.,.1*16/max(float(u.norm()),1e-30)))
      else:qs[arm]=posterior(recs,[x.detach().cpu()for x in zz],ids)
      path.append(rec);del zz,q,loss
     paths[arm]=dict(path=path,delta=(scale*u.detach()).cpu(),scale=scale,steps=10);del u,opt
    iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
    with np.load(PARENT/e['path'],allow_pickle=False)as z:
     for a,b in [('NativeG6','NativeG6'),('NoUpdate','NativeG6'),('DirectTarget','TargetSemanticLoop2'),('DirectReflected','ReflectedSemanticLoop2'),('DirectGlobal','ParentGlobalSemanticLoop2')]:assert np.array_equal(arrays[a],z[b])and iv[a]==z[b+'_interval'].tolist()
     arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
    f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,paths=paths,targets=targets,posterior_math_error=err,GT_read=False,native_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick);put(D/'progress.json',dict(completed=len(files),total=128,counts=counts,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['TargetQuery'],flush=True);del old,cap,kw,logs,qt,geom,arrays;gc.collect();torch.cuda.empty_cache()
   assert all(not x.requires_grad and x.grad is None for x in model.parameters());del model;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['optimizer_steps']==3840 and counts['time_decoder_calls']==8704;put(D/'sealed.json',dict(count=128,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,posterior_math_max_error=maxerr,network_weights_frozen=True,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),**who(),partial_preserved=True));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
