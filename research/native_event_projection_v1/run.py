"""Fixed old temporal-query optimizer with newly measured positive clause feedback."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_projection_hc64_v1';P=R/'results/native_event_projection_hc64_v1';PARENT=R/'outputs/native_event_shared_query_hc64_v1'
ARMS=['NativeG6','ClauseQuery','SharedQueryClauseFusion','SelfHullQuery','SelfHullFusion','ReversedClauseQuery','ReversedClauseDirect','NoUpdate']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(PARENT/'sealed.json')==p['parent_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PARENT/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/event_query_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_projection_hc64_v1',primary='ClauseQuery',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],parent_files=read(PARENT/'sealed.json')['files'],parent_seal_sha256=sha(PARENT/'sealed.json'),labels_sha256=parent['labels_sha256'],supported=parent['supported'],
 authorization='User overnight genuine event feedback research and temporary native state allowed; no extra expert/spatial change',rationale='New positive clause direct feedback gains2.5820pp time/1.7062pp completev but short harms persist. Full query sharing changes little. Test whether fitting same feedback through full-caption native temporal-query family filters incompatible clause outputs rather than simply reading their mixture. Earlier RawQuery failed with ReferDINO event feedback; optimizer/step budget is not increased here, only feedback differs.',
 difference_from_old='Reuse exact old256 shared-offset temporary query parameterization, Adam lr.01/10steps/radius.1, last state. Old event-token targets negative; current primary target is already sealed same-native positive SharedQueryClauseFusion, not normalized ReferDINO. No new query interface, OPD or optimizer novelty claimed.',
 fixed=dict(steps=10,lr=.01,radius=.1,seed=20260920,parameter='fresh disposable256 u, delta=native Full query RMS*u, shared offsets, normalized RMS ball.1; source decoder/head weights frozen eval',objective='Forward cross entropy to fixed parent shared-clause posterior, differentiable exact sixlayer/twooffset native hull, no per-query temperature or acceptance; last step only',controls='same-information Direct parent; SelfHullQuery identical10steps on self-hull target and SelfHullDirect; ReversedClauseQuery same10steps on reversed clause posterior and its direct readout; zero-state/NoUpdate native',scope='Same52 supported clause members, all52 actually fitted for all3 query arms; other12 exactNative by inherited grammar, not quality gate',space='unchanged geometry/expert8/per-interval DP; fixed-space metrics also; all64 predictions seal before new scoring'),
 new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_network_training=False,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification=parent['qualification'],references=parent['references'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','parent_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal_hidden import verify_decoder
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.event_query_v1.core import restore_inputs,torch_posterior
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU_query_fit',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();decoder=model.ground_decoder.time_decoder;head=model.temp_embed;verify_decoder(decoder);files={};maxerr=0;counts=dict(native_decoder_calls=0,optimizer_steps=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['parent_files'][k];assert sha(PARENT/e['path'])==e['sha256']and sha(PARENT/e['trace_path'])==e['trace_sha256'];old=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];offsets=cap['temporal_cache']['offsets'];kw=[restore_inputs(x,'cuda')for x in offsets];recs=cap['records'];tick=time.perf_counter();logs=[]
   with torch.no_grad():
    for inp,saved in zip(kw,offsets):
     hidden=decoder(**inp);counts['native_decoder_calls']+=1;assert torch.equal(hidden.cpu(),saved['hidden']),'Exact native layout/hidden restoration';z=head(hidden);assert torch.equal(z[-1].cpu(),saved['native_logits']);logs.append(z)
   q0=posterior(recs,[z.cpu()for z in logs],ids);assert np.array_equal(q0,old['q']['NativeG6']);qt=torch_posterior(recs,logs,ids);err=float(abs(qt.cpu().numpy()-q0).max());assert err<2e-12;maxerr=max(maxerr,err);qs={a:q0.copy()for a in ARMS};paths={};targets={}
   if e['supported']:
    targets=dict(ClauseQuery=old['q']['SharedQueryClauseFusion'],SelfHullQuery=old['q']['SelfHullFusion'],ReversedClauseQuery=old['q']['ReverseSharedClauses']);qs.update(SharedQueryClauseFusion=targets['ClauseQuery'],SelfHullFusion=targets['SelfHullQuery'],ReversedClauseDirect=targets['ReversedClauseQuery']);scale=float(torch.stack([x['query_tgt'][0,0].square().mean().sqrt()for x in kw]).mean());assert scale>0
    for arm,t in targets.items():
     u=torch.zeros(256,device='cuda',requires_grad=True);opt=torch.optim.Adam([u],lr=p['fixed']['lr']);tt=torch.as_tensor(t,dtype=torch.float64,device='cuda');path=[]
     for step in range(p['fixed']['steps']+1):
      zz=[]
      for x in kw:
       new=dict(x);new['query_tgt']=(x['query_tgt'][:1]+scale*u[None,None,:]).expand_as(x['query_tgt']);zz.append(head(decoder(**new)));counts['native_decoder_calls']+=1
      q=torch_posterior(recs,zz,ids);loss=-(tt*torch.log(q.clamp_min(1e-300))).sum();assert torch.isfinite(loss)
      if step==0:assert float(abs(q.detach().cpu().numpy()-q0).max())<2e-12,'Zero query state must preserve native'
      rec=dict(step=step,loss=float(loss.detach()),normalized_delta_RMS=float(u.detach().norm()/16))
      if step<p['fixed']['steps']:
       opt.zero_grad();loss.backward();rec['gradient_norm']=float(u.grad.norm());assert np.isfinite(rec['gradient_norm']);opt.step();counts['optimizer_steps']+=1
       with torch.no_grad():u.mul_(min(1.,p['fixed']['radius']*16/max(float(u.norm()),1e-30)))
      else:qs[arm]=posterior(recs,[x.detach().cpu()for x in zz],ids)
      path.append(rec);del zz,q,loss
     paths[arm]=dict(path=path,delta=(scale*u.detach()).cpu(),scale=scale,steps=p['fixed']['steps']);del u,opt
   iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PARENT/e['path'],allow_pickle=False)as z:
    for a in ['NativeG6','SharedQueryClauseFusion','SelfHullFusion']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['ClauseQuery'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=score,clauses=old['clauses'],paths=paths,targets=targets,posterior_math_error=err,GT_read=False,native_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=e['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,e['supported'],iv['NativeG6'],iv['ClauseQuery'],flush=True);del old,cap,kw,logs,qt,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();assert all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,posterior_math_max_error=maxerr,network_weights_frozen=True,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
