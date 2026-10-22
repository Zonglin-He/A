"""Actual original pre/post refinement endpoint distributions, fixed spatial method."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_pass_hc64_v1';P=R/'results/native_event_pass_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','RefinementMixture','DirectFirst6','ReversedFirstMixture','SecondSelfMixture','NativeSelfHull','ArchivedSelfHull','NoChange']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');audit=L/'native_pass_official_source_audit';proof=read(audit/'receipt.json')
 for f,v in proof['files'].items():assert sha(audit/f)==v['sha256']
 assert 'model(videos, texts, targets, iteration/max_iter)'in(audit/'scripts/train_net.py').read_text()and 'if iteration_rate < 0:'in(audit/'models/pipeline.py').read_text()
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_pass_hc64_v1',primary='RefinementMixture',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],official_recipe=proof,
 authorization='User bounded overnight event-information feasibility; native source temporal ability allowed, no new expert/evidence and fixed space',rationale='Official training recipe calls positive iteration/max_iter and supervises first pass; evaluation default negative adds query refinement and returns second pass. CurrentG6 uses second only. Test whether retaining actual first-pass ENDPOINT responses improves target-domain boundary decisions; code evidence not private checkpoint training history proof or known root cause.',
 fixed=dict(native='All64 original Full text and prefusion values, original FP16fusion/FP32 native two-pass replay. Scoped ground_decoder forward observer copies actual hidden on calls0,1,2,3 without replacing outputs. First offset hidden from0/2, second1/3. Head each six-layer batch, original offset hull and layer posterior mean. Second hidden/logits/q/Source boxes64bitwise against parent.',primary='.5qFirst6+.5qSecond6, no cross-pass envelope, same original physicalMBR and complete-boxDP.',controls='DirectFirst6 same evidence; time-reflected First positive mixture; SecondSelfMixture exact native; NativeSelfHull on ALL64 pure expansion; archived SelfHull reference original52support plus12native alias; Source/Native/NoChange. All64 primary no grammar/confidence/GT rejection.',space='Original expert8/complete boxes/geometry/DP unchanged. No foreground/actionness values used as NEW feedback/teacher/probability; original source refinement and gates unmodified.'),new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Primary actualtime and completev positive vsSource/G6, improvements>harms, both Source short<=5s andlong>5s timepositive; distinguish time-reflection/self/expansion and benefit beyond same-information DirectFirst, thenfixeddual-domainrecheck. No pass-weight/layer/gate scan or posthoc DirectFirst promotion.',scope='Native pre/post query-refinement endpoint feasibility. First hidden already computed in deployment; additional head/posterior cost only, but research fusion/loading/replay recording must not be presented as deployment timing.',references=[v['url']for v in proof['files'].values()])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','phase_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.visual_cache import transfer
 from stvg.native import FrozenNativeReplay
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_composition_v1.core import envelope
 from research.native_event_pass_v1.core import collect_passes,split_passes
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(new_text_encoder=0,fusion_offsets=0,native_replay_queries=0,original_decoder_calls=0,original_head_six_layer_calls=0,extra_head_six_layer_calls=0,new_decoder_executions_beyond_original=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');phase=torch.load(PHASE/pe['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();views=[]
   for original in old['prefusion']:
    kw=transfer(original,'cuda',clone=True);prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
    try:
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
    finally:h.remove()
    assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
   with torch.no_grad(),collect_passes(model.ground_decoder)as output_hidden:
    base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids))
   assert not model.ground_decoder._forward_hooks;first,second=split_passes(output_hidden);counts['original_decoder_calls']+=4;counts['native_replay_queries']+=1
   for a,b,c in zip(second,base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a,b.cpu())and torch.equal(a,c),'Original second hidden bitwise'
   with torch.no_grad():
    secondlog=[model.temp_embed(h).cpu()for h in base.temporal_inputs];firstlog=[model.temp_embed(h.to('cuda')).cpu()for h in first]
   counts['original_head_six_layer_calls']+=2;counts['extra_head_six_layer_calls']+=4
   for a,b in zip(secondlog,old['views']['Full']['logits']):assert torch.equal(a,b),'Original second all-layer logits bitwise'
   q0=posterior(old['views']['Full']['records'],secondlog,ids);qfirst=posterior(old['views']['Full']['records'],firstlog,ids);assert np.array_equal(q0,old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
   qs=dict(NativeG6=q0,RefinementMixture=.5*q0+.5*qfirst,DirectFirst6=qfirst,ReversedFirstMixture=.5*q0+.5*qfirst[::-1,::-1].T.copy(),SecondSelfMixture=.5*q0+.5*q0,NativeSelfHull=.5*q0+.5*envelope(q0,q0),ArchivedSelfHull=phase['q']['SelfHullFusion'],NoChange=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a,alias in [('NativeG6','NativeG6'),('SecondSelfMixture','NativeG6'),('ArchivedSelfHull','SelfHullFusion'),('NoChange','NativeG6')]:assert np.array_equal(arrays[a],z[alias])and iv[a]==z[alias+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['RefinementMixture'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=dict(First=qfirst,Second=q0),score=score,clauses=dict(supported=True),first_hidden=first,second_hidden=second,first_logits=firstlog,second_logits=secondlog,GT_read=False,second_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['DirectFirst6'],iv['RefinementMixture'],flush=True);del old,phase,cap,geom,arrays,base,views,output_hidden,first,second,firstlog,secondlog;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['fusion_offsets']==128 and counts['original_decoder_calls']==256 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
