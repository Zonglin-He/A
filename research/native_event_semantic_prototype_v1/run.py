"""Retain positive target context before genuine native multimodal inference."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_semantic_prototype_hc64_v1';P=R/'results/native_event_semantic_prototype_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','SemanticPrototypeLoop2','OneStepPrototype','DirectSemanticEvidence','ReversedSemanticLoop2','RandomProjectionLoop2','AppearancePrototypeLoop2','SharpenOriginal','NoProjection','ConstantProjectionNoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_semantic_prototype_hc64_v1',primary='SemanticPrototypeLoop2',arms=ARMS,count=64,seed=20260920,bootstrap_seed=20261022,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],design_sha256=sha(L/'next_semantic_prototype_design.md'),source_audit_sha256=sha(L/'semantic_projection_official_source_audit/receipt.json'),authorization='User active overnight frozen-native event research, no addedexpert/observations, fixedspace; deadline07AsiaShanghai',fixed=dict(projection='Original SpatialActivation.head BEFOREmean, denseallframes learnedmotionword34 coordinates andappearance20 rolecontrol; inputu256 onlyforrandomcontrol. Framewise extrapolation frompooledBCE, noteventprobability, noactionnessfeedback.',prototype='Sigmoidfloat64 clip1e-6, crossfitoppositeparity softNativeG6inclusiveeventmembership; centeredpositive/negativeclassmeans. Exactly2iterations, fixedq0prior eachtime, classSUM expectedBernoulli loglikelihoodratio, inclusiveframeSUM. Nooptimizer/gain/temperature/gate.',controls='OneStep;DirectusesSAMEsecondstepSwithuniformlegalprior;reverseall65semanticrows;fixedGaussianu256projection rankmappedEXACTperclassrealphistogram;appearancewordhead20;normalizedq0square;constantclassfeaturesL0/nativeexact. All64 retained.',native='OriginalFullcaption/original2D3D/all65nodes/native2pass/sixlayer2offsethull/physicalMBR/completeboxDP/eightobservations;Fullhidden/logits/q/Sourceboxesbitwise.',limitations='Class/framepseudolikelihoodcorrelated; crossoffsetfeaturesandq0arenotindependentvalidation; learnedwordspaceTA-onlyinterface, notOPD/DSEreproduction. NoRGB/visualencoder/textencoder/expert/observations/optimizer. No gain/iteration/projectioncalibrationscan.'),new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='SourceandG6actualt/fullvpositive,win>harm,Source shortANDlongpositive,beyondmatchedrandom/reverse/sharpen;rolecontrolreported,thenfixeddual-domainrecheck. No posthoccontrolpromotion.',references=['https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/classifier.py'])
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
 from research.native_event_semantic_prototype_v1.core import loop,matched_random,update
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(fusion_offsets=0,native_replay_queries=0,extra_dense_wordhead_forwards=0,new_text_encoder=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);views=[];cost={}
   for off,original in enumerate(old['prefusion']):
    kw=transfer(original,'cuda',clone=True);prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
    try:
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
    finally:hook.remove()
    assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
   with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
   q0=posterior(old['views']['Full']['records'],logs,ids);counts['native_replay_queries']+=1
   assert np.array_equal(q0,old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
   for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Fullhiddenbitwise'
   for a,b in zip(logs,old['views']['Full']['logits']):assert torch.equal(a,b),'Fulllogitsbitwise'
   cost['Full_noop_fusion_native']=time.perf_counter()-tick;tt=time.perf_counter();obs={'motion':[],'appearance':[]}
   with torch.no_grad(),torch.autocast('cuda',enabled=False):
    for off,v in enumerate(base.views):
     H=model.ground_encoder.encoder.norm(v['prefix']);h,w=v['info']['fea_map_size'];count=h*w;nf=H.shape[1];ft=H[count:-count].mean(1).unsqueeze(0)
     for role,head,part in [('motion',model.t_spatial_clas,H[-count:]),('appearance',model.s_spatial_clas,H[:count])]:
      features=part.permute(1,2,0).reshape(nf,256,h,w).detach();inputs=[];outputs=[];hooks=[head.head.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().cpu().clone())),head.head.register_forward_hook(lambda m,a,o:outputs.append(o.detach().cpu().clone()))]
      try:head(features,ft[:,:1])
      finally:
       for hook in hooks:hook.remove()
      assert len(inputs)==len(outputs)==1;assert inputs[0].shape==(nf,1,256)and outputs[0].shape==(nf,1,head.vocab_size);obs[role].append(dict(z=outputs[0][:,0].numpy(),u=inputs[0][:,0].numpy()));counts['extra_dense_wordhead_forwards']+=1
   torch.cuda.synchronize();cost['extra_dense_heads']=time.perf_counter()-tt;tt=time.perf_counter();ps={};us={}
   for role in obs:
    z=np.stack([obs[role][j%2]['z'][j//2]for j in range(len(ids))]).astype(np.float64);ps[role]=np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6);us[role]=np.stack([obs[role][j%2]['u'][j//2]for j in range(len(ids))])
   real,steps=loop(ps['motion'],q0);randomp=matched_random(ps['motion'],us['motion']);rand,rsteps=loop(randomp,q0);rev,vsteps=loop(ps['motion'][::-1],q0);app,asteps=loop(ps['appearance'],q0);uniform=np.where(q0>0,1.,0.);uniform/=uniform.sum();constant=np.broadcast_to(ps['motion'][0],ps['motion'].shape).copy();cq,csteps=loop(constant,q0);assert np.array_equal(cq,q0)
   sharp=q0*q0;sharp/=sharp.sum();qs=dict(NativeG6=q0,SemanticPrototypeLoop2=real,OneStepPrototype=steps[0]['q'],DirectSemanticEvidence=update(uniform,steps[1]['S']),ReversedSemanticLoop2=rev,RandomProjectionLoop2=rand,AppearancePrototypeLoop2=app,SharpenOriginal=sharp,NoProjection=q0.copy(),ConstantProjectionNoOp=cq);cost['all_prototype_arms_CPU']=time.perf_counter()-tt;iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a in ['NativeG6','NoProjection','ConstantProjectionNoOp']:assert np.array_equal(arrays[a],z['NativeG6'])and iv[a]==z['NativeG6_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=steps[1]['S'],clauses=dict(supported=True),semantic_p=ps,head_input=us,random_p=randomp,prototype_steps=dict(real=steps,random=rsteps,reversed=vsteps,appearance=asteps),cost=cost,GT_read=False,Full_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['SemanticPrototypeLoop2'],flush=True);del old,cap,geom,arrays,base,views;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['native_replay_queries']==64 and counts['fusion_offsets']==128 and counts['extra_dense_wordhead_forwards']==256 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
