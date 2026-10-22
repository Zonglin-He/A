"""Actual native temporal response to crossed target-local and event-text changes."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_spatial_text_hc64_v1';P=R/'results/native_event_spatial_text_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','InteractionResidual','DirectInteraction','TargetInteraction','SpatialOnly','IdentityOnly','TextOnly','Sharpen','ReverseInteraction','NoInteraction']
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
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_text_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_spatial_text_hc64_v1',primary='InteractionResidual',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],design_sha256=sha(L/'next_spatial_text_design.md'),
 authorization='User bounded overnight event-information research through07:00; no extra experts/evidence and same complete spatial method',rationale='Text-only and global temporal-motion nulls fail; native two-pass states nearly agree. Ask if propagated-target-local visual evidence influences trained interval response specifically for event text versus identity text, corrected by equal-area control intervention. New actual crossed spatial+text forward, not ReferDINO normalization, attention prior, old pre-MLP TTS foreground erasure or query-step scan.',
 fixed=dict(auxiliary='Original complete-box geometry/DP once on entire physical support defines auxiliary target location, independent of predicted event interval/GT. Actual final output still original same geometry/DP on temporal decision.',intervention='Fractional continuous complete-box cellcoverage on original prefusion7x10 grid times valid cells. BOTH2D/3D features replace covered fractions by per-frame valid outside-mask mean computedFP32 then FP16. All-covered fallback to full valid mean recorded, never query rejection. Control180reflects same softmask within valid rectangle exactly preserving area; not guaranteed wrong location. Padding unchanged.',text='Saved original Subject+Full vs Subject+existing Identity from previous text intervention. Full0/Id0 reproduce64 text hidden/q/Source boxes; actual FullTarget/IdTarget/FullControl/IdControl use same footprint/feature operation. Original native FP16fusion/FP32two-pass/sixhead/twooffsethull.',primary='LT=logqFull0-logqFullTarget-logqId0+logqIdTarget, LC analogousControl. Normalize q0*exp(LT-LC). Full0/Id0 cancel in effect but are required no-op and singlefactor controls. Coefficients1, no floor/eps/gain/threshold.',controls='DirectInteraction normalizedexp(LT-LC) withoutnativeprior; TargetInteraction LTonly; SpatialOnly logqFullControl-logqFullTarget; IdentityOnly logqIdControl-logqIdTarget; archived TextOnly q0²/qId; Sharpen q0²; temporally reflectedinteraction; Source/Native/nointeraction. All64, no quality gates.',scope='Feature interventions not photorealistic removal; native3D retains originalclipcontext; identity caption can be incomplete and control may overlap target. Not causal event probability or trained D-TSG/CAEM reproduction.'),new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,new_identity_text_forwards=64,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Primary Source/G6 actualt/fullvpositive, improved>harmed, bothSource short/longtimepositive; exceed spatial/identity/text/sharpen/time-mismatchcontrols thenfixeddual-domainrecheck. No spatialmask/gain/ratio sweep or posthoc promotion ofcontrols.',references=['https://arxiv.org/abs/2207.13457','https://github.com/CFM-MSG/CAEM_Code'])
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
 from research.native_event_spatial_text_v1.core import footprint,reflected,replace,posteriors
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');phase=torch.load(PHASE/pe['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'];masks=[];control=[];maskstats=[]
   for off,original in enumerate(old['prefusion']):
    valid=~original['videos'].mask;mm=[];cc=[]
    for j,v in enumerate(valid):
     m=footprint(aux[off+2*j],v.shape[0],v.shape[1],v);c=reflected(m,v);mm.append(m);cc.append(c);mass=float(m.sum());overlap=float(torch.minimum(m,c).sum())/mass if mass else None;maskstats.append(dict(mask_cell_mass=mass,control_cell_mass=float(c.sum()),target_control_overlap=overlap,fully_covered=bool(mass>=float(v.sum())-1e-10)))
    masks.append(torch.stack(mm));control.append(torch.stack(cc))
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):identity_text,_=model.text_encoder([old['text']['subject']+' '+old['text']['identity_caption']],'cuda')
   counts['text_forwards']+=1;outputs={};cost={};fallbacks={};zero_flags={}
   for kind in ['Full0','Id0','FullTarget','IdTarget','FullControl','IdControl']:
    t=time.perf_counter();views=[];fallbacks[kind]=0
    for off,original in enumerate(old['prefusion']):
     kw=transfer(original,'cuda',clone=True)
     if kind.startswith('Id'):kw['texts']=identity_text
     if kind.endswith('Target')or kind.endswith('Control'):
      m=transfer((masks if kind.endswith('Target')else control)[off],'cuda');v=kw['videos'];valid=~v.mask;x,n1=replace(v.tensors,m,valid);y,n2=replace(kw['vid'],m,valid);fallbacks[kind]+=n1+n2;kw['videos']=type(v)(x,v.mask,v.durations);kw['vid']=y
     prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind in ['Full0','Id0']:
     alias='Full'if kind=='Full0'else'Identity';assert np.array_equal(outputs[kind],old['intervened_q'][alias])and np.array_equal(base.zero['boxes'].cpu().numpy(),old['intervention_source_boxes'][alias]),'Full/Identity native no-op'
     for a,b in zip(base.temporal_inputs,old['views'][alias]['hidden']):assert torch.equal(a.cpu(),b),'Full/Identity hidden bitwise'
     for a,b in zip(logs,old['views'][alias]['logits']):assert torch.equal(a,b),'Full/Identity all-layer logits bitwise'
     zero_flags[kind]=True
    del base,views;gc.collect();torch.cuda.empty_cache()
   qs,score=posteriors(outputs);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(TEXT/te['path'],allow_pickle=False)as z:
    for a,alias in [('NativeG6','NativeG6'),('TextOnly','IdentityContrast'),('Sharpen','Sharpen'),('NoInteraction','NativeG6')]:assert np.array_equal(arrays[a],z[alias])and iv[a]==z[alias+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=dict(supported=True),mask_statistics=maskstats,background_fallbacks=fallbacks,zero_flags=zero_flags,cost=cost,GT_read=False),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['InteractionResidual'],flush=True);del old,phase,cap,geom,arrays,identity_text;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['text_forwards']==64 and counts['native_replay_queries']==384 and counts['fusion_offsets']==768 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
