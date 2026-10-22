"""Retain positive target context before genuine native multimodal inference."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_modality_context_hc64_v1';P=R/'results/native_event_modality_context_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023';SUPPORT=R/'outputs/native_event_support_view_hc64_v1'
ARMS=['NativeG6','MotionFocusFusion','DirectMotionTarget','MotionReflectedFusion','AppearanceFocusFusion','AppearanceReflectedFusion','WrongTimeMotionFusion','BothFocusFusion','NativeSelfMixture','NoView']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert sha(SUPPORT/'sealed.json')==p['support_seal_sha256'];assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_spatial_text_v1/core.py','research/native_event_support_view_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_modality_context_hc64_v1',primary='MotionFocusFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],support_files=read(SUPPORT/'sealed.json')['files'],support_seal_sha256=sha(SUPPORT/'sealed.json'),design_sha256=sha(L/'next_modality_context_design.md'),authorization='User bounded overnight event research using existing native frozen source and same8 observations; space fixed',
 rationale='Both-modality fill loses globalactors/relationobjects andmotion simultaneously. Preserve ORIGINALglobal2Dappearance while focusing ONLYoriginal3Dmotion onfixedtargetcontext; distinguish role/location/time controls before claim.',fixed=dict(support='Same fulltimecompleteBoxFixedauxtrajectory1.5extent context, anycontinuousoverlap validgrid, emptyROIframe fallbackwholevalid recorded/noqueryveto. Samearea180reflectedcontext withinvalidrectangle.',features='MotionTarget retains original2Dappearance everywhere andoriginal3DinsideROI/padding; outsidevalid3D fillperframeinside3Dmean FP32castFP16. AppearanceTarget is converse. Originaldenseattentionkeys/padding/native safetycorner unchanged. WrongTimeMotion reverses original3Dframeorder WITHINoffset then currentROI/fill; original3Dclips notinternallyreversed, not faithful backwardvideo/knownfalseevent.',native='Originalcompletecaption/all65physicalnodes/timepositions/FP16fusion/FP32native2pass/6head/twooffsethull/physicalMBR/finalDP/evidence8. Full64hidden/logits/q/Sourceboxesbitwise.',controls='MotionReflected/AppearanceTarget/AppearanceReflected/WrongTimeMotion all actualsame-budgetnative reads; DirectMotion sameinformation; BothFocus exactold17FillOnlyposterior/interval/finalboxes aliaswithoutnewmodelcalls; originalsource/native/self/noop. All64active,noGT/confidencegating.',scope='OriginalSource architecture has separate2D/3Dfeatures but jointtrainedfusion; this is prospectivefrozenmodalitycontext test, notproven pureeventsignal. Native3D includesfullclipcontext. Notattention/featureblend/context/fusionweightscan.'),new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='PrimarySource/G6actualt/fullvpositive,improved>harmed,Source shortANDlongtimepositive,targetmotion beyond reflectedmotion/wrongtime/appearance/BothFocus;fixeddual-domainrecheck. No context/fusionweightscan orposthoc controlpromotion.',references=['https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/modal_encoder.py'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','phase_files','support_files']})
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
 from research.native_event_modality_context_v1.core import keep_region,reflected,focus
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(fusion_offsets=0,native_replay_queries=0,new_text_encoder=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'];keeps=[];controls=[];supportstats=[]
   for off,original in enumerate(old['prefusion']):
    valid=~original['videos'].mask;mm=[];cc=[]
    for j,v in enumerate(valid):
     m,fb=keep_region(aux[off+2*j],v);c=reflected(m,v);mm.append(m);cc.append(c);supportstats.append(dict(offset=off,frame_index=off+2*j,valid_cells=int(v.sum()),keep_cells=int(m.sum()),control_cells=int(c.sum()),overlap_fraction=float((m&c).sum()/m.sum()),empty_ROI_fallback=fb,target_corner_kept=bool(m[0,0]),control_corner_kept=bool(c[0,0])))
    keeps.append(torch.stack(mm));controls.append(torch.stack(cc))
   pe17=p['support_files'][k];assert sha(SUPPORT/pe17['trace_path'])==pe17['trace_sha256']and sha(SUPPORT/pe17['path'])==pe17['sha256'];old17=torch.load(SUPPORT/pe17['trace_path'],weights_only=False,map_location='cpu');outputs={};cost={}
   for kind in ['Full','MotionTarget','MotionReflected','AppearanceTarget','AppearanceReflected','WrongTimeMotion']:
    t=time.perf_counter();views=[]
    for off,original in enumerate(old['prefusion']):
     kw=transfer(original,'cuda',clone=True);v=kw['videos'];valid=~v.mask;keep=transfer((controls if kind.endswith('Reflected')else keeps)[off],'cuda');originalmask=v.mask.clone();x,y=focus(v.tensors,kw['vid'],keep,valid,kind);kw['videos']=type(v)(x,v.mask.clone(),v.durations);kw['vid']=y
     prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1 and torch.equal(kw['videos'].mask,originalmask),'All original dense keys preserved'
     views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Fullhiddenbitwise'
     for a,b in zip(logs,old['views']['Full']['logits']):assert torch.equal(a,b),'Fulllogitsbitwise'
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];qs=dict(NativeG6=q0,MotionFocusFusion=.5*q0+.5*outputs['MotionTarget'],DirectMotionTarget=outputs['MotionTarget'],MotionReflectedFusion=.5*q0+.5*outputs['MotionReflected'],AppearanceFocusFusion=.5*q0+.5*outputs['AppearanceTarget'],AppearanceReflectedFusion=.5*q0+.5*outputs['AppearanceReflected'],WrongTimeMotionFusion=.5*q0+.5*outputs['WrongTimeMotion'],BothFocusFusion=old17['q']['FillOnlyFusion'].copy(),NativeSelfMixture=.5*q0+.5*q0,NoView=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a in ['NativeG6','NativeSelfMixture','NoView']:assert np.array_equal(arrays[a],z['NativeG6'])and iv[a]==z['NativeG6_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   with np.load(SUPPORT/pe17['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['BothFocusFusion'],z['FillOnlyFusion'])and iv['BothFocusFusion']==z['FillOnlyFusion_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['MotionFocusFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=dict(supported=True),support_statistics=supportstats,modality_preservation_checked=True,cost=cost,GT_read=False,Full_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['MotionFocusFusion'],flush=True);del old,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['native_replay_queries']==384 and counts['fusion_offsets']==768 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
