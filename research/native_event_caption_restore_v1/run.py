"""Retain positive target context before genuine native multimodal inference."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_caption_restore_hc64_v1';P=R/'results/native_event_caption_restore_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023';SUPPORT=R/'outputs/native_event_support_view_hc64_v1'
ARMS=['NativeG6','CaptionRestorationFusion','DirectCaptionView','DenseFillFusion','UniformRestorationFusion','SpatialPermutationFusion','TemporalRankFusion','NativeSelfMixture','NoView']
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
 p=dict(schema='native_event_caption_restore_hc64_v1',primary='CaptionRestorationFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],support_files=read(SUPPORT/'sealed.json')['files'],support_seal_sha256=sha(SUPPORT/'sealed.json'),design_sha256=sha(L/'next_caption_restore_design.md'),authorization='User bounded overnight event research using existing native frozen source and same8 observations; space fixed',
 rationale='Targetdrift ormissinginteractioncontext are concrete competing failures. Restore ORIGINALfeatures outsidefixed1.5trajectorycontext using actualtrainedSource COMPLETEcaption-to-gridattention; matched-strengthcontrols separate spatialplacement from merely lessperturbation.',fixed=dict(attention='Observe ORIGINALlastspatiallayer self_attn existing averagedheadweights without modifyingargs/kernel/need_weights/outputs. Textrows ORIGINALcaptionBPE only, excludeaddedsubjectprefix/specials, assertCPUtokenizerIDs matchcachedFull. Meantextrows andsum appearance/motionkeys samecell. Not eventprobability orcausality.',restoration='Attention dividedbyvalidperframe maximum. OriginalROI W1; outsideoriginalvalidweightsA/maxA. ZeroattentionfallbackW1 recorded. DenseFill=oldinside-meanfill ofoutside2D/3D; W*original+(1-W)*DenseFill FP32thenFP16; inside/paddingexact. FulltimeauxBoxFixed/all65/wholecaption/nativefusion/twooffset/sixlayers/hull/MBR/spaceDPunchanged. DENSEoriginalkeymask, originalSource safetycorner unchanged.',controls='DenseFillFusion exactlyaliasoldFillOnly; UniformusesEXACTsameoutsideperframemean, SpatialPermutationreversessameoutsideweightmultiset, TemporalRankreassignssamecurrentmultiset byreversedtimeattentionrank. AllsameOriginalROI/currentrestorationmass. Noquality/GT/confidencegate. DirectCaption sameinformation; originalself/source/noop.',scope='Prospectivefeaturecontextrestoration, notposthocpromotedFillOnly orattentiongain/layer/weightscan; native3D fullclipcontextremains. Match controls do not constitute known false events.'),new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='PrimarySource/G6actualt/fullvpositive,improved>harmed,Source shortANDlongtimepositive,captionrestoration beyond DenseFill/uniform/spatialpermutation/temporalrank;fixeddual-domainrecheck. No context/fusionweightscan orposthoc controlpromotion.',references=['https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/grounding_model/modal_encoder.py'])
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
 from research.native_event_caption_restore_v1.core import keep_region,inside_fill,caption_positions,observe,weights,restore
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(fusion_offsets=0,native_replay_queries=0,new_text_encoder=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'];keeps=[];supportstats=[]
   tokenpos,textlen=caption_positions(model.text_encoder.tokenizer,old['text'])
   for off,original in enumerate(old['prefusion']):
    assert textlen==original['texts'][0].shape[1];valid=~original['videos'].mask;mm=[]
    for j,v in enumerate(valid):
     m,fb=keep_region(aux[off+2*j],v);mm.append(m);supportstats.append(dict(offset=off,frame_index=off+2*j,valid_cells=int(v.sum()),keep_cells=int(m.sum()),empty_ROI_fallback=fb))
    keeps.append(torch.stack(mm))
   outputs={};cost={};attentions=[];restoration=[];restoration_stats=[]
   for kind in ['Full','DenseFill','Caption','Uniform','SpatialPermutation','TemporalRank']:
    t=time.perf_counter();views=[]
    for off,original in enumerate(old['prefusion']):
     kw=transfer(original,'cuda',clone=True);v=kw['videos'];valid=~v.mask;keep=transfer(keeps[off],'cuda');unchanged_mask=v.mask.clone()
     if kind!='Full':
      x=inside_fill(v.tensors,keep,valid);y=inside_fill(kw['vid'],keep,valid)
      if kind!='DenseFill':
       w=transfer(restoration[off][kind],'cuda');x=restore(v.tensors,x,w,keep,valid);y=restore(kw['vid'],y,w,keep,valid)
      kw['videos']=type(v)(x,v.mask.clone(),v.durations);kw['vid']=y
     prefixes=[];captured=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     from contextlib import nullcontext
     observer=observe(model.ground_encoder.encoder.spatial_layers[-1].self_attn,tokenpos,valid.shape[1]*valid.shape[2],textlen,captured)if kind=='Full'else nullcontext()
     try:
      with observer,torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1 and torch.equal(kw['videos'].mask,unchanged_mask),'Originaldensekeymask preserved'
     if kind=='Full':
      assert len(captured)==1 and not model.ground_encoder.encoder.spatial_layers[-1].self_attn._forward_hooks
      attention=captured[0].reshape(valid.shape).double();ws,stats=weights(attention,keeps[off],valid.cpu());attentions.append(attention);restoration.append(ws);restoration_stats.append(stats)
     views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Fullhiddenbitwise'
     for a,b in zip(logs,old['views']['Full']['logits']):assert torch.equal(a,b),'Fulllogitsbitwise'
    if kind=='DenseFill':
     pe17=p['support_files'][k];assert sha(SUPPORT/pe17['trace_path'])==pe17['trace_sha256'];parent17=torch.load(SUPPORT/pe17['trace_path'],weights_only=False,map_location='cpu');assert np.array_equal(outputs[kind],parent17['intervened_q']['SupportFillOnly']),'DenseFillparentbitwise'
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];qs=dict(NativeG6=q0,CaptionRestorationFusion=.5*q0+.5*outputs['Caption'],DirectCaptionView=outputs['Caption'],DenseFillFusion=.5*q0+.5*outputs['DenseFill'],UniformRestorationFusion=.5*q0+.5*outputs['Uniform'],SpatialPermutationFusion=.5*q0+.5*outputs['SpatialPermutation'],TemporalRankFusion=.5*q0+.5*outputs['TemporalRank'],NativeSelfMixture=.5*q0+.5*q0,NoView=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a in ['NativeG6','NativeSelfMixture','NoView']:assert np.array_equal(arrays[a],z['NativeG6'])and iv[a]==z['NativeG6_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   with np.load(SUPPORT/pe17['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['DenseFillFusion'],z['FillOnlyFusion'])and iv['DenseFillFusion']==z['FillOnlyFusion_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['CaptionRestorationFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=dict(supported=True),support_statistics=supportstats,caption_attention=attentions,restoration_weights=restoration,restoration_stats=restoration_stats,caption_token_count=len(tokenpos),cost=cost,GT_read=False,Full_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['CaptionRestorationFusion'],flush=True);del old,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['native_replay_queries']==384 and counts['fusion_offsets']==768 and all(not x.requires_grad and x.grad is None for x in model.parameters());assert not model.ground_encoder.encoder.spatial_layers[-1].self_attn._forward_hooks;put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
