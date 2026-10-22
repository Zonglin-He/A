"""Actual positive event reads in a shared Source-defined temporal episode."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_episode_context_hc64_v1';P=R/'results/native_event_episode_context_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','EpisodeClauseFusion','GlobalClauseFusion','EpisodeFullFusion','EpisodeSelfHullFusion','ShiftedEpisodeClauseFusion','DirectEpisodeClauses','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert sha(L/'episode_context_source_only_preflight.json')==p['preflight_sha256'];assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');pre=read(L/'episode_context_source_only_preflight.json');assert pre['queries']==pre['proper_crops']==pre['distinct_shifted_windows']==52
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py','research/native_event_position_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_episode_context_hc64_v1',primary='EpisodeClauseFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),decompositions_sha256=sha(PHASE/'decompositions.json'),labels_sha256=parent['labels_sha256'],preflight_sha256=sha(L/'episode_context_source_only_preflight.json'),windows={x['query']:x for x in pre['rows']},supported=52,
 authorization='User authorizes bounded overnight native event-information feasibility research through07:00; original spatial/eight expert observations unchanged',rationale='Independent full-video positive clauses may refer to different occurrences. Change actual native event-reading context BEFORE producing clause posteriors, without monotonic AND assumptions, query injection, ratios or rejection.',
 fixed=dict(window='NativeG6 MBR index boundaries i,j; w=max(49,j-i+1), lo=clip(floor((i+j-w+1)/2),0,65-w). Both native endpoints included. Shifted control same width at farthest edge. Source-only preflight52proper52distinct, noGT.',native='Actual Full global text/no-op64. Same52 grammar-supported queries actual EpisodeFull/First/Last and ShiftedFirst/Last. Original FP16 pre-fusion values selected, original global-offset time positions retained with scoped hook; reorder local parity for odd window starts while preserving physical IDs/values. FP32 original two-pass decoder/six normalized heads/native offset hull.',primary='.5qNative+.5Envelope(EpisodeFirst,EpisodeLast), embedded zeros outside episode on original physical grid, original MBR and fixed complete-box DP.',controls='Archived GlobalClauseFusion exact; EpisodeFullFusion; EpisodeSelfHullFusion; same-width ShiftedEpisodeClauseFusion; DirectEpisodeClauses unblended same information; Source/Native/NoComposition. All52 receive all reads;12unsupported native, no confidence or GT gates.',space='Original geometry/DP/evidence8 unchanged. Local Source boxes diagnostic only. Cached3D still carries original clip context; not raw-video causal cropping.'),weights_frozen=True,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Primary Source and G6 time/fullv positive; improvements>harms; Source short<=5s AND long>5s positive; temporal benefit over GlobalClause/context-only/shifted controls; fixed HC/Vid recheck before promotion. No weight/window scan or posthoc control promotion.',references=parent['references'],scope='Own frozen-native actual-context feasibility; not reproduction of trained temporal debiasing or TFVTG additional VLM/LLM.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','phase_files','windows']})
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
 from research.native_event_position_v1.core import original_positions
 from research.native_event_episode_context_v1.core import window,slices,embed
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();assert sha(PHASE/'decompositions.json')==p['decompositions_sha256'];dc=read(PHASE/'decompositions.json');files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');phase=torch.load(PHASE/pe['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];clauses=dc[k];tick=time.perf_counter();outputs={};cost={};hooks={};source_boxes={};native=final_mbr(ids,old['q']['NativeG6']);ii,jj=native['indices'] if 'indices'in native else (None,None)
   if ii is None:ii=ids.index(native['interval'][0]);jj=ids.index(native['interval'][1]-1)
   lo,hi,shift=window(ii,jj,len(ids));width=hi-lo+1
   if clauses['supported']:
    pre=p['windows'][k];assert (lo,hi,shift,width)==(pre['lo'],pre['hi'],pre['shifted_lo'],pre['width'])
   texts={}
   for kind,caption in [('Full',row['metadata']['caption'])]+([('First',clauses['first']),('Last',clauses['last'])]if clauses['supported']else[]):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):texts[kind],_=model.text_encoder([old['text']['subject']+' '+caption],'cuda')
    counts['text_forwards']+=1
   jobs=[('Full','Full',0,64)]+([('EpisodeFull','Full',lo,hi),('EpisodeFirst','First',lo,hi),('EpisodeLast','Last',lo,hi),('ShiftedFirst','First',shift,shift+width-1),('ShiftedLast','Last',shift,shift+width-1)]if clauses['supported']else[])
   for kind,tk,left,right in jobs:
    t=time.perf_counter();views=[];recs=[];selected=[];lengths=[]
    for localoff,(off,pos,sel)in enumerate(slices(left,right,len(ids))):
     original=old['prefusion'][off];kw=transfer(original,'cuda',clone=True)
     if kind=='Full':assert all(torch.equal(a,b)for a,b in zip(texts[tk],kw['texts'])),'Full text tuple bitwise'
     kw['texts']=texts[tk]
     if kind!='Full':
      v=kw['videos'];ix=torch.tensor(sel,device=v.tensors.device);kw['videos']=type(v)(v.tensors.index_select(0,ix),v.mask.index_select(0,ix),[len(sel)]);kw['vid']=kw['vid'].index_select(0,ix);kw['vis_pos']=kw['vis_pos'].index_select(0,ix)
      assert torch.equal(kw['videos'].tensors,original['videos'].tensors.to('cuda').index_select(0,ix))and torch.equal(kw['vid'],original['vid'].to('cuda').index_select(0,ix))and torch.equal(kw['vis_pos'],original['vis_pos'].to('cuda').index_select(0,ix))
     selected.append(sel);lengths.append(len(range(off,len(ids),2)));recs.append(dict(offset=localoff,original_offset=off,flip=False,frame_ids=[ids[i]for i in pos]));prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad(),original_positions(model.ground_decoder.time_embed,lengths,selected)as hc:
     base=FrozenNativeReplay(model,transfer(views,'cuda'),right-left+1);logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    assert not model.ground_decoder.time_embed._forward_hooks;hooks[kind]=hc;local=posterior(recs,logs,ids[left:right+1]);outputs[kind]=embed(local,left,right,len(ids));source_boxes[kind]=base.zero['boxes'].cpu().numpy();cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6'])and np.array_equal(source_boxes[kind],cap['Source'])
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Full hidden bitwise'
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];qs={a:q0.copy()for a in ARMS}
   if clauses['supported']:
    localhull=envelope(outputs['EpisodeFirst'],outputs['EpisodeLast']);qs.update(EpisodeClauseFusion=.5*q0+.5*localhull,GlobalClauseFusion=phase['q']['ClauseFusion'],EpisodeFullFusion=.5*q0+.5*outputs['EpisodeFull'],EpisodeSelfHullFusion=.5*q0+.5*envelope(outputs['EpisodeFull'],outputs['EpisodeFull']),ShiftedEpisodeClauseFusion=.5*q0+.5*envelope(outputs['ShiftedFirst'],outputs['ShiftedLast']),DirectEpisodeClauses=localhull)
   iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a,alias in [('NativeG6','NativeG6'),('GlobalClauseFusion','ClauseFusion'),('NoComposition','NoComposition')]:assert np.array_equal(arrays[a],z[alias])and iv[a]==z[alias+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['EpisodeClauseFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=clauses,window=dict(lo=lo,hi=hi,shifted_lo=shift,width=width),position_calls=hooks,source_boxes=source_boxes,cost=cost,GT_read=False,Full_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=clauses['supported'],cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,lo,hi,iv['NativeG6'],iv['EpisodeClauseFusion'],flush=True);del old,phase,cap,geom,arrays,texts;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['text_forwards']==168 and counts['native_replay_queries']==324 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
