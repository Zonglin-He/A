"""Keep full-grid native positions during real overlapping context inference."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_position_hc64_v1';P=R/'results/native_event_position_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';CONTEXT=R/'outputs/native_event_context_hc64_v1';PRIOR=R/'outputs/native_event_innovation_hc64_v1'
ARMS=['NativeG6','GlobalPositionInnovation','ContextInnovation','CensoredContext','Sharpen','ReverseGlobalPosition','NoInnovation']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(CONTEXT,'context'),(PRIOR,'prior')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PRIOR/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_text_v1/core.py','research/native_event_context_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_position_hc64_v1',primary='GlobalPositionInnovation',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],captures=parent['captures'],text_files=parent['text_files'],context_files=parent['context_files'],prior_files=read(PRIOR/'sealed.json')['files'],text_seal_sha256=sha(TEXT/'sealed.json'),context_seal_sha256=sha(CONTEXT/'sealed.json'),prior_seal_sha256=sha(PRIOR/'sealed.json'),labels_sha256=parent['labels_sha256'],
 authorization='User authorized overnight event-information research, same frozen backbone/evidence and space; no additional expert',rationale='Real contexts change both content and temporal position numbering. Suffix native slots8..32/31 become0..24/23; preserve original full-grid embeddings to isolate content effect. ContextInnovation t−.1596pp with short-event harm is not adopted.',
 fixed=dict(windows={'Full':[0,64],'Prefix':[0,48],'Suffix':[16,64]},position='scoped hook on ground_decoder.time_embed, original full_offset_length output sliced at actual original offset indices; shared by original spatial/time branches in both native passes. No weight/state_dict edit.',no_op='Full hidden/posterior bitwise64 and Prefix posterior bitwise64 against previous context. Full/Prefix positional values equal native. All four hook calls per replay validated; hook always removed.',primary='same censored joint/one-boundary evidence log(qGlobalPositionWindow/bSameFullLogitsWindow), averaged two contexts, q0*exp(effect), original physical MBR and complete-box DP',controls='previous rebased ContextInnovation, CensoredContext, Sharpen, reversed global-position evidence, exact no-innovation',space='same original expert8/geometry/DP; Source/NativeG6 outputs exact aliases'),
 new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification=parent['qualification'],references=['https://arxiv.org/abs/2111.04321','https://aclanthology.org/2022.findings-emnlp.326/'],scope='Frozen positional-context feasibility test, not a trained temporal debiasing reproduction; projected3D features remain from original clips, no raw-video temporal recrop.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','captures','text_files','context_files','prior_files']})
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
 from research.native_event_text_v1.core import normalize
 from research.native_event_context_v1.core import censored_evidence
 from research.native_event_position_v1.core import original_positions
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};calls=dict(fusion_offsets=0,native_replay_queries=0,new_expert=0,new_visual_encoder=0,new_text_encoder=0,new_RGB=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['text_files'][k];assert sha(TEXT/e['trace_path'])==e['trace_sha256'];old=torch.load(TEXT/e['trace_path'],map_location='cpu',weights_only=False);cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];ce=p['context_files'][k];assert sha(CONTEXT/ce['trace_path'])==ce['trace_sha256'];ct=torch.load(CONTEXT/ce['trace_path'],weights_only=False,map_location='cpu');pe=p['prior_files'][k];assert sha(PRIOR/pe['path'])==pe['sha256']and sha(PRIOR/pe['trace_path'])==pe['trace_sha256'];pt=torch.load(PRIOR/pe['trace_path'],weights_only=False,map_location='cpu');tick=time.perf_counter();outputs={};cost={};scores={};position_calls={};max_posterior_difference={}
   for kind,(lo,hi)in [('Full',(0,64)),('Prefix',(0,48)),('Suffix',(16,64))]:
    wids=ids[lo:hi+1];views=[];recs=[];selected_offsets=[];t=time.perf_counter()
    for offset,original in enumerate(old['prefusion']):
     positions=[i for i in range(offset,len(ids),2)if lo<=i<=hi];sel=[(i-offset)//2 for i in positions];selected_offsets.append(sel);assert [ids[i]for i in range(offset,65,2)]==old['views']['Full']['records'][offset]['frame_ids'];kw=transfer(original,'cuda',clone=True)
     if kind!='Full':
      v=kw['videos'];ix=torch.tensor(sel,device=v.tensors.device);kw['videos']=type(v)(v.tensors.index_select(0,ix),v.mask.index_select(0,ix),[len(sel)]);kw['vid']=kw['vid'].index_select(0,ix);kw['vis_pos']=kw['vis_pos'].index_select(0,ix)
     recs.append(dict(offset=offset,flip=False,frame_ids=[ids[i]for i in positions]));prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));calls['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad(),original_positions(model.ground_decoder.time_embed,[33,32],selected_offsets)as hook_calls:
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(wids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    assert not model.ground_decoder.time_embed._forward_hooks;position_calls[kind]=hook_calls;outputs[kind]=posterior(recs,logs,wids);calls['native_replay_queries']+=1;cost[kind]=time.perf_counter()-t;max_posterior_difference[kind]=float(abs(outputs[kind]-ct['intervened_q'][kind]).max())
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6']),'Full posterior bitwise'
     for x,y in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(x.cpu(),y),'Full hidden bitwise'
    if kind in ['Full','Prefix']:assert all(x['same_native_values']for x in hook_calls)and np.array_equal(outputs[kind],ct['intervened_q'][kind]),'Full/Prefix unchanged positions no-op'
    if kind!='Full':scores[kind],_=censored_evidence(ids,wids,outputs[kind],pt['bases'][kind])
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];effect=(scores['Prefix']+scores['Suffix'])/2;qs=dict(NativeG6=q0,GlobalPositionInnovation=normalize(q0,effect),ContextInnovation=pt['q']['ContextInnovation'],CensoredContext=pt['q']['CensoredContext'],Sharpen=pt['q']['Sharpen'],ReverseGlobalPosition=normalize(q0,effect[::-1,::-1].T.copy()),NoInnovation=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PRIOR/pe['path'],allow_pickle=False)as z:
    for a in ['NativeG6','ContextInnovation','CensoredContext','Sharpen','NoInnovation']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);out=D/'traces'/f'{k}.pt';out.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=effect,context_scores=scores,position_calls=position_calls,posterior_difference=max_posterior_difference,cost=cost,GT_read=False,full_prefix_bitwise=True),out);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(out.relative_to(D)),trace_sha256=sha(out),seconds=time.perf_counter()-tick,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,calls=calls,**who()));print(k,iv['NativeG6'],iv['GlobalPositionInnovation'],max_posterior_difference['Suffix'],flush=True);del old,cap,ct,pt,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=calls,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
