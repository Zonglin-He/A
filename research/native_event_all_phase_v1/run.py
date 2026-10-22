"""Read real omitted middle predicates with the same frozen native temporal model."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_all_phase_hc64_v1';P=R/'results/native_event_all_phase_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';ORDER=R/'outputs/native_event_order_hc64_v1'
ARMS=['NativeG6','AllPhaseOrdered','DirectAllOrdered','EndpointOrdered','WrongMiddleOrdered','FullAsMiddleOrdered','SelfCountOrdered','AllEnvelope','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase'),(ORDER,'order')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');dc=read(PHASE/'decompositions.json');middle_members={k:len(x['clauses'])-2 for k,x in dc.items()if x['supported']and len(x['clauses'])>2};assert len(middle_members)==21 and sum(middle_members.values())==30;code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py','research/native_event_order_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_all_phase_hc64_v1',primary='AllPhaseOrdered',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),order_files=read(ORDER/'sealed.json')['files'],order_seal_sha256=sha(ORDER/'sealed.json'),decompositions_sha256=sha(PHASE/'decompositions.json'),labels_sha256=parent['labels_sha256'],supported=parent['supported'],new_middle_members=middle_members,new_middle_texts=30,
 authorization='User bounded overnight event-information research, existing frozen native backbone only; no added expert or observations, fixed space',rationale='Only first+last positive event texts were read previously.21 descriptions have30 omitted middle predicates, which may distinguish same-event endpoints from separate occurrences. This branch obtains actual NEW native event text responses, not a scalar normalization, query interface tweak or ordering strength scan. Existing two-event ordering negative and AND ambiguity preserved.',
 fixed=dict(grammar='Same52 deterministic main-AND descriptions.31two-event members MUST alias archived EndpointOrdered, not be excluded.12unsupported native.21members have3..5events and30new middle texts. No quality/GT gates.',native='Original Full subject prefix; existing cached projected visual values, true new middle RoBERTa+resizer/FP16fusion/original FP32 native two-pass/2offset/6normalized layer nativehull posterior. Reencoded Full text/hidden/posterior/Source boxes bitwise64. Reuse already sealed first/last; no expert/RGB/visual encoder/query edits.',primary='Exact3D DP over[first_start,current_start,current_end] assigns ALL K events with nondecreasing start AND end; overlap allowed. Output[first_start,last_end], normalize valid-order mass, fixed.5q0+.5joint, original physical MBR/complete-box DP. Order mass never used to reject.',controls='DirectAllOrdered unblended same all-phase joint information; Archived EndpointOrdered; WrongMiddleOrdered flips only real middle time placements, outer events untouched; FullAsMiddleOrdered repeats fullq0 for middle slots; SelfCountOrdered repeats q0 Ktimes; AllEnvelope independent minimal hull of ALL real events; NoComposition exact. Same52scope and.5fusion.',space='Same eight native ReferDINO observations/ERef/complete-box propagation/per-interval DP; actual and fixed-space metrics separately. All64 predictions seal before any new scoring'),
 weights_frozen=True,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Positive Source/NativeG6 time and completev, improvements>harms, positive Source short AND long time groups; benefit over EndpointOrdered/wrong-middle/full-as-middle/count controls, fixed dual-domain recheck before adoption. Report all64 and prespecified21new/31unchanged members.',references=parent['references'],scope='Own deterministic weak positive event-composition feasibility study, not TFVTG extra LLM/VLM or trained DeCo reconstruction reproduction.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','phase_files','order_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from functools import reduce
 from stvg.runtime import setup,lease,load_source
 from stvg.visual_cache import transfer
 from stvg.native import FrozenNativeReplay
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_composition_v1.core import envelope
 from research.native_event_all_phase_v1.core import all_ordered
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();assert sha(PHASE/'decompositions.json')==p['decompositions_sha256'];dc=read(PHASE/'decompositions.json');files={};counts=dict(text_forwards=0,new_middle_texts=0,fusion_offsets=0,native_replay_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k];oe=p['order_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe),(ORDER,oe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');phase=torch.load(PHASE/pe['trace_path'],weights_only=False,map_location='cpu');order=torch.load(ORDER/oe['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];subject=old['text']['subject'];clauses=dc[k];tick=time.perf_counter();outputs={};cost={};middle_boxes={};newinfo=bool(k in p['new_middle_members'])
   variants=[('Full',row['metadata']['caption'])]+([(f'Middle{j}',caption)for j,caption in enumerate(clauses['clauses'][1:-1],1)]if newinfo else [])
   for kind,caption in variants:
    t=time.perf_counter();views=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):text,_=model.text_encoder([subject+' '+caption],'cuda')
    counts['text_forwards']+=1;counts['new_middle_texts']+=int(kind!='Full')
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if kind=='Full':assert len(text)==len(kw['texts'])and all(torch.equal(x,y)for x,y in zip(text,kw['texts'])),'Full text tuple bitwise'
     kw['texts']=text;prefixes=[];h=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:h.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Full hidden bitwise'
    else:middle_boxes[kind]=base.zero['boxes'].cpu().numpy()
    del base,views,text;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];qs={a:q0.copy()for a in ARMS};masses={}
   if clauses['supported']:
    phases=[phase['intervened_q']['First']]+[outputs[f'Middle{j}']for j in range(1,len(clauses['clauses'])-1)]+[phase['intervened_q']['Last']];assert len(phases)==len(clauses['clauses']);joint,masses['real_all']=all_ordered(phases);wrong=[phases[0]]+[x[::-1,::-1].T.copy()for x in phases[1:-1]]+[phases[-1]];full=[phases[0]]+[q0 for _ in phases[1:-1]]+[phases[-1]];qw,masses['wrong_middle']=all_ordered(wrong);qf,masses['full_as_middle']=all_ordered(full);selfq,masses['self_count']=all_ordered([q0]*len(phases));allenv=reduce(envelope,phases);qs.update(AllPhaseOrdered=.5*q0+.5*joint,DirectAllOrdered=joint,EndpointOrdered=order['q']['OrderedClauseFusion'],WrongMiddleOrdered=.5*q0+.5*qw,FullAsMiddleOrdered=.5*q0+.5*qf,SelfCountOrdered=.5*q0+.5*selfq,AllEnvelope=.5*q0+.5*allenv)
    if not newinfo:
     for a in ['AllPhaseOrdered','WrongMiddleOrdered','FullAsMiddleOrdered']:assert np.array_equal(qs[a],order['q']['OrderedClauseFusion']),'two-event exact old operation'
   iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(ORDER/oe['path'],allow_pickle=False)as z:
    for a,alias in [('NativeG6','NativeG6'),('EndpointOrdered','OrderedClauseFusion'),('NoComposition','NoComposition')]:assert np.array_equal(arrays[a],z[alias])and iv[a]==z[alias+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['AllPhaseOrdered'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=clauses,order_mass=masses,new_middle_information=newinfo,middle_source_boxes=middle_boxes,cost=cost,GT_read=False,full_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=clauses['supported'],new_middle_information=newinfo,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,newinfo,iv['NativeG6'],iv['EndpointOrdered'],iv['AllPhaseOrdered'],flush=True);del old,phase,order,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['new_middle_texts']==30 and counts['text_forwards']==94 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
