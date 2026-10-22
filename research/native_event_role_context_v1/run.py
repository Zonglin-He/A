"""Bounded genuine Source auxiliary argument grounding plus Full temporal context."""
import os,sys,time,json,hashlib,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_role_context_hc64_v1';P=R/'results/native_event_role_context_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','NativeRoleContextFusion','DirectPair','TargetOnly','ReflectedArgumentExtra','WrongTimeArgumentExtra','SubjectSourceContext','AuxiliaryObjectTimeFusion','Sharpen','FullKeysNoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(TEXT/'plan.json');paper=read(R/'outputs/tastvg_paper_v1/plan.json');print('parentkeys',list(parent));code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/event_query_v1/core.py','research/native_event_support_view_v1/core.py','research/native_event_spatial_text_v1/core.py','research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_role_context_hc64_v1',primary=ARMS[1],arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=read(TEXT/'sealed.json')['files'],text_seal_sha256=sha(TEXT/'sealed.json'),grammar_sha256=sha(L/'role_context_input_preflight.json'),design_sha256=sha(L/'next_role_context_design.md'),seed=20260920,labels_sha256=parent['labels_sha256'],stop_epoch=1792709970,maximum_GPU_seconds=600,
 authorization='User overnight research until07:00; unchanged complete-box spatial method and same8 expert observations. Auxiliary originalSource queries authorized; no added expert.',
 fixed=dict(argument='Fixed first main-event obj/iobj/obl noun phrase from original Stanza, no GT/quality choice. Prefix headlemma plus nounphrase. All59 supported;5 structurally unsupported all native aliases.',context='Original cached Full TimeDecoder kwargs/strides/query/full65grid/textkeys unchanged. Only motion key boolmask in actual text-motion order: union1.5 completeBoxFixedactor and1.5 auxiliarySourceargument. Original full fusion already contains global clip context. No pure causal interpretation.',readout='Original6 layers,2offset hull mean; .5qNative+.5qPair, original physical MBR+unchanged same geometryDP. One round.',controls='TargetOnly identicalmaskmechanism; reflectedargument/wrongtimeargument/Sourceactoraux EXACTsame EXTRAkeycount outsidefixedactor perframe, rowmajor deterministicfill; DirectPair/AuxObjectTime/Sharpen/FullKeysNoOp. Control not guaranteed negativeobject.'),
 new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0,maximum_new_Source_text=59,maximum_new_Source_fusion_offsets=246,GT_in_inference=False,qualification='PrimarySource/G6 actualt+fullvpositive; Source short<=5s andlong>5spositive,win>harm; exceedTargetOnly/Reflected/WrongTime/SubjectSource/sharpen then unchangedfixedHC32/Vid32. No controlpromotion/ROI/fusion sweep.',exposure='HC64 and recheckHC32/Vid32 exposed development only, not independent confirmation',references=['https://github.com/HengLan/TA-STVG','https://arxiv.org/abs/2207.13457'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files']})
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
 from research.event_query_v1.core import restore_inputs
 from research.native_event_role_context_v1.core import contexts,keymask
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,time_decoder_calls=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 grammar=read(L/'role_context_input_preflight.json');grows=grammar['rows'];grows={x['alias']:x for x in grows if x['group']=='HC64'}
 try:
  for row in records():
   assert time.time()<p['stop_epoch']and time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];assert sha(TEXT/te['trace_path'])==te['trace_sha256'];cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);actor=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'];supported=grows[k]['supported'];out={};native_logs=[];argboxes=None;stats=[];noop=[]
   def fusion_read(text=None):
    views=[]
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if text is not None:kw['texts']=text
     prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1
    with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    counts['native_replay_queries']+=1;return base,logs,posterior(cap['records'],logs,ids)
   full,flog,q0=fusion_read();assert np.array_equal(q0,old['intervened_q']['Full'])and np.array_equal(full.zero['boxes'].cpu().numpy(),cap['Source'])
   assert all(torch.equal(a.cpu(),b)for a,b in zip(full.temporal_inputs,old['views']['Full']['hidden']));assert all(torch.equal(a,b)for a,b in zip(flog,old['views']['Full']['logits']));del full
   for off in cap['temporal_cache']['offsets']:
    kw=restore_inputs(off,'cuda')
    with torch.no_grad():h=model.ground_decoder.time_decoder(**kw);z=model.temp_embed(h).cpu()
    counts['time_decoder_calls']+=1;assert torch.equal(h.cpu(),off['hidden'])and torch.equal(z[-1],off['native_logits']);native_logs.append(z);noop.append(True)
   assert np.array_equal(posterior(cap['records'],native_logs,ids),q0)
   if supported:
    a=grows[k]['argument']
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):text,_=model.text_encoder([a['subject']+' '+a['nounphrase']],'cuda')
    counts['text_forwards']+=1;arg,alogs,qa=fusion_read(text);argboxes=arg.zero['boxes'].cpu();del arg,text;masked={n:[]for n in ['Pair','TargetOnly','ReflectedArgumentExtra','WrongTimeArgumentExtra','SubjectSourceContext']}
    for offindex,off in enumerate(cap['temporal_cache']['offsets']):
     valid=~old['prefusion'][offindex]['videos'].mask;keys,ss=contexts(actor,argboxes,torch.as_tensor(cap['Source']),offindex,valid);stats.extend(ss);kw=restore_inputs(off,'cuda')
     for name,keep in keys.items():
      kk=dict(kw);kk['encoded_mask']=keymask(kw['encoded_mask'],keep)
      with torch.no_grad():masked[name].append(model.temp_embed(model.ground_decoder.time_decoder(**kk)).cpu())
      counts['time_decoder_calls']+=1
    mq={n:posterior(cap['records'],z,ids)for n,z in masked.items()};out['NativeRoleContextFusion']=(q0+mq['Pair'])/2;out['DirectPair']=mq['Pair']
    for n in ['TargetOnly','ReflectedArgumentExtra','WrongTimeArgumentExtra','SubjectSourceContext']:out[n]=(q0+mq[n])/2
    out['AuxiliaryObjectTimeFusion']=(q0+qa)/2;qs=q0*q0;out['Sharpen']=qs/qs.sum()
   else:out={a:q0.copy()for a in ARMS[1:]}
   out['NativeG6']=q0;out['FullKeysNoOp']=q0.copy();iv={a:final_mbr(ids,q)['interval']for a,q in out.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(TEXT/te['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['NativeG6'],z['NativeG6'])and iv['NativeG6']==z['NativeG6_interval'].tolist();arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=out,argument_source_boxes=argboxes,statistics=stats,native_bitwise=True,key_noop=noop,GT_read=False),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=supported);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv[ARMS[1]],flush=True);del old,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  assert counts['text_forwards']==59 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
