"""Frozen native temporal event dependence: motion variation versus appearance."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_motion_hc64_v1';P=R/'results/native_event_motion_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';OLD=R/'outputs/event_query_hc64_v1'
ARMS=['NativeG6','SelectiveMotion','MotionContrast','AppearanceContrast','MotionNull','AppearanceNull','Sharpen','ReverseSelective']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(TEXT/'sealed.json')==p['text_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();paper=read(R/'outputs/tastvg_paper_v1/plan.json');text=read(TEXT/'sealed.json');old=read(OLD/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_text_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_motion_hc64_v1',authorization='Overnight user event-information research, original trained temporal capability, no extra experts, space fixed',primary='SelectiveMotion',arms=ARMS,count=64,code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],captures=old['captures'],prior_files=read(OLD/'sealed.json')['files'],text_files=text['files'],text_seal_sha256=sha(TEXT/'sealed.json'),labels_sha256=old['labels_sha256'],
  rationale='True text interventions fail matched event-pair ranking and alter identity proxy. Keep FULL text fixed, test independent native motion-vs-appearance temporal variation as event evidence.',
  fixed=dict(intervention='at original pre-fusion tensors, MotionNull replaces projected native3D-motion tensor over time by its temporal mean at each spatial/channel location; AppearanceNull similarly replaces native2D appearance; full text, pos/grid/mask unchanged',precision='mean computedFP32 cast back to originalFP16; no arbitrary gain or new parameter',replay='original FP16 crossmodal fusion then exact frozen FP32 two-pass native replay; no-op hidden/posterior bitwise recovery required',primary='q0*qAppearanceNull/qMotionNull; compare both individual q0²/qNull, directNull posteriors, sharpening and reversed-placement effect',space='same expert8 and complete-box geometry/DP; both real per-interval and fixed-space scoring'),
  new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,query_steps=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure='same repeatedly exposed HC64 development, no confirmation claim',
  references=['https://arxiv.org/abs/2207.13457','https://github.com/CFM-MSG/CAEM_Code'],scope='diagnostic frozen intervention, not trained D-TSG/CAEM reproduction. Native3D feature is not assumed pure motion, temporal mean is not a realistic static video.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','captures','prior_files','text_files']})
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
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};calls=dict(fusion_offsets=0,native_replay_queries=0,new_expert=0,new_visual_encoder=0,new_text_encoder=0,new_RGB=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['text_files'][k];assert sha(TEXT/e['trace_path'])==e['trace_sha256'];old=torch.load(TEXT/e['trace_path'],map_location='cpu',weights_only=False);cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];tick=time.perf_counter();outputs={};hidden={};cost={}
   for kind in ['Full','MotionNull','AppearanceNull']:
    views=[];t=time.perf_counter()
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if kind=='MotionNull':kw['vid']=kw['vid'].float().mean(0,keepdim=True).to(kw['vid'].dtype).expand_as(kw['vid']).contiguous()
     elif kind=='AppearanceNull':
      v=kw['videos'];x=v.tensors.float().mean(0,keepdim=True).to(v.tensors.dtype).expand_as(v.tensors).contiguous();kw['videos']=type(v)(x,v.mask,list(v.durations))
     prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));calls['fusion_offsets']+=1;del info,kw,prefixes
    with torch.no_grad():
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);hidden[kind]=[h.cpu()for h in base.temporal_inputs];calls['native_replay_queries']+=1;cost[kind]=time.perf_counter()-t
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6']),'Full pre-fusion replay must restore bitwise'
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Native full hidden must restore exactly'
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];take=q0>0
   def log(q):
    assert np.array_equal(take,q>0);a=np.zeros_like(q);a[take]=np.log(q[take]);return a
   l=log(q0);m=log(outputs['MotionNull']);a=log(outputs['AppearanceNull']);effect=a-m;reverse=effect[::-1,::-1].T.copy();qs=dict(NativeG6=q0,SelectiveMotion=normalize(q0,effect),MotionContrast=normalize(q0,l-m),AppearanceContrast=normalize(q0,l-a),MotionNull=outputs['MotionNull'],AppearanceNull=outputs['AppearanceNull'],Sharpen=normalize(q0,l),ReverseSelective=normalize(q0,reverse));iv={name:final_mbr(ids,q)['interval']for name,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   prior=p['prior_files'][k]
   with np.load(OLD/prior['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['NativeG6'],z['NativeG6']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=effect,cost=cost,GT_read=False,native_hidden_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,calls=calls,**who()));print(k,iv['NativeG6'],iv['SelectiveMotion'],round(time.perf_counter()-tick,3),flush=True);del old,cap,hidden,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=calls,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
