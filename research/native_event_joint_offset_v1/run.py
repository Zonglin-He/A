"""Frozen trained native event decoder can attend across cached offset streams."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_joint_offset_hc64_v1';P=R/'results/native_event_joint_offset_hc64_v1';PARENT=R/'outputs/native_event_projection_hc64_v1'
ARMS=['NativeG6','JointOffsetFusion','JointDirect','BlockedOffsetFusion','NativeMarginalFusion','SelfHullFusion','ReverseJointFusion','NoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(PARENT/'sealed.json')==p['parent_seal_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PARENT/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/event_query_v1/core.py','research/native_event_composition_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_joint_offset_hc64_v1',primary='JointOffsetFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],parent_files=read(PARENT/'sealed.json')['files'],parent_seal_sha256=sha(PARENT/'sealed.json'),labels_sha256=parent['labels_sha256'],supported=64,
 authorization='User overnight genuine event information research using frozen trained native temporal capacity/cache; no new expert or spatial change',rationale='Native temporal self-attention currently cannot exchange other-offset frame evidence; hull merges only after independent decoding. Test actual cross-offset temporal context rather than query changes or arbitrary ratio weights. Original cached motion already contains original clip context; no new independent visual evidence claimed.',
 difference_from_old='Earlier real prefix/suffix contexts remove video context or rebase time positions; old query fits alter state. This keeps ALL original Full-caption final-pass7 decoder input per-node values/time positions and physical nodes, interleaves both offsets for native event attention. New grid/legal support controlled by identical parity-blocked attention arm.',
 fixed=dict(seed=20260920,primary='FP32 original frozen six-layer time decoder, actual two offset kwargs interleaved into65 physical ordered nodes, original time slots unchanged; single legal posterior per layer mean6 then.5originalGlobal6+.5joint; original MBR/DP',block='same all65 input/output grid; temporary self_attn attn_mask blocks cross-parity edges in each6layers; finally cleanup, no source/state edits',controls='Source/NativeG6 exact; same-info JointDirect; same-grid BlockedFusion; independent single-offset marginal average fusion without hull; old SelfHullFusion; reversed per-frame encoded memory/pos/padding with original query/time retained; NoOp',no_op='original individual actual stored layout hidden/final logits/NativeG6 bitwise64; inverse interleave all7 kwargs exactly; blocked kernel size differs so numerical intervention error recorded not called bitwise equivalence',weights='all source/head parameters frozen eval, no optimizer; complete original Full text already in cache, no extra text/fusion',space='same eight observations, complete native box extents, unchanged ERef/geometry/per-interval DP; fixed-space results separate'),
 new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_expert=0,new_observations=0,new_optimizer=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Positive Source and NativeG6 actualtime/fullv, more improvements than harms, positive short AND long Source temporal gains; distinguish same-grid blocked control; fixed dual-domain recheck before main/full promotion. No mean/loss-only qualification.',references=['https://github.com/HengLan/TA-STVG','https://proceedings.iclr.cc/paper_files/paper/2025/file/9b5b12ea5acdf5d79544b412ac32cc51-Paper-Conference.pdf'])
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','parent_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal_hidden import verify_decoder
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.event_query_v1.core import restore_inputs
 from research.native_event_joint_offset_v1.core import interleave,parity_block,single_posterior,independent_marginal
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();decoder=model.ground_decoder.time_decoder;head=model.temp_embed;verify_decoder(decoder);files={};counts=dict(native_decoder_calls=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_expert=0,new_observations=0,new_optimizer=0);maxerr=0.
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['parent_files'][k];assert sha(PARENT/e['path'])==e['sha256'];cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];offsets=cap['temporal_cache']['offsets'];assert len(ids)==65 and all(x['frame_ids']==ids[o::2]for o,x in enumerate(offsets));kw=[restore_inputs(x,'cuda')for x in offsets];logs=[];tick=time.perf_counter()
   with torch.no_grad():
    for inp,saved in zip(kw,offsets):
     h=decoder(**inp);counts['native_decoder_calls']+=1;assert torch.equal(h.cpu(),saved['hidden']);z=head(h);assert torch.equal(z[-1].cpu(),saved['native_logits']);logs.append(z.cpu())
    q0=posterior(cap['records'],logs,ids);joint=interleave(kw);hjoint=decoder(**joint);zjoint=head(hjoint).cpu();counts['native_decoder_calls']+=1
    with parity_block(decoder,len(ids))as blocked_calls:hblock=decoder(**joint);zblock=head(hblock).cpu();counts['native_decoder_calls']+=1
    assert all(not l.self_attn._forward_pre_hooks for l in decoder.layers)
    # Same input values plus block isolation; kernel packing can change FP32
    # roundoff. Record actual discrepancy without waiving native bitwise checks.
    err=max(float(abs(hblock[:,0,o::2].cpu()-offsets[o]['hidden'][:,0]).max())for o in [0,1]);maxerr=max(maxerr,err)
    rev=dict(joint)
    for name,axis in [('encoded_feature',1),('encoded_pos',1),('encoded_mask',0)]:rev[name]=joint[name].flip(axis)
    zreverse=head(decoder(**rev)).cpu();counts['native_decoder_calls']+=1
   qjoint=single_posterior(zjoint);qblock=single_posterior(zblock);qmarg=independent_marginal(logs);qreverse=single_posterior(zreverse)
   with np.load(PARENT/e['path'],allow_pickle=False)as z:
    source=z['Source'].copy();source_iv=z['Source_interval'].tolist();oldnative=z['NativeG6'].copy();oldnative_iv=z['NativeG6_interval'].tolist();self_boxes=z['SelfHullFusion'].copy();self_iv=z['SelfHullFusion_interval'].tolist()
   # Self hull table is content-bound original parent trace, not reconstructed
   # from teacher scores or new labels.
   assert sha(PARENT/e['trace_path'])==e['trace_sha256'];trparent=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');assert np.array_equal(q0,trparent['q']['NativeG6'])
   qs=dict(NativeG6=q0,JointOffsetFusion=(q0+qjoint)/2,JointDirect=qjoint,BlockedOffsetFusion=(q0+qblock)/2,NativeMarginalFusion=(q0+qmarg)/2,SelfHullFusion=trparent['q']['SelfHullFusion'],ReverseJointFusion=(q0+qreverse)/2,NoOp=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS};assert np.array_equal(arrays['NativeG6'],oldnative)and iv['NativeG6']==oldnative_iv;assert np.array_equal(arrays['SelfHullFusion'],self_boxes)and iv['SelfHullFusion']==self_iv;assert np.array_equal(arrays['NoOp'],oldnative);arrays['Source']=source;iv['Source']=source_iv
   take=q0>0;score=np.zeros_like(q0);score[take]=np.log(np.maximum(qs['JointOffsetFusion'][take],1e-300))-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,score=score,clauses=dict(supported=True),blocked_hidden_error=err,blocked_calls=blocked_calls,joint_hidden=hjoint.cpu(),blocked_hidden=hblock.cpu(),joint_logits=zjoint,blocked_logits=zblock,GT_read=False,native_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=True);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,iv['NativeG6'],iv['JointOffsetFusion'],iv['BlockedOffsetFusion'],'blocked_error',err,flush=True);del cap,kw,logs,joint,rev,hjoint,hblock,geom,arrays,trparent;gc.collect();torch.cuda.empty_cache()
  verify();assert all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,network_weights_frozen=True,blocked_native_hidden_max_error=maxerr,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
