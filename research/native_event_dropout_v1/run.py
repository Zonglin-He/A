"""Fixed native trained-dropout replay, no changes to spatial/evidence/model weights."""
import os,sys,time,json,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put
D=R/'outputs/native_event_dropout_v1';P=R/'results/native_event_dropout_v1';L=R/'outputs/overnight_event_loop_20261023';FIRST=R/'outputs/native_event_semantic_prototype_hc64_v1';RECHECK=R/'outputs/native_event_semantic_recheck_v1'
ARMS=['NativeG6','NativeDropoutFusion','DirectMCMean','FirstDrawFusion','MeanLogitsFusion','MomentMatchedGaussianFusion','WrongTimeDeltaFusion','NoDropoutMean','SharpenOriginal','ZeroDropoutNoOp']
CONTROLS=['MomentMatchedGaussianFusion','WrongTimeDeltaFusion','SharpenOriginal']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for name,root in [('first',FIRST),('recheck',RECHECK)]:assert sha(root/'sealed.json')==p[name+'_seal_sha256']and sha(root/'score_receipt.json')==p[name+'_score_receipt_sha256']
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(R/'outputs/native_event_three_state_v1/plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/event_query_v1/core.py','research/native_event_semantic_recheck_v1/score.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 assert set(CONTROLS)<=set(ARMS)
 p=dict(schema='native_event_dropout_v1',primary='NativeDropoutFusion',qualification_controls=CONTROLS,arms=ARMS,count=128,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],first_seal_sha256=sha(FIRST/'sealed.json'),first_score_receipt_sha256=sha(FIRST/'score_receipt.json'),recheck_seal_sha256=sha(RECHECK/'sealed.json'),recheck_score_receipt_sha256=sha(RECHECK/'score_receipt.json'),input_selection_plan_sha256=sha(R/'outputs/native_event_three_state_v1/plan.json'),items=parent['items'],labels=parent['labels'],development_labels_sha256=parent['development_labels_sha256'],seed=20260920,bootstrap_seed=20261022,stop_epoch=1792708800,maximum_GPU_seconds=1200,authorization='User bounded overnight event research until07; same completebox space/evidence8/fullcaption/frozen source',design_sha256=sha(L/'next_dropout_design.md'),official_audit_sha256=sha(L/'dropout_official_source_audit.json'),fixed=dict(draws=4,scope='ONLY nn.Dropout and nn.MultiheadAttention training flags INSIDE TimeDecoder; original probabilities, frozen model elsewhere eval, finally restore. No source weight adaptation.',seed='20260920 plus stable SHA of FULL input record, scoped fork_rng CPU/CUDA; no outcome/GT choice',posterior='EACH draw six layers/two-offset original hull/physical table; MC mean then .5native+.5MC, original MBR and fullbox DP',controls='same4draw meanlogits/firstdraw/direct, per-coordinate delta exact empiricalmean/std Gaussian fourdraw, halfcycle wrong-time delta, sharpen and native/zero-noop aliases',no_op='Native128 hidden/logits/q bitwise; scoped training with all dropout probabilities0 ALSO actual hidden/logits/q bitwise, restore probabilities/flags',limitations='Original source recipe containsdropout. This is conditional time-decoder stochastic stability, not checkpoint training history proof, full Bayesian calibration, new expert/semantic supervision or trained adaptation loop. MomentGaussian requires same actual MC statistics/budget. Wrong-time delta not known negative event.'),qualification='EACH HC64/HC32/Vid32 actualSource/G6 tANDfullvpositive, Source shortANDlongpositive, win>harm, beyond predeclared MomentGaussian/WrongTimeDelta/Sharpen. No p/K/fusion/seed sweep or posthoc controlpromotion.',new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_ASA_head=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all128_seal_before_GT=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','code','paper_code','provider_sources','assets']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal_hidden import verify_decoder
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.event_query_v1.core import restore_inputs
 from research.native_event_dropout_v1.core import stochastic_time,input_seed,gaussian_delta,wrong_delta
 p=verify();setup();torch.set_num_threads(1);assert os.environ.get('CUDA_VISIBLE_DEVICES')!='';lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU_native_dropout',fingerprint=p['fingerprint'],**who()));assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items());files={};momerr=0.;counts=dict(time_decoder_calls=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_ASA_head=0,new_expert=0,new_observations=0,new_optimizer=0)
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();decoder=model.ground_decoder.time_decoder;head=model.temp_embed;verify_decoder(decoder)
   for item in [x for x in p['items']if x['dataset']==ds]:
    verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];root=FIRST if item['source']=='first'else RECHECK;e=item['entry'];assert sha(root/e['path'])==e['sha256']and sha(root/e['trace_path'])==e['trace_sha256'];old=torch.load(root/e['trace_path'],weights_only=False,map_location='cpu');cf=PROVIDER/item['capture']['path'];assert sha(cf)==item['capture']['sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];offsets=cap['temporal_cache']['offsets'];kw=[restore_inputs(x,'cuda')for x in offsets];recs=cap['records'];tick=time.perf_counter();logs=[];hidden0=[]
    with torch.no_grad():
     for inp,saved in zip(kw,offsets):
      h=decoder(**inp);counts['time_decoder_calls']+=1;assert torch.equal(h.cpu(),saved['hidden']),'Nativehiddenbitwise';z=head(h).cpu();assert torch.equal(z[-1],saved['native_logits']);logs.append(z);hidden0.append(h.cpu())
     q0=posterior(recs,logs,ids);assert np.array_equal(q0,old['q']['NativeG6'])
     zero=[]
     with stochastic_time(decoder,zero=True):
      for i,inp in enumerate(kw):
       h=decoder(**inp);counts['time_decoder_calls']+=1;assert torch.equal(h.cpu(),hidden0[i]),'ZeroDropoutNativehiddenbitwise';z=head(h).cpu();assert torch.equal(z,logs[i]),'ZeroDropoutNativelogitsbitwise';zero.append(z)
     assert np.array_equal(posterior(recs,zero,ids),q0),'ZeroDropoutqbitwise'
     seed=input_seed(item,p['seed']);draws=[]
     with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
      torch.manual_seed(seed)
      with stochastic_time(decoder)as modules:
       for draw in range(4):
        zz=[]
        for inp in kw:zz.append(head(decoder(**inp)).cpu());counts['time_decoder_calls']+=1
        draws.append(zz)
     assert all(not m.training for m in decoder.modules())and not head.training
    qq=[posterior(recs,z,ids)for z in draws];qmc=np.mean(qq,0);meanlog=posterior(recs,[torch.stack([z[i]for z in draws]).mean(0)for i in range(2)],ids);deltas=[np.stack([(z[i]-logs[i]).numpy()for z in draws])for i in range(2)];gauss=[gaussian_delta(d,seed+i+411)for i,d in enumerate(deltas)];wrong=[wrong_delta(d)for d in deltas]
    for a,b in zip(gauss,deltas):momerr=max(momerr,float(abs(a.astype(float).mean(0)-b.astype(float).mean(0)).max()),float(abs(a.astype(float).std(0)-b.astype(float).std(0)).max()))
    qg=np.mean([posterior(recs,[logs[i]+torch.from_numpy(gauss[i][j])for i in range(2)],ids)for j in range(4)],0);qw=np.mean([posterior(recs,[logs[i]+torch.from_numpy(wrong[i][j])for i in range(2)],ids)for j in range(4)],0);sharp=q0*q0;sharp/=sharp.sum();qs=dict(NativeG6=q0,NativeDropoutFusion=.5*(q0+qmc),DirectMCMean=qmc,FirstDrawFusion=.5*(q0+qq[0]),MeanLogitsFusion=.5*(q0+meanlog),MomentMatchedGaussianFusion=.5*(q0+qg),WrongTimeDeltaFusion=.5*(q0+qw),NoDropoutMean=q0.copy(),SharpenOriginal=sharp,ZeroDropoutNoOp=q0.copy());assert set(qs)==set(ARMS)
    for q in qs.values():assert np.isfinite(q).all()and abs(q.sum()-1)<1e-12 and not np.tril(q).any()
    iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
    with np.load(root/e['path'],allow_pickle=False)as z:
     for a,b in [('NativeG6','NativeG6'),('NoDropoutMean','NativeG6'),('ZeroDropoutNoOp','NativeG6'),('SharpenOriginal','SharpenOriginal')]:assert np.array_equal(arrays[a],z[b])and iv[a]==z[b+'_interval'].tolist()
     arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
    k=item['alias'];f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,native_logits=logs,MC_logits=draws,seed=seed,modules=modules,moment_max_error=momerr,GT_read=False,native_bitwise=True,zero_dropout_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick);put(D/'progress.json',dict(completed=len(files),total=128,counts=counts,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['NativeDropoutFusion'],flush=True);del old,cap,kw,logs,hidden0,draws,geom,arrays;gc.collect();torch.cuda.empty_cache()
   assert all(not x.requires_grad and x.grad is None for x in model.parameters());del model;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['time_decoder_calls']==1536;assert momerr<2e-6;put(D/'sealed.json',dict(count=128,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start,moment_max_FP32_error=momerr,native_and_ZeroDropout_bitwise=128,weights_frozen=True));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),**who(),partial_preserved=True));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
