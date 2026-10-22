"""Fixed HC32/Vid32 recheck of stage21, no algorithm or parameter changes."""
import sys,os,time,json,hashlib,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put,records,B
from research.native_event_semantic_prototype_v1.core import loop,matched_random,update
D=R/'outputs/native_event_semantic_recheck_v1';P=R/'results/native_event_semantic_recheck_v1';PARENT=R/'outputs/native_event_semantic_prototype_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','SemanticPrototypeLoop2','OneStepPrototype','DirectSemanticEvidence','ReversedSemanticLoop2','RandomProjectionLoop2','SharpenOriginal','NoProjection','ConstantProjectionNoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert sha(PARENT/'sealed.json')==p['parent_seal_sha256'];assert sha(PARENT/'score_receipt.json')==p['parent_score_receipt_sha256'];assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();old=read(PARENT/'plan.json');selection=PROVIDER/'outputs/single448_recheck_v1/plan.json';panel=read(selection);items=[]
 for ds in ['hc2','vidstg']:
  for i,item in enumerate(panel['items'][ds]):
   k=item['query_id'];base=PROVIDER/f'outputs/interval_pool/recheck_{ds}';fs={name:base/sub/f'{k}.{ext}'for name,sub,ext in [('capture','capture','pt'),('posterior','posteriors','pt'),('prediction','predictions','npz')]};assert all(f.exists()for f in fs.values());items.append(dict(dataset=ds,key=k,alias=f'{ds.upper()}_RECHECK{i+1:03d}',row=item['record'],parent=item['parent'],files={name:dict(path=str(f.relative_to(PROVIDER)),sha256=sha(f))for name,f in fs.items()}))
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/native_event_semantic_prototype_v1/core.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_semantic_recheck_v1',primary='SemanticPrototypeLoop2',arms=ARMS,count=64,items=items,selection_sha256=sha(selection),selection='EXACTpreviousfixedHC32/Vid32 exposedparentpanels; no newindependenttest claim',parent_plan_sha256=sha(PARENT/'plan.json'),parent_seal_sha256=sha(PARENT/'sealed.json'),parent_score_receipt_sha256=sha(PARENT/'score_receipt.json'),code=code,paper_code=old['paper_code'],provider_sources=old['provider_sources'],assets=old['assets'],labels=panel['labels'],seed=20260920,bootstrap_seed=20261022,fixed='EXACTstage21core2roundoppositeparityclassSUM/inclusiveframeSUM/fixedq0prior, no gain/iteration/classcountnormalization. VidSource head34, HCSource head60, dimensionchangeexplicit nottuned.',bridge='Cached actualTimeDecoder encoded_feature is originalFullnormalizedH[text+motion]. Reconstruct originalfm andft; learnedhead input/output mustbitwise stage21 all64 before HC32/Vid32. Originaltimeheadcachedhidden posterior no-op. This avoidsallnewfusion/visual/text anddoesnotmodifyforward. Appearancefeature unavailable incompacttimecache: rolecontrol alreadylocked/measuredHC64, no pretendingrecheckappearance.',exposure=old['exposure'],authorization='User bounded overnight: afterfixedHC64qualification immediateSAMEalgorithm dualdomainrecheck, spacefixed noexpertadded',new_RGB=0,new_visual_encoder=0,new_fusion=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,qualification='EACHdomainprimarySource/G6actualt/fullvpositive,Source shortANDlongpositive,win>harm,beyondrandom/reversed/sharpen; preserveallharms/CI. No parameterchange orfallbackcontrolpromotion.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','code','paper_code','provider_sources','assets','labels']})
 for f in code:
  dst=D/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def head_read(model,cap):
 import torch,numpy as np
 from stvg.visual_cache import transfer
 obs=[]
 with torch.no_grad(),torch.autocast('cuda',enabled=False):
  for off in cap['temporal_cache']['offsets']:
   M=transfer(off['inputs']['encoded_feature'],'cuda');h,w=off['grid_size'];count=h*w;nf=M.shape[1];fm=M[-count:].permute(1,2,0).reshape(nf,256,h,w).detach();ft=M[:-count].mean(1).unsqueeze(0);inputs=[];outputs=[];head=model.t_spatial_clas;hooks=[head.head.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().cpu().clone())),head.head.register_forward_hook(lambda m,a,o:outputs.append(o.detach().cpu().clone()))]
   try:head(fm,ft[:,:1])
   finally:
    for hook in hooks:hook.remove()
   assert len(inputs)==len(outputs)==1;obs.append(dict(z=outputs[0][:,0].numpy(),u=inputs[0][:,0].numpy()))
 n=len(cap['frame_ids']);z=np.stack([obs[j%2]['z'][j//2]for j in range(n)]).astype(np.float64);p=np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6);u=np.stack([obs[j%2]['u'][j//2]for j in range(n)]);return p,u

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 p=verify();setup();torch.set_num_threads(1);lock=lease();started=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items());counts=dict(bridge_head_forwards=0,recheck_head_forwards=0,new_fusion=0,new_native_decoder=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_expert=0,new_observations=0,new_optimizer=0);files={}
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache()
   if ds=='hc2':
    parent=read(PARENT/'sealed.json');bridge=[]
    for row in records():
     verify();k=row['id'];cf=B/'captures'/f'{k}.pt';assert sha(cf)==read(PARENT/'plan.json')['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');e=parent['files'][k];assert sha(PARENT/e['trace_path'])==e['trace_sha256'];old=torch.load(PARENT/e['trace_path'],weights_only=False,map_location='cpu');prob,u=head_read(model,cap);assert np.array_equal(prob,old['semantic_p']['motion'])and np.array_equal(u,old['head_input']['motion']),('Cachedheadbridgebitwise',k);counts['bridge_head_forwards']+=2;bridge.append(k)
    put(D/'bridge.json',dict(count=64,learned_semantic_p_and_u_bitwise=True,original_native_memory_used=True,new_fusion=0,new_visual=0,new_expert=0,queries=bridge))
   for item in [x for x in p['items']if x['dataset']==ds]:
    verify();assert time.perf_counter()-started<p['maximum_GPU_seconds'];tick=time.perf_counter();k=item['key'];key=item['alias'];fs=item['files']
    for e in fs.values():assert sha(PROVIDER/e['path'])==e['sha256']
    cap=torch.load(PROVIDER/fs['capture']['path'],weights_only=False,map_location='cpu');post=torch.load(PROVIDER/fs['posterior']['path'],weights_only=False,map_location='cpu');ids=cap['frame_ids'];logs=[model.temp_embed(o['hidden'].cuda()).detach().cpu()for o in cap['temporal_cache']['offsets']];q0=posterior(cap['records'],logs,ids);assert all(torch.equal(a,b)for a,b in zip(logs,post['logits'])),'Cachedheadlogitsbitwise';assert np.array_equal(q0,posterior(cap['records'],post['logits'],ids)),'NativeG6cachedposteriorbitwise';assert np.max(abs(q0-post['hull']))<2e-15,'historicalhullnormalizationonly';prob,u=head_read(model,cap);counts['recheck_head_forwards']+=2
    primary,steps=loop(prob,q0);randomp=matched_random(prob,u);rand,rsteps=loop(randomp,q0);rev,vsteps=loop(prob[::-1],q0);uniform=np.where(q0>0,1.,0.);uniform/=uniform.sum();constant=np.broadcast_to(prob[0],prob.shape).copy();cq,_=loop(constant,q0);assert np.array_equal(cq,q0);sharp=q0*q0;sharp/=sharp.sum();qs=dict(NativeG6=q0,SemanticPrototypeLoop2=primary,OneStepPrototype=steps[0]['q'],DirectSemanticEvidence=update(uniform,steps[1]['S']),ReversedSemanticLoop2=rev,RandomProjectionLoop2=rand,SharpenOriginal=sharp,NoProjection=q0.copy(),ConstantProjectionNoOp=cq);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
    with np.load(PROVIDER/fs['prediction']['path'],allow_pickle=False)as z:assert np.array_equal(cap['Source'],z['Source_Native224']);arrays['Source']=z['Source_Native224'].copy();iv['Source']=z['Source_Native224_interval'].tolist()
    f=D/'predictions'/f'{key}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{key}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,semantic_p=prob,random_p=randomp,prototype_steps=steps,GT_read=False),tr);files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,dataset=ds,query_id=k);put(D/'progress.json',dict(completed=len(files),total=64,counts=counts,seconds=time.perf_counter()-started,**who()));print(key,iv['NativeG6'],iv['SemanticPrototypeLoop2'],flush=True)
   del model;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['bridge_head_forwards']==counts['recheck_head_forwards']==128;put(D/'sealed.json',dict(count=64,files=files,GT_read=False,plan_sha256=sha(D/'plan.json'),counts=counts,seconds=time.perf_counter()-started));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),**who(),partial_preserved=True));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
