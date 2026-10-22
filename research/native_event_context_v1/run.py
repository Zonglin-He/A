"""Native overlapping contexts on unchanged cached features and full text."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_context_hc64_v1';P=R/'results/native_event_context_hc64_v1'
TEXT=R/'outputs/native_event_text_hc64_v1';OLD=R/'outputs/event_query_hc64_v1'
ARMS=['NativeG6','CensoredContext','PrefixOnly','SuffixOnly','HullCalibratedFull','Sharpen','ReverseContext','UninformativeContext']
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
 p=dict(schema='native_event_context_hc64_v1',authorization='Overnight event-information loop; real contexts with original frozen trained temporal backbone; spatial method fixed',primary='CensoredContext',arms=ARMS,count=64,code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],captures=old['captures'],prior_files=read(OLD/'sealed.json')['files'],text_files=text['files'],text_seal_sha256=sha(TEXT/'sealed.json'),labels_sha256=old['labels_sha256'],
  rationale='Synthetic null denominators amplify unsupported tails and do not rank events reliably. Test real overlapping contexts with full text and unchanged feature values, recomputed native initialization and temporal positions; censor unavailable endpoints.',
  fixed=dict(windows=dict(Prefix=[0,48],Suffix=[16,64]),offsets='original parity; exact physical IDs; prefix0..48 and suffix16..64 inclusive',replay='cached original prefusion tensors; FP16 fusion then exact original FP32 two-pass native replay; complete Full hidden/posterior no-op must restore bitwise64',evidence='strict interior visible endpoints: joint log(qW/uW) if both available, start or end log marginal ratio if one, zero if neither/unsupported; uW is original zero-endpoint-logit two-offset hull, NOT uniform interval distribution',primary='q0*exp(mean of Prefix and Suffix censored log evidence), no fit or weight scan',controls='same-information full q0²/uFull; q0² sharpening; reversed-placement context; individual contexts; structural-null exact no-op',space='same8 expert observations; unchanged geometry and interval-conditioned complete-box DP; fixed-space evaluation separately'),
  new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,query_steps=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure='Repeatedly exposed HC64 development; no independent confirmation claim',qualification='Positive temporal and complete-v increments over Source and NativeG6, more improvements than harms, inspect severe and short/long tails and same-information controls; frozen cross-domain recheck before main adoption',
  references=['https://arxiv.org/abs/2012.02646','https://ojs.aaai.org/index.php/AAAI/article/view/6897'],scope='A frozen inference feasibility test; trained multi-scale/context boundary papers motivate context only, not a reproduction or validated calibration.')
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
 from stvg.temporal import final_mbr,probabilities,mass_table
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_text_v1.core import normalize
 from research.native_event_context_v1.core import censored_evidence,log_ratio
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();files={};calls=dict(fusion_offsets=0,native_replay_queries=0,new_expert=0,new_visual_encoder=0,new_text_encoder=0,new_RGB=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['text_files'][k];assert sha(TEXT/e['trace_path'])==e['trace_sha256'];old=torch.load(TEXT/e['trace_path'],map_location='cpu',weights_only=False);cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];assert len(ids)==65;tick=time.perf_counter();outputs={};nulls={};windows={};cost={};coverage={};scores={}
   for kind,bounds in [('Full',[0,64]),('Prefix',[0,48]),('Suffix',[16,64])]:
    lo,hi=bounds;window_ids=ids[lo:hi+1];windows[kind]=window_ids;views=[];recs=[];t=time.perf_counter()
    for offset,original in enumerate(old['prefusion']):
     orig_rec=old['views']['Full']['records'][offset];positions=[i for i in range(offset,len(ids),2)if lo<=i<=hi];selected=[(i-offset)//2 for i in positions];assert [ids[i]for i in range(offset,len(ids),2)]==orig_rec['frame_ids']
     kw=transfer(original,'cuda',clone=True)
     if kind!='Full':
      v=kw['videos'];ix=torch.tensor(selected,device=v.tensors.device);x=v.tensors.index_select(0,ix);m=v.mask.index_select(0,ix);kw['videos']=type(v)(x,m,[len(selected)]);kw['vid']=kw['vid'].index_select(0,ix);kw['vis_pos']=kw['vis_pos'].index_select(0,ix)
      assert torch.equal(x,transfer(original['videos'].tensors[selected],'cuda')) and torch.equal(kw['vid'],transfer(original['vid'][selected],'cuda'))
     recs.append(dict(offset=offset,flip=False,frame_ids=[ids[i]for i in positions]));prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));calls['fusion_offsets']+=1;del info,kw,prefixes
    with torch.no_grad():
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(window_ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(recs,logs,window_ids);uids,nulls[kind]=mass_table(probabilities(recs,[np.zeros((1,len(r['frame_ids']),2),np.float32)for r in recs]));assert uids==window_ids;calls['native_replay_queries']+=1;cost[kind]=time.perf_counter()-t
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6']),'Full replay must restore bitwise'
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Full hidden must restore exactly'
    else:scores[kind],coverage[kind]=censored_evidence(ids,window_ids,outputs[kind],nulls[kind])
    del base,views;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];effect=(scores['Prefix']+scores['Suffix'])/2;reverse=effect[::-1,::-1].T.copy();nullscore=sum(censored_evidence(ids,windows[x],nulls[x],nulls[x])[0]for x in ['Prefix','Suffix'])/2;assert np.array_equal(nullscore,np.zeros_like(q0));l=np.zeros_like(q0);take=q0>0;l[take]=np.log(q0[take]);qs=dict(NativeG6=q0,CensoredContext=normalize(q0,effect),PrefixOnly=normalize(q0,scores['Prefix']),SuffixOnly=normalize(q0,scores['Suffix']),HullCalibratedFull=normalize(q0,log_ratio(q0,nulls['Full'])),Sharpen=normalize(q0,l),ReverseContext=normalize(q0,reverse),UninformativeContext=q0.copy());iv={name:final_mbr(ids,q)['interval']for name,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   prior=p['prior_files'][k]
   with np.load(OLD/prior['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['NativeG6'],z['NativeG6']);assert np.array_equal(arrays['UninformativeContext'],z['NativeG6']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,null_q=nulls,window_ids=windows,score=effect,context_scores=scores,coverage=coverage,cost=cost,GT_read=False,native_hidden_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,calls=calls,**who()));print(k,iv['NativeG6'],iv['CensoredContext'],round(time.perf_counter()-tick,3),flush=True);del old,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=calls,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
