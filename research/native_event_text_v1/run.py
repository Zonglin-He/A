"""HC64 true native text intervention; no experts and no spatial changes."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_text_hc64_v1';P=R/'results/native_event_text_hc64_v1';OLD=R/'outputs/event_query_hc64_v1';ARMS=['NativeG6','Selective','EventOnly','ControlOnly','IdentityContrast','Sharpen','ReverseSelective']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();paper=read(R/'outputs/tastvg_paper_v1/plan.json');prior=read(OLD/'sealed.json');old=read(OLD/'plan.json')
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_text_hc64_v1',authorization='User native-event information loop until2026-10-23 07:00 Asia/Shanghai, no added expert',primary='Selective',arms=ARMS,queries=64,code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],captures=old['captures'],assets=paper['assets'],prior_files=prior['files'],labels_sha256=old['labels_sha256'],manifest_sha256=old['manifest_sha256'],
  difference_from_25daec='Old actually recomputed whole static phrase at final-layer readout and usedq0^2/qStatic. Here main-clause predicate token IDs masked before true RoBERTa encoding, equal BPE-count function-token masking protects same subject subtree, identical token positions/masks; both native offsets/sixlayers. Primary q0*qControl/qEvent separates action-specific response from generic masked-text perturbation; identity q0^2/qId retained only old-route control.',
  fixed=dict(event='only root/main-clause coordinated verb tokens; preserve noun-attached identity predicates',control='same number of nonprotected function-word BPEs in deterministic text order; record unsupported if unavailable, no GT',identity='original queried nominal dependency subtree, same forced subject prefix; diagnostic not guaranteed identity equivalence',readout='unit difference-of-differences logqControl-logqEvent on native support + originalq0, original physicalMBR',space='reuse same capture complete-box geometry/teacher8 and DP, fixed-native-space metrics also',visual='exact native VisualCache4GiB text-independent encoders only; real text encoder/fusion/native two-pass/time head recomputed per intervention'),
  new_expert=0,new_observations=0,query_steps=0,weights_frozen=True,GT_in_inference=False,all64_seal_before_new_GT=True,exposure='repeated HC64 development; not independent; fixed recheck required',maximum_GPU_seconds=3600,stop_epoch=1792708800, #06:40 local
  references=['https://github.com/CFM-MSG/CAEM_Code','https://arxiv.org/abs/2207.13457'],literature_scope='CAEM/D-TSG train debiasing/event matching; we only test a frozen-native intervention, not reproduce their learned causal estimators')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['captures','prior_files','paper_code','provider_sources','assets']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.query import QuerySubjectParser
 from stvg.decode import decode
 from stvg.native import frozen_native_forward
 from stvg.visual_cache import VisualCache,transfer
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_text_v1.core import intervention,masked_inputs,posteriors
 p=verify();setup();lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert not(D/'sealed.json').exists();cfg=read(R/'configs/default.json')['datasets']['hc2'];assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(cfg);parser=QuerySubjectParser(R/'checkpoints/stanza');cache=VisualCache(4*2**30);files={};counts=dict(native_queries=0,visual_misses=0,visual_hits=0,text_forwards=0,new_expert=0);model.eval().requires_grad_(False)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];capfile=B/'captures'/f'{k}.pt';assert sha(capfile)==p['captures'][k]['capture_sha256'];cap=torch.load(capfile,map_location='cpu',weights_only=False);prior=p['prior_files'][k];assert sha(OLD/prior['trace_path'])==prior['trace_sha256'];oldtr=torch.load(OLD/prior['trace_path'],weights_only=False,map_location='cpu');qold=oldtr['q']['NativeG6'];meta=row['metadata'];subject=parser(meta['caption'])['subject'];inter=intervention(meta['caption'],subject,parser,model.text_encoder.tokenizer);frames,ids=decode(meta,'hc2');assert ids==cap['frame_ids'];tick=time.perf_counter();out={};native_boxes={};cost={};prefusion=[];views={};before=cache.stats()
   for name in ['Full','Event','Control','Identity']:
    if name!='Full'and not inter['supported']:
     out[name]=out['Full'];continue
    rr=dict(input=dict(meta),frame_ids=ids,parses=dict(subject=subject))
    if name=='Identity':rr['input']['caption']=inter['identity_caption']
    changed=[];hook=None;prefixhook=None
    if name in ['Event','Control']:
     positions=inter['event_positions'if name=='Event'else'control_positions']
     def modify(module,args,kwargs):
      assert not args and torch.equal(kwargs['input_ids'].cpu(),inter['full_input_ids']);new=dict(kwargs);new['input_ids']=masked_inputs(kwargs['input_ids'],positions,model.text_encoder.tokenizer.mask_token_id);changed.append(True);return args,new
     hook=model.text_encoder.body.register_forward_pre_hook(modify,with_kwargs=True)
    if name=='Full':
     def preserve(module,args,kwargs):
      assert not args;prefusion.append(transfer(kwargs,'cpu',clone=True))
     prefixhook=model.ground_encoder.register_forward_pre_hook(preserve,with_kwargs=True)
    t=time.perf_counter()
    try:
     with torch.no_grad(),cache.context(model,frames,ids):batch,recs,base=frozen_native_forward(model,frames,rr)
    finally:
     if hook:hook.remove()
     if prefixhook:prefixhook.remove()
    if name in ['Event','Control']:assert len(changed)==2
    with torch.no_grad():logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    counts['native_queries']+=1;counts['text_forwards']+=2;out[name]=posterior(recs,logs,ids);native_boxes[name]=base.zero['boxes'].cpu().numpy();views[name]=dict(logits=logs,records=recs,hidden=[h.cpu()for h in base.temporal_inputs]);cost[name]=time.perf_counter()-t
    if name=='Full':
     assert np.array_equal(out[name],qold),'Full native posterior must restore bitwise before any feedback'
     assert np.array_equal(native_boxes[name],cap['Source']),'Full source boxes must restore bitwise'
     for a,b in zip(base.temporal_inputs,cap['temporal_cache']['offsets']):assert torch.equal(a.cpu(),b['hidden'])
    del batch,base;gc.collect();torch.cuda.empty_cache()
   qs,score=posteriors(out['Full'],out['Event'],out['Control'],out['Identity']);assert np.array_equal(qs['NativeG6'],qold);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);torch.set_num_threads(1);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS};torch.set_num_threads(4)
   with np.load(OLD/prior['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['NativeG6'],z['NativeG6']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions'])
   tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=out,score=score,text=inter,intervention_source_boxes=native_boxes,views=views,prefusion=prefusion,cost=cost,GT_read=False),tr)
   after=cache.stats();counts['visual_misses']+=after['misses']-before['misses'];counts['visual_hits']+=after['hits']-before['hits'];files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,cost=cost,supported=inter['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,inter['supported'],{a:iv[a]for a in ['NativeG6','Selective']},round(time.perf_counter()-tick,3),flush=True);del cap,frames,views,prefusion,oldtr,arrays;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  put(D/f'failure-{time.time_ns()}.json',dict(traceback=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True,**who()));raise
