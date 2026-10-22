"""Read positive clauses using the original full-caption second-pass time query."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_shared_query_hc64_v1';P=R/'results/native_event_shared_query_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PARENT=R/'outputs/native_event_composition_hc64_v1'
ARMS=['NativeG6','SharedQueryClauseFusion','DirectSharedClauses','ClauseFusion','SelfHullFusion','IdentityHullFusion','ReverseSharedClauses','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PARENT,'parent')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 assert sha(PARENT/'decompositions.json')==p['decompositions_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 assert not(D/'plan.json').exists();parent=read(PARENT/'plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_shared_query_hc64_v1',primary='SharedQueryClauseFusion',arms=ARMS,count=64,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],parent_files=read(PARENT/'sealed.json')['files'],parent_seal_sha256=sha(PARENT/'sealed.json'),text_seal_sha256=sha(TEXT/'sealed.json'),decompositions_sha256=sha(PARENT/'decompositions.json'),supported=parent['supported'],labels_sha256=parent['labels_sha256'],
 authorization='User overnight event-information loop, no additional expert/observations, unchanged spatial method',rationale='Positive clauses improve mean but short events suffer, order reversal nearly equal. Clause Full-source box proxy drifts(.8038first/.7479last,8last<.5); not identity truth and not an acceptance veto. Retain the original full-caption native temporal query during clause reads to test conditioning drift.',
 difference_from_history='Object-conditioned temporal query and actual query_tgt writes already tested (d0bc2395/fc369f37), and oldTubeReplay changes per-head cross attention on full-caption memory. Here only original Full second-pass temporal query_tgt is shared across actual positive first/last clause memories; no new query interface/attention-ROI novelty claimed.',
 fixed=dict(members='Identical52/64 query-only main-AND clauses/wording from parent sealed protocol; other12 exactNative',intervention='Scoped actual TimeDecoder prehook at calls1,3 of native four calls; restore same-offset original full query vector with actual expanded stride, leave first pass and all other kwargs unchanged. New clause text+motion memory/query_content remain. No fitting, query steps, ROI attention edit or identity claim.',no_op='Full replacement text/hidden/posterior/Source boxes must restore bitwise64; all replaced Full vectors equal actual current native vectors; hook removed on error',readout='Same independent first/last minimal-envelope posterior, same0.5Full fusion, original physical MBR and complete-box DP',controls='UnsharedClauseFusion same texts and native evidence; SelfHullFusion pure repeated native expansion; IdentityHullFusion; same-information direct shared envelope; reversed-placement shared clauses; no-composition exact',space='Same full geometry/expert8, per-interval DP and fixed-space evaluation'),
 GT_in_inference=False,all64_seal_before_new_GT=True,weights_frozen=True,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification=parent['qualification'],references=parent['references'],scope='Full native query contains action/global context as well as possible identity; do not call it a pure identity representation or prove identity preservation from Source boxes.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','parent_files']})
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
 from research.native_event_shared_query_v1.core import shared_second_pass
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();dc=read(PARENT/'decompositions.json');files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,shared_second_pass_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['text_files'][k];assert sha(TEXT/e['trace_path'])==e['trace_sha256'];old=torch.load(TEXT/e['trace_path'],map_location='cpu',weights_only=False);cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,map_location='cpu',weights_only=False);pe=p['parent_files'][k];assert sha(PARENT/pe['path'])==pe['sha256']and sha(PARENT/pe['trace_path'])==pe['trace_sha256'];prior=torch.load(PARENT/pe['trace_path'],map_location='cpu',weights_only=False);ids=cap['frame_ids'];subject=old['text']['subject'];clauses=dc[k];tick=time.perf_counter();out={};cost={};hook_audit={};actual_inputs={};hidden={};fullq=[x['inputs']['query_tgt']for x in cap['temporal_cache']['offsets']]
   variants=[('Full',row['metadata']['caption'])]+([('First',clauses['first']),('Last',clauses['last'])]if clauses['supported']else[])
   for kind,caption in variants:
    t=time.perf_counter();views=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):text,_=model.text_encoder([subject+' '+caption],'cuda')
    counts['text_forwards']+=1
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if kind=='Full':assert all(torch.equal(x,y)for x,y in zip(text,kw['texts'])),'Full text bitwise'
     kw['texts']=text;prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad(),shared_second_pass(model.ground_decoder.time_decoder,fullq)as audit:
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    assert not model.ground_decoder.time_decoder._forward_pre_hooks;counts['shared_second_pass_queries']+=2;hook_audit[kind]=audit
    out[kind]=posterior(old['views']['Full']['records'],logs,ids);actual_inputs[kind]=transfer(base.time_inputs,'cpu',clone=True);hidden[kind]=[h.cpu()for h in base.temporal_inputs];cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert all(x.get('same_values',True)for x in audit),'Full native query replacement must be exact'
     assert np.array_equal(out[kind],old['q']['NativeG6'])and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
     for x,y in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(x.cpu(),y),'Full hidden bitwise'
    del base,views,text;gc.collect();torch.cuda.empty_cache()
   q0=out['Full'];qs={a:q0.copy()for a in ARMS}
   if clauses['supported']:
    comp=envelope(out['First'],out['Last']);reverse=envelope(out['First'][::-1,::-1].T.copy(),out['Last'][::-1,::-1].T.copy());qs.update(SharedQueryClauseFusion=.5*q0+.5*comp,DirectSharedClauses=comp,ClauseFusion=prior['q']['ClauseFusion'],SelfHullFusion=prior['q']['SelfHullFusion'],IdentityHullFusion=prior['q']['IdentityHullFusion'],ReverseSharedClauses=.5*q0+.5*reverse)
   iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PARENT/pe['path'],allow_pickle=False)as z:
    for a in ['NativeG6','ClauseFusion','SelfHullFusion','IdentityHullFusion','NoComposition']:assert np.array_equal(arrays[a],z[a])and iv[a]==z[a+'_interval'].tolist()
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['SharedQueryClauseFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=out,score=score,clauses=clauses,actual_time_inputs=actual_inputs,hidden=hidden,hook_audit=hook_audit,cost=cost,GT_read=False,full_text_hidden_posterior_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,cost=cost,supported=clauses['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,clauses['supported'],iv['NativeG6'],iv['SharedQueryClauseFusion'],flush=True);del old,cap,prior,geom,arrays,actual_inputs,hidden;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
