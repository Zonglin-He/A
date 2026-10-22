"""Positive coordinated subevent queries through the same frozen native model."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_composition_hc64_v1';P=R/'results/native_event_composition_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';OLD=R/'outputs/event_query_hc64_v1'
ARMS=['NativeG6','ClauseFusion','DirectClauses','SelfHullFusion','IdentityHullFusion','MaskedEventFusion','ReverseClauseFusion','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user';assert sha(TEXT/'sealed.json')==p['text_seal_sha256'];assert sha(D/'decompositions.json')==p['decompositions_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p
def prepare():
 from stvg.query import QuerySubjectParser
 from research.native_event_composition_v1.core import decompose
 assert not(D/'plan.json').exists();paper=read(R/'outputs/tastvg_paper_v1/plan.json');text=read(TEXT/'sealed.json');old=read(OLD/'plan.json');parser=QuerySubjectParser(R/'checkpoints/stanza');dc={row['id']:decompose(row['metadata']['caption'],parser)for row in records()};put(D/'decompositions.json',dc);code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_composition_hc64_v1',primary='ClauseFusion',arms=ARMS,count=64,code=code,paper_code=paper['code'],provider_sources=paper['provider_sources'],assets=read(TEXT/'plan.json')['assets'],captures=old['captures'],prior_files=read(OLD/'sealed.json')['files'],text_files=text['files'],text_seal_sha256=sha(TEXT/'sealed.json'),decompositions_sha256=sha(D/'decompositions.json'),supported=sum(x['supported']for x in dc.values()),labels_sha256=old['labels_sha256'],
 authorization='Same user overnight loop; no added expert model/observations, existing source query parser and frozen original temporal model only',rationale='Concrete remaining failures describe multiple same-person actions:approach+sit,raise+pour+turn. Full-caption endpoints may select one primitive rather than span the described compound event. Test genuine positive subevent reads, not masked/static absence likelihood ratios or mere position perturbation.',
 fixed=dict(decomposition='Existing Source Stanza parser only. Single main VERB plus main-chain VERB conjs, one inherited subject, no explicit new subject or before/after/while/until/when/unless/or/but relation. Retain original nominal identity subtree and actual predicate/arguments. Read first and last conjunct only, max2 texts; all unsupported captions keep full posterior exactly. No GT/quality selection or learned grammar.',readout='Native original two-offset sixlayer posterior for each positive clause. Exact distribution of minimal envelope of first+last independent intervals, not averaging boxes or choosing GT-best subevent. Primary0.5qFull+0.5qClauseEnvelope, original physical MBR and spatial DP; no ratio/parameter updates or weight sweep.',controls='DirectClauses same information; SelfHullFusion uses repeated original qFull to isolate mere interval expansion; IdentityHullFusion uses prior same-target nominal query; MaskedEventFusion uses existing predicate-masked posterior; ReverseClauseFusion reverses both clause posterior placements; NoComposition exact no-op. Same supported set for all intervention arms.',replay='Reuse cached full visual projected features, no RGB/vision/model expansion. Actual native RoBERTa+resizer, FP16 fusion and original FP32 two-pass decoder. Full reencoded text tuple and hidden/posterior must restore bitwise64; Original subject prefix fixed for all clauses.',space='Same complete-box geometry, same expert8 and DP; both actual per-interval output and fixed NativeG6 spatial path evaluated'),
 GT_in_inference=False,all64_seal_before_new_GT=True,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure='Repeatedly exposed HC64 development; not independent confirmation',qualification='Positive actual temporal and complete-v increments over Source and Global6, more improvements than harms, severe/short-long audit, beat same-information expansion controls, frozen HC/Vid recheck before adoption',references=['https://github.com/minghangz/TFVTG','https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10687.pdf','https://openaccess.thecvf.com/content/CVPR2023/html/Yang_DeCo_Decomposition_and_Reconstruction_for_Compositional_Temporal_Grounding_via_Coarse-To-Fine_CVPR_2023_paper.html'],scope='TFVTG uses extra LLM/VLM and DeCo trains representations; neither reproduced. This is weaker deterministic main-AND composition through the existing trained native temporal model only.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','prior_files','text_files']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'],p['supported'])
def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.visual_cache import transfer
 from stvg.native import FrozenNativeReplay
 from stvg.global6 import posterior
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_composition_v1.core import envelope
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();dc=read(D/'decompositions.json');files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];e=p['text_files'][k];assert sha(TEXT/e['trace_path'])==e['trace_sha256'];old=torch.load(TEXT/e['trace_path'],map_location='cpu',weights_only=False);cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,map_location='cpu',weights_only=False);ids=cap['frame_ids'];subject=old['text']['subject'];clauses=dc[k];tick=time.perf_counter();out={};cost={};phase_boxes={}
   variants=[('Full',row['metadata']['caption'])]+([('First',clauses['first']),('Last',clauses['last'])]if clauses['supported']else[])
   for kind,caption in variants:
    t=time.perf_counter();views=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):text,_=model.text_encoder([subject+' '+caption],'cuda')
    counts['text_forwards']+=1
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if kind=='Full':
      assert len(text)==len(kw['texts']) and all(torch.equal(x,y)for x,y in zip(text,kw['texts'])),'True original text encoding must restore bitwise'
     kw['texts']=text;prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():
     base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    out[kind]=posterior(old['views']['Full']['records'],logs,ids);phase_boxes[kind]=base.zero['boxes'].cpu().numpy();cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(out[kind],old['q']['NativeG6'])and np.array_equal(phase_boxes[kind],cap['Source'])
     for x,y in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(x.cpu(),y),'Full hidden bitwise'
    del base,views,text;gc.collect();torch.cuda.empty_cache()
   q0=out['Full'];qs={a:q0.copy()for a in ARMS}
   if clauses['supported']:
    comp=envelope(out['First'],out['Last']);rev=envelope(out['First'][::-1,::-1].T.copy(),out['Last'][::-1,::-1].T.copy());qs.update(ClauseFusion=.5*q0+.5*comp,DirectClauses=comp,SelfHullFusion=.5*q0+.5*envelope(q0,q0),IdentityHullFusion=.5*q0+.5*envelope(q0,old['intervened_q']['Identity']),MaskedEventFusion=.5*q0+.5*old['intervened_q']['Event'],ReverseClauseFusion=.5*q0+.5*rev)
   assert np.array_equal(qs['NoComposition'],q0);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS};prior=p['prior_files'][k]
   with np.load(OLD/prior['path'],allow_pickle=False)as z:
    assert np.array_equal(arrays['NativeG6'],z['NativeG6']);arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['ClauseFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=out,score=score,clauses=clauses,phase_source_boxes=phase_boxes,cost=cost,GT_read=False,full_text_hidden_posterior_bitwise=True),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,cost=cost,supported=clauses['supported']);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,clauses['supported'],iv['NativeG6'],iv['ClauseFusion'],round(time.perf_counter()-tick,3),flush=True);del old,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
