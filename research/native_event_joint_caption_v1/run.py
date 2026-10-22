"""One true native temporal read for a complete, explicitly bound event caption."""
import os,sys,time,hashlib,json,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_joint_caption_hc64_v1';P=R/'results/native_event_joint_caption_hc64_v1';TEXT=R/'outputs/native_event_text_hc64_v1';PHASE=R/'outputs/native_event_composition_hc64_v1';L=R/'outputs/overnight_event_loop_20261023'
ARMS=['NativeG6','JointCaptionFusion','DirectJoint','ImplicitSubjectFusion','PredicateBindingFusion','ReverseJointFeedback','SharpenOriginal','OldClauseFusion','OldSelfHull','NoComposition']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for root,key in [(TEXT,'text'),(PHASE,'phase')]:assert sha(root/'sealed.json')==p[key+'_seal_sha256']
 assert sha(D/'variants.json')==p['variants_sha256']
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['paper_code'].items():assert sha(R/f)==h,f
 for f,h in p['provider_sources'].items():assert sha(R.parent/'stvg-opd'/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 from stvg.query import QuerySubjectParser
 from research.native_event_joint_caption_v1.core import from_parser
 assert not(D/'plan.json').exists();parent=read(PHASE/'plan.json');parser=QuerySubjectParser(R/'checkpoints/stanza');expected=read(L/'canonical_joint_source_only_inputs.json');pre=read(L/'canonical_joint_source_only_preflight.json');assert sha(L/'canonical_joint_source_only_inputs.json')==pre['private_inputs_sha256'];assert sha(L/'canonical_joint_preflight.py')==pre['preflight_code_sha256'];old=read(PHASE/'decompositions.json');variants={}
 for row in records():
  k=row['id'];x=from_parser(row['metadata']['caption'],parser);assert x['parent']==old[k]and x['supported']==expected[k]['supported']
  if x['supported']:
   for f in ['canonical','predicate_binding_control','implicit_shared_subject']:assert x[f]==expected[k][f],(k,f)
   x['subject']=expected[k]['subject'];x['token_count']=expected[k]['canonical_tokens']
  variants[k]=x
 assert sum(x['supported']for x in variants.values())==52;put(D/'variants.json',variants);code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/joint_event_v1/run.py','research/boundary_feedback_v1/score.py','research/native_event_composition_v1/core.py','research/native_event_text_v1/core.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_joint_caption_hc64_v1',primary='JointCaptionFusion',seed=20260920,bootstrap_seed=20261022,arms=ARMS,count=64,supported=52,code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],captures=parent['captures'],text_files=parent['text_files'],text_seal_sha256=sha(TEXT/'sealed.json'),phase_files=read(PHASE/'sealed.json')['files'],phase_seal_sha256=sha(PHASE/'sealed.json'),labels_sha256=parent['labels_sha256'],variants_sha256=sha(D/'variants.json'),source_only_preflight_sha256=sha(L/'canonical_joint_source_only_preflight.json'),source_only_code_sha256=pre['preflight_code_sha256'],design_sha256=sha(L/'next_joint_caption_design.md'),authorization='User overnight event-information loop until07; no added expert/observations, fixed space',rationale='Independent positive clause envelopes improve longbut damage short, and allmiddle ordering/sharedquery/coupling didnotfix. Test one COMPLETEpositive jointly read caption with explicit repeatedidentity bindings, NOT a different hull/ratio/queryinterface.',fixed=dict(grammar='Exactold52SourceStanza mainANDscope/alloriginalclauses+arguments+nominalidentity subtree. Onejointtext joinsfullpositiveclauses withand, explicitfullsubject perpredicate.12unsupported everyinterventionnative, noGT/qualityveto.',controls='SameinformationDirectJoint; ImplicitSharedSubject removesnominalsubtree onlylaterconjuncts; PredicateBindingControlcyclesonlymainverb surfacewords amongsamearguments, EXACTsameBPEtokenbag/length actualtokenizer52. WordslotsnotliteralunchangedBPEpositions, canbeungrammatical/accidentallyvalid, notknownnegative. ReverseJoint mirrorsqindices then.5mix, preservesindexspan notexactseconds/backwardvideo. SharpenOriginal normalizedq0squared same52scope. OldClauseFusion/OldSelfHull strictparentq/interval/finalspacealiases0newcalls.',native='OriginalFullsubjectprefix FIXEDnotreparsed. Wholecaption actualRoBERTa/resizer+FP16fusion+FP32native2pass/6layers/2offsethull, original65timepositions/physicalMBR andcompleteboxspace/evidence8. No independentclauseprediction/newenvelope/monotonicDP/featureediting/querylearning. Full64texttuple/hidden/logits/q/Sourceboxbitwise.',cost='64Full+3*52positive/controltexts=220actualtext/nativequeries440fusionoffset. Allvisionprefusioncacheoriginal;0RGB/visualencoder/expert/observations/optimizer. Researchreadouttiming notdeploymentcost.'),weights_frozen=True,new_RGB=0,new_visual_encoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all64_seal_before_new_GT=True,maximum_GPU_seconds=1200,stop_epoch=1792708800,exposure=parent['exposure'],qualification='Primaryactualt/fullv>SourceANDG6,Source shortANDlongpositive,win>harm,primarybeyondImplicit/PredicateBinding/Reverse/Sharpen thenfixeddual-domainrecheck. OldfailedshortClauseFusion mean notmustbeat. No template/repetition/fusion/lr/steps scan orcontrolpromotion.',references=parent['references'],scope='Own fixed source-only complete-caption composition feasibility; not trainedDeCo orTFVTG additionalLLM/VLM reproduction; not claimed causaleventprobability')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['paper_code','provider_sources','assets','captures','text_files','phase_files']})
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
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(R.parent/'stvg-opd'/f)==h for f,h in p['assets'].items());model=load_source(read(R/'configs/default.json')['datasets']['hc2']);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache();variants=read(D/'variants.json');files={};counts=dict(text_forwards=0,fusion_offsets=0,native_replay_queries=0,new_expert=0,new_observations=0,new_RGB=0,new_visual_encoder=0,new_optimizer=0)
 try:
  for row in records():
   verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];k=row['id'];te=p['text_files'][k];pe=p['phase_files'][k]
   for root,e in [(TEXT,te),(PHASE,pe)]:assert sha(root/e['trace_path'])==e['trace_sha256']and sha(root/e['path'])==e['sha256']
   old=torch.load(TEXT/te['trace_path'],weights_only=False,map_location='cpu');oldphase=torch.load(PHASE/pe['trace_path'],weights_only=False,map_location='cpu');cf=B/'captures'/f'{k}.pt';assert sha(cf)==p['captures'][k]['capture_sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');ids=cap['frame_ids'];v=variants[k];subject=old['text']['subject'];tick=time.perf_counter();outputs={};boxes={};cost={}
   captions=[('Full',row['metadata']['caption'])]
   if v['supported']:
    assert v['subject']==subject;tokens=model.text_encoder.tokenizer([subject+' '+v['canonical'],subject+' '+v['predicate_binding_control']],padding=False)['input_ids'];assert len(tokens[0])==v['token_count']and sorted(tokens[0])==sorted(tokens[1]),'ActualSourceBPEbag/lengthcounterfactual'
    captions += [('Joint',v['canonical']),('Implicit',v['implicit_shared_subject']),('Binding',v['predicate_binding_control'])]
   for kind,caption in captions:
    t=time.perf_counter();views=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):text,_=model.text_encoder([subject+' '+caption],'cuda')
    counts['text_forwards']+=1
    for original in old['prefusion']:
     kw=transfer(original,'cuda',clone=True)
     if kind=='Full':assert len(text)==len(kw['texts'])and all(torch.equal(a,b)for a,b in zip(text,kw['texts'])),'Full texttuplebitwise'
     kw['texts']=text;prefixes=[];hook=model.ground_encoder.encoder.norm.register_forward_pre_hook(lambda m,a:prefixes.append(transfer(a[0],'cpu',clone=True)))
     try:
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):info=model.ground_encoder(**kw)
     finally:hook.remove()
     assert len(prefixes)==1;views.append(dict(prefix=prefixes[0],info=transfer(info,'cpu',clone=True),vis_pos=transfer(kw['vis_pos'],'cpu',clone=True)));counts['fusion_offsets']+=1;del kw,info,prefixes
    with torch.no_grad():base=FrozenNativeReplay(model,transfer(views,'cuda'),len(ids));logs=[model.temp_embed(h).cpu()for h in base.temporal_inputs]
    outputs[kind]=posterior(old['views']['Full']['records'],logs,ids);boxes[kind]=base.zero['boxes'].cpu().numpy();cost[kind]=time.perf_counter()-t;counts['native_replay_queries']+=1
    if kind=='Full':
     assert np.array_equal(outputs[kind],old['q']['NativeG6'])and np.array_equal(boxes[kind],cap['Source'])
     for a,b in zip(base.temporal_inputs,old['views']['Full']['hidden']):assert torch.equal(a.cpu(),b),'Fullhiddenbitwise'
     for a,b in zip(logs,old['views']['Full']['logits']):assert torch.equal(a,b),'Fulllogitsbitwise'
    del base,views,text;gc.collect();torch.cuda.empty_cache()
   q0=outputs['Full'];qs={a:q0.copy()for a in ARMS}
   if v['supported']:
    qs.update(JointCaptionFusion=.5*q0+.5*outputs['Joint'],DirectJoint=outputs['Joint'],ImplicitSubjectFusion=.5*q0+.5*outputs['Implicit'],PredicateBindingFusion=.5*q0+.5*outputs['Binding'],ReverseJointFeedback=.5*q0+.5*outputs['Joint'][::-1,::-1].T.copy(),SharpenOriginal=normalize(q0,np.log(np.maximum(q0,1e-300))),OldClauseFusion=oldphase['q']['ClauseFusion'].copy(),OldSelfHull=oldphase['q']['SelfHullFusion'].copy())
   iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
   with np.load(PHASE/pe['path'],allow_pickle=False)as z:
    for a,pa in [('NativeG6','NativeG6'),('NoComposition','NativeG6'),('OldClauseFusion','ClauseFusion'),('OldSelfHull','SelfHullFusion')]:assert np.array_equal(arrays[a],z[pa])and iv[a]==z[pa+'_interval'].tolist(),a
    arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
   if not v['supported']:assert all(np.array_equal(q,q0)for q in qs.values())
   score=np.zeros_like(q0);take=q0>0;score[take]=np.log(qs['JointCaptionFusion'][take])-np.log(q0[take]);f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':span for a,span in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,intervened_q=outputs,score=score,clauses=dict(supported=v['supported']),variant_source_boxes=boxes,cost=cost,GT_read=False,Full_bitwise=True,actual_predicate_BPE_bag_match=v['supported']),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick,supported=v['supported'],cost=cost);put(D/'progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-start,counts=counts,**who()));print(k,v['supported'],iv['NativeG6'],iv['JointCaptionFusion'],flush=True);del old,oldphase,cap,geom,arrays;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['text_forwards']==220 and counts['native_replay_queries']==220 and counts['fusion_offsets']==440 and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'sealed.json',dict(count=64,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**who()));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
