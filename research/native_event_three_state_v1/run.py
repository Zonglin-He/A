"""Cached-only semantic three-state boundary test, same information and two rounds."""
import os,sys,time,json,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put,records,B
D=R/'outputs/native_event_three_state_v1';P=R/'results/native_event_three_state_v1';L=R/'outputs/overnight_event_loop_20261023';FIRST=R/'outputs/native_event_semantic_prototype_hc64_v1';RECHECK=R/'outputs/native_event_semantic_recheck_v1'
ARMS=['NativeG6','ThreeStateLoop2','OneStepThreeState','DirectThreeState','UnorderedBackgroundLoop2','SwappedBackgroundLoop2','ReversedThreeStateLoop2','RandomThreeStateLoop2','BinaryPrototypeLoop2','SharpenOriginal','ConstantProjectionNoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for name,root in [('first',FIRST),('recheck',RECHECK)]:assert sha(root/'sealed.json')==p[name+'_seal_sha256']and sha(root/'score_receipt.json')==p[name+'_score_receipt_sha256']
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();first=read(FIRST/'plan.json');second=read(RECHECK/'plan.json');items=[]
 for row in records():
  k=row['id'];cap=B/'captures'/f'{k}.pt';items.append(dict(group='HC64',dataset='hc2',alias=k,key=k,parent=row['metadata']['source'],row=row,source='first',entry=read(FIRST/'sealed.json')['files'][k],capture=dict(path=str(cap),sha256=sha(cap))))
 for item in second['items']:items.append(dict(group=item['dataset']+'_recheck',dataset=item['dataset'],alias=item['alias'],key=item['key'],parent=item['parent'],row=item['row'],source='recheck',entry=read(RECHECK/'sealed.json')['files'][item['alias']],capture=item['files']['capture']))
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/native_event_semantic_prototype_v1/core.py','research/native_event_semantic_recheck_v1/score.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_three_state_v1',primary='ThreeStateLoop2',arms=ARMS,count=128,code=code,paper_code=first['paper_code'],provider_sources=first['provider_sources'],assets=first['assets'],first_seal_sha256=sha(FIRST/'sealed.json'),first_score_receipt_sha256=sha(FIRST/'score_receipt.json'),recheck_seal_sha256=sha(RECHECK/'sealed.json'),recheck_score_receipt_sha256=sha(RECHECK/'score_receipt.json'),postseal_mechanism_audit_sha256=sha(R/'results/native_event_semantic_recheck_v1/MECHANISM_AUDIT.json'),items=items,labels=second['labels'],development_labels_sha256=first['labels_sha256'],seed=20260920,bootstrap_seed=20261022,stop_epoch=1792708800,maximum_CPU_seconds=1200,authorization='User overnight event-feedbackloop, unchangedspace/same8expertobservations; failure-based minimalchange, no parameter scans',rationale='HC64 binary semantic loop +2.927t/+1.271v but fixedHC32v-.578 andVid32t-.615/v-.269, Sourceshortnegative both. HC2recheck014 preeventAUC.467/post.899;recheck011 prepostsemanticKL.0875 vsinsideoutside.0162. Pooled outside maymix distinct before/after states. Hypothesis notprovedcause; no GTstatefitonline.',fixed=dict(state='Before(t<s),Event(s<=t<=e),After(t>e) softmembership derivedonlyfromq. Crossfitoppositeoffset SAME learnedmotionp classcoordinates. ThreeweightedBernoullimeans; undefinedzero-weight statefallbackfitparitymean recorded. No GT/qualityrefusal.',scoring='FULLtimeline expectedBernoullilogpseudolikelihood assignsALLframes tobefore/event/after. Subtractperframebefore likelihood commonconstant. Legalpairscore includespre+inclusiveevent+post, no eventlengthparameter; qnew proportional FIXEDq0*exp(score). EXACT2rounds asbinaryparent.',controls='Same3statesandamount UnorderedBackground uses equalmixture before/afterdensities onbothoutside sides; SwappedBackground flips learnedbefore/afterroles; reverseallsemanticrows; exactparenthistogramrandomprojection; same-second-evidenceDirectuniformprior; OneStep; Binary parentexactalias;Sharpenparentexactalias;constantprojectionexactq0.',scope='Temporalstatefactorization changed ONLY, no new signals/models/querytext/vision/hull/MBR/space, notlearnedverb-orderrecognition. Classes/timecorrelated. Source priormaymisswrongoccurrence. Nonewgain/iterations/classcountnorm/thresholdscan. All128 predictions seal before anynewCPUscore; panelsalreadyexposed, notindependentconfirmation.'),qualification='HC64andBOTHfixedpanelsSource/G6actualt/fullvpositive,win>harm,Source shortANDlongpositive,beyondrandom/reverse/sharpenandunorderedstatecontrol. Do notpromotecontrolposthoc.',new_model_calls=0,new_expert=0,new_observations=0,GT_in_inference=False,all128_seal_before_GT=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','code','paper_code','provider_sources','assets','labels']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def predict():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy as np
 sys.path.insert(0,str(R/'external/TA-STVG'));import utils.misc,utils.video_list
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_three_state_v1.core import loop
 from research.native_event_semantic_prototype_v1.core import update
 p=verify();torch.set_num_threads(1);start=time.perf_counter();put(D/'state.json',dict(status='predicting_CPU',fingerprint=p['fingerprint'],**who()));files={}
 for item in p['items']:
  verify();assert time.perf_counter()-start<p['maximum_CPU_seconds'];root=FIRST if item['source']=='first'else RECHECK;e=item['entry'];assert sha(root/e['path'])==e['sha256']and sha(root/e['trace_path'])==e['trace_sha256'];cf=PROVIDER/item['capture']['path'];assert sha(cf)==item['capture']['sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');old=torch.load(root/e['trace_path'],weights_only=False,map_location='cpu');q0=old['q']['NativeG6'];prob=old['semantic_p']['motion']if item['source']=='first'else old['semantic_p'];ids=cap['frame_ids'];primary,steps=loop(prob,q0);unordered,usteps=loop(prob,q0,'unordered');swap,ssteps=loop(prob,q0,'swap');reverse,rsteps=loop(prob[::-1],q0);random,randomsteps=loop(old['random_p'],q0);constant,_=loop(np.broadcast_to(prob[0],prob.shape),q0);assert np.array_equal(constant,q0);uniform=np.where(q0>0,1.,0.);uniform/=uniform.sum();qs=dict(NativeG6=q0,ThreeStateLoop2=primary,OneStepThreeState=steps[0]['q'],DirectThreeState=update(uniform,steps[1]['S']),UnorderedBackgroundLoop2=unordered,SwappedBackgroundLoop2=swap,ReversedThreeStateLoop2=reverse,RandomThreeStateLoop2=random,BinaryPrototypeLoop2=old['q']['SemanticPrototypeLoop2'],SharpenOriginal=old['q']['SharpenOriginal'],ConstantProjectionNoOp=constant);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
  with np.load(root/e['path'],allow_pickle=False)as z:
   for a,b in [('NativeG6','NativeG6'),('BinaryPrototypeLoop2','SemanticPrototypeLoop2'),('SharpenOriginal','SharpenOriginal'),('ConstantProjectionNoOp','NativeG6')]:assert np.array_equal(arrays[a],z[b])and iv[a]==z[b+'_interval'].tolist()
   arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
  k=item['alias'];f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,steps=steps,GT_read=False),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr));put(D/'progress.json',dict(completed=len(files),total=128,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['ThreeStateLoop2'],flush=True)
 verify();assert len(files)==128 and not torch.cuda.is_initialized();put(D/'sealed.json',dict(count=128,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=dict(new_model_calls=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_expert=0,new_observations=0,new_optimizer=0),seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),**who(),partial_preserved=True));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
