"""Original word semantic projection on complete-box target support; cached views."""
import os,sys,time,json,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));PROVIDER=R.parent/'stvg-opd'
from research.joint_event_v1.run import sha,read,put
D=R/'outputs/native_event_target_semantic_v1';P=R/'results/native_event_target_semantic_v1';L=R/'outputs/overnight_event_loop_20261023';FIRST=R/'outputs/native_event_semantic_prototype_hc64_v1';RECHECK=R/'outputs/native_event_semantic_recheck_v1'
ARMS=['NativeG6','TargetSemanticLoop2','OneStepTarget','DirectTargetEvidence','ReflectedSemanticLoop2','WrongTimeSemanticLoop2','RandomTargetSemanticLoop2','ParentGlobalSemanticLoop2','SharpenOriginal','AllKeysNoOp']
def who():return dict(pid=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]))
def verify():
 p=read(D/'plan.json');assert read(R/'outputs/tastvg_paper_v1/state.json')['status']=='paused_by_user'
 for name,root in [('first',FIRST),('recheck',RECHECK)]:assert sha(root/'sealed.json')==p[name+'_seal_sha256']and sha(root/'score_receipt.json')==p[name+'_score_receipt_sha256']
 for root,key in [(R,'code'),(R,'paper_code'),(PROVIDER,'provider_sources')]:
  for f,h in p[key].items():assert sha(root/f)==h,f
 assert time.time()<p['stop_epoch'];return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(R/'outputs/native_event_three_state_v1/plan.json');code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 for f in ['research/native_event_semantic_prototype_v1/core.py','research/native_event_support_view_v1/core.py','research/native_event_spatial_text_v1/core.py','research/native_event_semantic_recheck_v1/score.py','research/joint_event_v1/run.py']:code[f]=sha(R/f)
 p=dict(schema='native_event_target_semantic_v1',primary='TargetSemanticLoop2',arms=ARMS,count=128,items=parent['items'],labels=parent['labels'],development_labels_sha256=parent['development_labels_sha256'],code=code,paper_code=parent['paper_code'],provider_sources=parent['provider_sources'],assets=parent['assets'],first_seal_sha256=sha(FIRST/'sealed.json'),first_score_receipt_sha256=sha(FIRST/'score_receipt.json'),recheck_seal_sha256=sha(RECHECK/'sealed.json'),recheck_score_receipt_sha256=sha(RECHECK/'score_receipt.json'),input_selection_sha256=sha(R/'outputs/native_event_three_state_v1/plan.json'),source_mask_audit_sha256=sha(L/'time_keymask_source_only_audit.json'),seed=20260920,bootstrap_seed=20261022,maximum_GPU_seconds=1200,stop_epoch=1792708800,authorization='User overnight eventfeedbackloop permits frozenSource reads, same8expertobservations/space fixed; no addedexpert model',rationale='Global learnedwordprojection yielded HC64positive but fixedrecheckfailure; sharedcovariance and3state pathsfailed. Test whether propagatedtargetlocation yields moretarget-specific eventsemantic readings withoutchangingFullcaption/fusion/q0. This is feasiblehypothesis, notprovedglobalheadmisidentitycause. SourceTimeDecodermaskorderaudit128allzero: excludedasexplanation,0maskrepairmodelruns.',fixed=dict(target='OriginalfulltimecompleteBoxFixedDP trajectory definesfixed1.5context pernode; sameexisting keep_region/reflected helper. ReadcachedORIGINALnormalizedFulltext+motion features; wholecaption andall65time nodes preserved; original3Dstillincludesclipcontext. No perframeRGBzoom/motionediting/newvisualinfo.',interface='ScopedprehookonEACHoriginalASA BertSelfAttention.dropout restricts alreadysoftmaxkeyprobabilitiestoROIandrenormalizes, mathematically maskedattention; no sourcefile/weight/queryinjection. Originalpre-dropout att_map clone isunmaskedandunusedasnewfeedback. HeadlearnedclasslogitsBEFOREframemean onlynewsemanticp. AllkeysNoOp returnsORIGINALinputobject; exactglobalheadp all128 anduHC64 restore.',native='Nativeq0/Source/ERef/8evidence/geometry/6layerhull/MBR/finalDP unchanged. Spatialtrajectory conditions extra wordhead only, not originalnative timequery/decoderattention. Originalsourceheadswordclasses NOTactionness.',readouts='Global,AllKeysNoOp,TargetContext,ReflectedContext,WrongTimeContext:fiveheadreads*2offsets*128=1280actualASAforwards. WrongTime uses reversed65-nodeauxtrajectory positions, notmotionfeatures; reflectedsamearea mayoverlaptargetandisnotknownfalseevent. EmptyROIframe wholevalidfallbackrecorded,noqueryveto.',loop='EXACTsealedstage21Bernoulliprototype algorithm/oppositeparity/twoiterations/fixedq0, no gain/step/covariance/dimensionchanges. TargetRandomsameu256fixedGaussianrankmatchedEXACTtargetperclasshistogram. DirectusesSAMEsecondstepS/uniformprior. ParentGlobal/Sharpenaliasesexact.'),qualification='EACHHC64/HC32/Vid32actualSource/G6t/fullvpositive,win>harm,Source shortANDlongpositive,primarybeyondParentGlobal/reflected/wrongtime/random/sharpen;allnegativecasespreserved,nocontrolpromotion',new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_time_decoder=0,new_expert=0,new_observations=0,new_optimizer=0,GT_in_inference=False,all128_seal_before_GT=True,reference='https://github.com/HengLan/TA-STVG/blob/904ad0c344eb12fe34513f97a33f93118f6d3be4/models/bert_model/bert_module.py')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();put(D/'plan.json',p);put(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','code','paper_code','provider_sources','assets']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 put(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])

def head_read(model,cap,aux,kind):
 import torch,numpy as np
 from stvg.visual_cache import transfer
 from research.native_event_target_semantic_v1.core import support_keys,masks
 from contextlib import nullcontext
 obs=[];supportstats=[];mass=[]
 with torch.no_grad(),torch.autocast('cuda',enabled=False):
  for off in cap['temporal_cache']['offsets']:
   M=transfer(off['inputs']['encoded_feature'],'cuda');h,w=off['grid_size'];n=M.shape[1];assert not off['inputs']['encoded_mask'].any(),'Actualcohortunmasked; noborrowedmaskorderingassumption';count=h*w;fm=M[-count:].permute(1,2,0).reshape(n,256,h,w).detach();ft=M[:-count].mean(1).unsqueeze(0);inputs=[];outputs=[];head=model.t_spatial_clas;handles=[head.head.register_forward_pre_hook(lambda m,a:inputs.append(a[0].detach().cpu().clone())),head.head.register_forward_hook(lambda m,a,o:outputs.append(o.detach().cpu().clone()))]
   if kind in ['Global','AllKeysNoOp']:keep=torch.ones(n,count,dtype=torch.bool);stats=[]
   else:keep,stats=masks(aux,off['offset'],n,h,w,torch.ones(n,h,w,dtype=torch.bool),kind)
   try:
    with (nullcontext([])if kind=='Global'else support_keys(head,keep.cuda()))as calls:head(fm,ft[:,:1])
   finally:
    for handle in handles:handle.remove()
   if kind!='Global':assert len(calls)==2;mass.extend(float(x['keep_mass'].mean())for x in calls)
   assert len(inputs)==len(outputs)==1;obs.append(dict(z=outputs[0][:,0].numpy(),u=inputs[0][:,0].numpy()));supportstats.extend(stats)
 n=len(cap['frame_ids']);z=np.stack([obs[j%2]['z'][j//2]for j in range(n)]).astype(np.float64);prob=np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6);u=np.stack([obs[j%2]['u'][j//2]for j in range(n)]);return prob,u,supportstats,mass

def predict():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.temporal import final_mbr
 from stvg.spatial import SpatialGeometry,evidence_dp
 from research.native_event_semantic_prototype_v1.core import loop,matched_random,update
 p=verify();setup();torch.set_num_threads(1);lock=lease();start=time.perf_counter();put(D/'state.json',dict(status='predicting_GPU',fingerprint=p['fingerprint'],**who()));assert all(sha(PROVIDER/f)==h for f,h in p['assets'].items());files={};counts=dict(ASA_head_forwards=0,new_RGB=0,new_visual_encoder=0,new_text_encoder=0,new_fusion=0,new_time_decoder=0,new_expert=0,new_observations=0,new_optimizer=0)
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds]);model.vid.cpu();model.vis_encoder.cpu();gc.collect();torch.cuda.empty_cache()
   for item in [x for x in p['items']if x['dataset']==ds]:
    verify();assert time.perf_counter()-start<p['maximum_GPU_seconds'];tick=time.perf_counter();root=FIRST if item['source']=='first'else RECHECK;e=item['entry'];assert sha(root/e['path'])==e['sha256']and sha(root/e['trace_path'])==e['trace_sha256'];cf=PROVIDER/item['capture']['path'];assert sha(cf)==item['capture']['sha256'];cap=torch.load(cf,weights_only=False,map_location='cpu');old=torch.load(root/e['trace_path'],weights_only=False,map_location='cpu');ids=cap['frame_ids'];q0=old['q']['NativeG6'];oldp=old['semantic_p']['motion']if item['source']=='first'else old['semantic_p'];geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'];probs={};us={};stats={};masses={};cost={}
    for kind in ['Global','AllKeysNoOp','TargetContext','ReflectedContext','WrongTimeContext']:
     t=time.perf_counter();probs[kind],us[kind],stats[kind],masses[kind]=head_read(model,cap,aux,kind);counts['ASA_head_forwards']+=2;cost[kind]=time.perf_counter()-t
    assert np.array_equal(probs['Global'],oldp)and np.array_equal(probs['AllKeysNoOp'],oldp)and np.array_equal(us['AllKeysNoOp'],us['Global'])
    if item['source']=='first':assert np.array_equal(us['Global'],old['head_input']['motion'])
    primary,steps=loop(probs['TargetContext'],q0);reflect,rfsteps=loop(probs['ReflectedContext'],q0);wrong,wtsteps=loop(probs['WrongTimeContext'],q0);rp=matched_random(probs['TargetContext'],us['TargetContext']);random,rrsteps=loop(rp,q0);uniform=np.where(q0>0,1.,0.);uniform/=uniform.sum();qs=dict(NativeG6=q0,TargetSemanticLoop2=primary,OneStepTarget=steps[0]['q'],DirectTargetEvidence=update(uniform,steps[1]['S']),ReflectedSemanticLoop2=reflect,WrongTimeSemanticLoop2=wrong,RandomTargetSemanticLoop2=random,ParentGlobalSemanticLoop2=old['q']['SemanticPrototypeLoop2'],SharpenOriginal=old['q']['SharpenOriginal'],AllKeysNoOp=q0.copy());iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(geom.for_interval(iv[a]))['final'].numpy()for a in ARMS}
    with np.load(root/e['path'],allow_pickle=False)as z:
     for a,b in [('NativeG6','NativeG6'),('ParentGlobalSemanticLoop2','SemanticPrototypeLoop2'),('SharpenOriginal','SharpenOriginal'),('AllKeysNoOp','NativeG6')]:assert np.array_equal(arrays[a],z[b])and iv[a]==z[b+'_interval'].tolist()
     arrays['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
    k=item['alias'];f=D/'predictions'/f'{k}.npz';assert not f.exists();f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arrays,**{a+'_interval':v for a,v in iv.items()},frame_ids=ids,positions8=cap['planned_positions']);tr=D/'traces'/f'{k}.pt';tr.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,steps=steps,semantic_p=probs,head_inputs=us,random_p=rp,support_statistics=stats,keep_mass=masses,cost=cost,GT_read=False),tr);files[k]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(tr.relative_to(D)),trace_sha256=sha(tr),seconds=time.perf_counter()-tick);put(D/'progress.json',dict(completed=len(files),total=128,counts=counts,seconds=time.perf_counter()-start,**who()));print(k,iv['NativeG6'],iv['TargetSemanticLoop2'],flush=True)
   assert all(not x.requires_grad and x.grad is None for x in model.parameters());del model;gc.collect();torch.cuda.empty_cache()
  verify();assert counts['ASA_head_forwards']==1280;put(D/'sealed.json',dict(count=128,files=files,plan_sha256=sha(D/'plan.json'),GT_read=False,counts=counts,seconds=time.perf_counter()-start));put(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:lock.close()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:put(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),**who(),partial_preserved=True));put(D/'state.json',dict(status='engineering_failed',partial_preserved=True));raise
