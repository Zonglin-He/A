"""Risk-prioritized finite Native2D calibration; stages sealed before CPU labels."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,hashlib,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(Path(__file__).parent))
from full import check,sha,read,atomic,identity,K16,OLD,RUN as FULL
from core import apply,VALUES,BETAS,CONFIGS
D=R/'outputs/native2d_risk_v2';P=R/'results/native2d_risk_v2';DEV=R/'outputs/local_visual_v2';TUNE=R/'outputs/local_visual_tune_v2';STAGE=K16/'outputs/general_region_v1';ACQ=OLD/'outputs/target_semantic128_collect_v3'

def verify(assets=False):
 p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['inputs'].items():assert sha(f)==h,f
 check(assets);return p

def prepare():
 assert not(D/'plan.json').exists();old=read(DEV/'plan.json');t=read(TUNE/'plan.json');sp=read(STAGE/'plan.json');ss=read(STAGE/'sealed.json');cs=read(ACQ/'sealed.json');dev=[];confirm=[];s1=read(DEV/'sealed.json');s2=read(TUNE/'confirm/sealed.json');previous=R/'outputs/native2d_risk_v1';pp=read(previous/'plan.json');s3=read(previous/'confirm/sealed.json')
 for j,items in enumerate([old['items'],t['confirm_items'],pp['confirm_items']]):
  for i in items:
   e=([s1,s2,s3][j])['files'][i['alias']]
   if j==0:cap=K16/'outputs/general_region_recheck_v3'/i['entry']['capture_path']
   else:cap=ACQ/i['capture_entry']['capture_path']
   dev.append(dict(**{k:i[k]for k in ['alias','dataset','parent','row']},trace_path=str([DEV,TUNE,previous][j]/e['trace_path']),trace_sha256=e['trace_sha256'],reference_path=str([DEV,TUNE,previous][j]/e['path']),reference_sha256=e['sha256'],capture_path=str(cap),capture_sha256=sha(cap),old_arm='Native2D'if j==0 else'K16'))
 for ds in ['hc2','vidstg']:
  blocked={str(i['parent'])for i in dev if i['dataset']==ds};pool=[i for i in sp['items']if i['dataset']==ds and str(i['parent'])not in blocked];assert len(pool)==64
  chosen=sorted(pool,key=lambda i:hashlib.sha256((ds+'|'+str(i['parent'])+'|native2d-risk-confirm-v2|20261023').encode()).hexdigest())[:32]
  assert not(blocked&{str(i['parent'])for i in chosen})
  for i in chosen:confirm.append(dict(**i,entry=ss['files'][i['alias']],capture_entry=cs['files'][i['alias']]))
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 p=dict(schema='native2d_risk_v2',configs=CONFIGS,dev_items=dev,confirm_items=confirm,labels=t['labels'],code=code,inputs={str(f):sha(f)for f in [DEV/'plan.json',DEV/'sealed.json',TUNE/'plan.json',TUNE/'confirm/sealed.json',STAGE/'plan.json',STAGE/'sealed.json',ACQ/'sealed.json',FULL/'plan.json',previous/'plan.json',previous/'confirm/sealed.json',R/'results/native2d_risk_v1/confirm_report.json']},
 authorization='Latest user explicitly requests Native2D tuning to reduce severely harmed sample fraction rather than maximize mean, promote Native2D and run NEWfull; old full pausedHC401 with immutable bindings. No experts/observations/training/newbackbone.',
 fixed='prefusion projected2D/input/time/q0/1.5fulltimeBoxFixed/centerL2/cosinekernel/ONEupdate/originalMBR/DP/evidence8/seed20260920',
 budget=dict(dev_queries=192,candidates=6,confirm_queries=64,max_GPU_seconds=1800,new_expert=0,new_observations=0,new_training=0),
 selection='Dev96parents/direction exposed, all1152candidateoutputs seal before CPUlabels. Risk query= t OR completev drops>5pp vsG6, counted ONCE. Eligible: EACHdirection mean t andv >=G6-.1pp, completev>=Native2D16-.5pp and meancompletev>Source. Minimize lexicographic(maxdirection severe fraction,mean severe fraction,macro worst10percent mean of max(0,-delta_t_G6,-delta_v_G6),negative macrocompletev); tie prefer smaller deviation from16/1. ONE sharedpair. If noneeligible use16/1 as disclosedfallback, no fake pass.',
 confirmation='Source-onlySHA fixes new32parents/direction BEFOREdevscore, ZEROparentoverlapwithall192dev. Historical originalfull/othermethod exposure disclosed. Freeze selectedpair then ONEconfirm64 seal beforeCPUlabels. Accept risk-reduction claim only if union severe count<=Native2D16 EACHdirection and lower acrossboth plus EACH t/v>=Native2D16-.5pp. No reselection. Deploy selectedpair if confirmationguardpasses (even equalrisk, marked noimprovement); otherwise predeclaredfallbackNative2D16/1. User explicitly selectsNative2D asengineeringmain; scientificfailures preserved, notpermissiontopicknewconfig.',
 exposure='Previous v1 failedmeanprotection with K64beta.25: HCcompletev-2.176pp despite severe5->0 and Vid3->1. That confirmation is NOW EXPOSED development, never treated as validation again. Same sixconfigs, no gridexpansion; one FINAL extra32parents/direction fixed before this stage score. Previously96/direction Native2D outcomes exposed; remaining Source/othermethod outcomes historically exposed. Newcandidate-specific fixedrecheck NOTuntouchedtest. No full-result selection.',GT_in_inference=False)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k not in ['dev_items','confirm_items','labels','inputs']})
 for f in code:q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));verify(True);print(p['fingerprint'])

def dev_predict():
 import torch,numpy as np
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.temporal import final_mbr
 p=verify();torch.set_num_threads(1);assert not(D/'dev/sealed.json').exists();files={};start=time.perf_counter()
 for i in p['dev_items']:
  key=i['alias'];assert sha(i['trace_path'])==i['trace_sha256']and sha(i['capture_path'])==i['capture_sha256'];tr=torch.load(i['trace_path'],weights_only=False,map_location='cpu');cap=torch.load(i['capture_path'],weights_only=False,map_location='cpu');z=tr['features']['Native2D']if isinstance(tr['features'],dict)else tr['features'];q0=tr['q']['NativeG6'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];arr={};iv={};posterior={}
  for a,(k,b)in CONFIGS.items():
   q,_=apply(z,q0,k,b);posterior[a]=q;iv[a]=final_mbr(ids,q)['interval'];arr[a]=evidence_dp(g.for_interval(iv[a]))['final'].numpy()
  refq=tr['q'].get(i['old_arm'],tr['q'].get('Selected'));assert np.array_equal(posterior['K16_B100'],refq)
  assert sha(i['reference_path'])==i['reference_sha256']
  with np.load(i['reference_path'],allow_pickle=False)as ref:
   assert np.array_equal(arr['K16_B100'],ref[i['old_arm']])and iv['K16_B100']==ref[i['old_arm']+'_interval'].tolist()
   for a in ['Source','CurrentRegion','NativeG6']:arr[a]=ref[a].copy();iv[a]=ref[a+'_interval'].tolist()
  f=D/'dev/predictions'/f'{key}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,**arr,**{a+'_interval':v for a,v in iv.items()},positions8=cap['positions8'],frame_ids=ids);trace=D/'dev/traces'/f'{key}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(features=z,q=posterior,q0=q0),trace);files[key]=dict(path=str(f.relative_to(D)),sha256=sha(f),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace));print(key,flush=True)
 atomic(D/'dev/sealed.json',dict(count=192,files=files,GT_read=False,seconds=time.perf_counter()-start));atomic(D/'state.json',dict(status='dev_sealed'))
def confirm_predict():
 import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 from stvg.native import frozen_native_forward
 from stvg.global6 import read_logits,posterior
 from stvg.temporal import final_mbr
 from stvg.prototype import one_step
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.representation import weights,pool,extract
 p=verify(True);frozen=read(D/'frozen_config.json');k,beta=frozen['kappa'],frozen['beta'];assert k in VALUES and beta in BETAS and frozen['plan_sha256']==sha(D/'plan.json');setup();torch.set_num_threads(1);lock=lease();assert not(D/'confirm/predictions').exists();files={};begin=time.perf_counter();parser=QuerySubjectParser(R/'checkpoints/stanza');atomic(D/'confirm_worker.json',identity());atomic(D/'state.json',dict(status='confirm_predicting',fingerprint=p['fingerprint'],kappa=k))
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds])
   for item in [x for x in p['confirm_items']if x['dataset']==ds]:
    assert time.perf_counter()-begin<p['budget']['max_GPU_seconds'];tick=time.perf_counter();row=item['row'];key=item['alias'];e=item['capture_entry'];parent=item['entry'];torch.manual_seed(20260920);np.random.seed(20260920)
    assert sha(ACQ/e['capture_path'])==e['capture_sha256']and sha(STAGE/parent['trace_path'])==parent['trace_sha256']and sha(STAGE/parent['path'])==parent['sha256'];cap=torch.load(ACQ/e['capture_path'],map_location='cpu',weights_only=False);prior=torch.load(STAGE/parent['trace_path'],map_location='cpu',weights_only=False);g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy();frames,physical=decode(row['metadata'],ds);assert ids==physical;observed=[]
    def keep(m,a,kw):
     assert not a;v=kw['videos'];observed.append((v.tensors.detach().cpu().clone(),(~v.mask).cpu().clone()))
    hook=model.ground_encoder.register_forward_pre_hook(keep,with_kwargs=True)
    try:
     with torch.no_grad():batch,recs,base=frozen_native_forward(model.cuda(),frames,dict(input=row['metadata'],frame_ids=ids,parses={'subject':parser(row['metadata']['caption'])['subject']}));logs,_=read_logits(model.temp_embed,base.temporal_inputs,[x.detach().cpu()for x in base.zero['logits']])
    finally:hook.remove()
    assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);q0=posterior(recs,logs,ids);assert np.array_equal(q0,prior['q']['NativeG6'])
    for a,b in zip(logs,cap['logits']):assert torch.equal(a,b)
    features=np.empty((len(ids),256))
    for off,(v,valid)in enumerate(observed):
     x=v.permute(0,2,3,1).numpy();n,h,w,c=x.shape
     for j in range(n):
      t=off+2*j;support,_=weights(aux[t],h,w,valid[j].numpy());features[t]=pool(x[j],support)
    current,st=extract(cap,aux);qcurrent,_=one_step(current,q0);assert np.array_equal(qcurrent,prior['q']['RegionKernel']);q16,_=apply(features,q0,16);qnew=q16 if (k,beta)==(16,1.) else apply(features,q0,k,beta)[0];qs=dict(Selected=qnew,K16=q16,CurrentRegion=qcurrent,NativeG6=q0);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arr={a:evidence_dp(g.for_interval(iv[a]))['final'].numpy()for a in qs}
    with np.load(STAGE/parent['path'],allow_pickle=False)as z:
     for a,b in [('CurrentRegion','RegionKernel'),('NativeG6','NativeG6')]:assert np.array_equal(arr[a],z[b])and iv[a]==z[b+'_interval'].tolist()
     arr['Source']=z['Source'].copy();iv['Source']=z['Source_interval'].tolist()
    dest=D/'confirm/predictions'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arr,**{a+'_interval':v for a,v in iv.items()},positions8=cap['positions8'],frame_ids=ids);trace=D/'confirm/traces'/f'{key}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,features=features,GT_read=False),trace);files[key]=dict(path=str(dest.relative_to(D)),sha256=sha(dest),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),Source_G6_CurrentRegion_bitwise=True,seconds=time.perf_counter()-tick);atomic(D/'confirm_progress.json',dict(completed=len(files),total=64,seconds=time.perf_counter()-begin,**identity()));print(key,round(files[key]['seconds'],3),flush=True);del cap,prior,g,frames,batch,base,observed,arr;gc.collect();torch.cuda.empty_cache()
   del model;gc.collect();torch.cuda.empty_cache()
  verify();atomic(D/'confirm/sealed.json',dict(count=64,files=files,GT_read=False,Source_queries=64,new_expert=0,new_training=0,frozen_config_sha256=sha(D/'frozen_config.json'),seconds=time.perf_counter()-begin));atomic(D/'state.json',dict(status='confirm_sealed',fingerprint=p['fingerprint'],kappa=k))
 finally:lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:atomic(D/f'failure-{time.time_ns()}.json',dict(error=traceback.format_exc(),partial_preserved=True,**identity()));raise
