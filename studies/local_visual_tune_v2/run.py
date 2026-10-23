"""Three fixed strengths, exposeddev64 and prospectively frozen parent-separated64."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,hashlib,gc,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(Path(__file__).parent))
from full import check,sha,read,atomic,identity,K16,OLD,RUN as FULL
from core import apply,VALUES
D=R/'outputs/local_visual_tune_v2';P=R/'results/local_visual_tune_v2';DEV=R/'outputs/local_visual_v2';STAGE=K16/'outputs/general_region_v1';ACQ=OLD/'outputs/target_semantic128_collect_v3'

def verify(assets=False):
 p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['inputs'].items():assert sha(f)==h,f
 check(assets);return p

def prepare():
 assert not(D/'plan.json').exists();dev=read(DEV/'plan.json');sp=read(STAGE/'plan.json');ss=read(STAGE/'sealed.json');cs=read(ACQ/'sealed.json');selected=[]
 for ds in ['hc2','vidstg']:
  blocked={str(i['parent'])for i in dev['items']if i['dataset']==ds};pool=[i for i in sp['items']if i['dataset']==ds and str(i['parent'])not in blocked];assert len(pool)==128
  rank=lambda i:hashlib.sha256((ds+'|'+str(i['parent'])+'|native2d-kappa-confirm-v1|20261023').encode()).hexdigest()
  chosen=sorted(pool,key=rank)[:32];assert len({str(x['parent'])for x in chosen})==32
  for item in chosen:
   selected.append(dict(**item,entry=ss['files'][item['alias']],capture_entry=cs['files'][item['alias']]))
 code={str(f.relative_to(R)):sha(f)for f in Path(__file__).parent.glob('*.py')}
 p=dict(schema='local_visual_tune_v2',values=list(VALUES),shared_parameter=True,dev_items=dev['items'],confirm_items=selected,labels=dev['labels'],count_dev=64,count_confirm=64,code=code,inputs={str(f):sha(f)for f in [DEV/'plan.json',DEV/'sealed.json',DEV/'score_receipt.json',R/'results/local_visual_v2/report.json',STAGE/'plan.json',STAGE/'sealed.json',ACQ/'sealed.json',FULL/'plan.json']},
 authorization='Latest user asks boundedparameter adjustment with overfit protection. ONLYkappa4/16/64, no separate domainchoice, no additionalbackbone/expert/observation/training; one-time prospective parent-separated confirmation before any adoptionclaim.',
 fixed='Native2D beforefusion/currentq0/1.5fulltimeBoxFixed/otherparitycenterL2/nonnegativecosinekernel/ONEupdate/originalMBR+DP/eightobservations unchanged. Only common-mean shrinkage kappa is varied. No beta/layer/iterations/template/ROI scan or stackedcorrection.',
 dev_selection='CurrentHC32/Vid32 are EXPOSED dev. Eachcandidate must have mean t andcompletev >=K16-.25pp in EACHdirection, severe>5pp t/v vsG6 no greaterthanK16 in EACHdirection. Amongeligible choose equal-direction meancompletev gainvsK16; tiepref16,64,4. K16 fallback isnot claimed repair. ONEsharedkappa, no query/domain GT selector.',
 confirmation='Source-only SHA chooses32parents/direction FROMexpanded128, disjoint devparents, before devscore. ONLYselectedkappa andK16 plusCurrent/Source/G6. All64seal thenCPUlabels; no reselection onconfirm. Engineering transfer screen: t/v >=K16-.25pp andsevereG6 no greaterthanK16 EACHdirection; macrocompletev gain>=0. Ifchangedparameter has no positive confirmationmacro gain orfails screen, report notsupported andSTOP, retainbaseline16; do not trysecondwinner or expandgrid. Notstatisticalequivalence.',
 exposure='All data have historical originalfull/othermethod exposure. Newconfirm parents are disjoint from Native2Ddev and Native2D candidateoutputs not read before freezing; candidate-specific prospective recheck, NOT untouchedbenchmark/independentunexposedtest.',
 budget=dict(new_Source_queries=64,max_GPU_seconds=1800,new_expert=0,new_observations=0,new_training=0,seed=20260920),GT_in_inference=False,original_full_plan_unchanged=True,no_automatic_main_method_replacement=True)
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k not in ['dev_items','confirm_items','labels','inputs']})
 for f in code:
  q=D/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((R/f).read_bytes())
 atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));verify(True);print(p['fingerprint'])

def dev_predict():
 import torch,numpy as np
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.temporal import final_mbr
 p=verify();torch.set_num_threads(1);assert not(D/'dev/sealed.json').exists();s=read(DEV/'sealed.json');b=K16/'outputs/general_region_recheck_v3';files={};start=time.perf_counter()
 for item in p['dev_items']:
  key=item['alias'];e=s['files'][key];raw=item['entry'];assert sha(DEV/e['trace_path'])==e['trace_sha256'];tr=torch.load(DEV/e['trace_path'],map_location='cpu',weights_only=False);assert sha(b/raw['capture_path'])==raw['capture_sha256'];cap=torch.load(b/raw['capture_path'],map_location='cpu',weights_only=False);g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);ids=cap['frame_ids'];qs={};arr={};iv={}
  for k in VALUES:
   a='K'+str(k);qs[a],_=apply(tr['features']['Native2D'],tr['q']['NativeG6'],k);iv[a]=final_mbr(ids,qs[a])['interval'];arr[a]=evidence_dp(g.for_interval(iv[a]))['final'].numpy()
  assert np.array_equal(qs['K16'],tr['q']['Native2D'])
  with np.load(DEV/e['path'],allow_pickle=False)as z:
   assert np.array_equal(arr['K16'],z['Native2D'])and iv['K16']==z['Native2D_interval'].tolist()
   for a in ['Source','CurrentRegion','NativeG6']:arr[a]=z[a].copy();iv[a]=z[a+'_interval'].tolist()
  dest=D/'dev/predictions'/f'{key}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arr,**{a+'_interval':v for a,v in iv.items()},positions8=cap['positions8'],frame_ids=ids);files[key]=dict(path=str(dest.relative_to(D)),sha256=sha(dest));del tr,cap,g
 atomic(D/'dev/sealed.json',dict(count=64,files=files,GT_read=False,plan_sha256=sha(D/'plan.json'),seconds=time.perf_counter()-start));atomic(D/'state.json',dict(status='dev_sealed',fingerprint=p['fingerprint']));print('Dev64*3 sealed')

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
 p=verify(True);frozen=read(D/'frozen_config.json');k=frozen['kappa'];assert k in VALUES and frozen['plan_sha256']==sha(D/'plan.json');setup();torch.set_num_threads(1);lock=lease();assert not(D/'confirm/predictions').exists();files={};begin=time.perf_counter();parser=QuerySubjectParser(R/'checkpoints/stanza');atomic(D/'confirm_worker.json',identity());atomic(D/'state.json',dict(status='confirm_predicting',fingerprint=p['fingerprint'],kappa=k))
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
    current,st=extract(cap,aux);qcurrent,_=one_step(current,q0);assert np.array_equal(qcurrent,prior['q']['RegionKernel']);q16,_=apply(features,q0,16);qnew=q16 if k==16 else apply(features,q0,k)[0];qs=dict(Selected=qnew,K16=q16,CurrentRegion=qcurrent,NativeG6=q0);iv={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arr={a:evidence_dp(g.for_interval(iv[a]))['final'].numpy()for a in qs}
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
