"""Fixed TA-only input representation comparison, no tuning/new expert."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,traceback,gc
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'tools'))
from full import check,sha,read,atomic,identity,K16,OLD,RUN as FULL
D=R/'outputs/local_visual_v1';P=R/'results/local_visual_v1';B=K16/'outputs/general_region_recheck_v3'
ARMS=['Native2D','CurrentRegion','Native3D','Fused2D','Reflected2D','WrongTime2D','NativeG6','Source']

def verify(assets=False):
 p=read(D/'plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
 for f,h in p['code'].items():assert sha(R/f)==h,f
 for f,h in p['inputs'].items():assert sha(f)==h,f
 check(assets);return p

def prepare():
 assert not(D/'plan.json').exists();parent=read(B/'plan.json');seal=read(B/'sealed.json');items=[]
 for item in parent['items']:
  e=seal['files'][item['alias']];items.append(dict(**item,entry=e))
 assert len(items)==64 and all(sum(x['dataset']==ds for x in items)==32 for ds in ['hc2','vidstg'])
 code={str(f.relative_to(R)):sha(f)for folder in ['studies/local_visual_v1','interfaces','adapters']for f in(R/folder).glob('*.py')}
 p=dict(schema='local_visual_v1',primary='Native2D',count=64,arms=ARMS,items=items,labels=parent['labels'],code=code,inputs={str(f):sha(f)for f in[B/'plan.json',B/'sealed.json',FULL/'plan.json']},
 authorization='Latest user asks actualTA-only validation whether broader original localvisual representation loses substantial accuracy; no otherbackbone/MLLM, no training/newexpert/tuning. Temporary full pause preservedHC130/Vid0, restore sameplan afterdiagnostic, no automatic candidatepromotion.',
 claim='Only change extra-feedback feature: original2Dappearance input_proj(vis_encoder) grid BEFORE crossmodal fusion. No separate motion branch/textfusion/classifier required by feedback. Original nativeSource q0 remains unchanged including its ownASA/TTS/motion. This doesnot remove these from the backbone.',
 fixed='Same exact acceptedHC32/Vid32 additionalparent inputs, alreadyEXPOSED. Original224/twooffset grid/precision/seed20260920/fullcaption/Source/E/geometry/eightobservations fixed. Same1.5fulltimeBoxFixed region, fractionalpool, otherparitycenter/L2/cosinekernel/common16/ONEq0update/originalMBR+DP. No scan.',
 controls='2x2 representations from SAME ONE Source call/query: prefusion2D(primary), prefusion3D(control), postfusion2D(control), postfusion3D(currentRegion). Reflected2Dsamearea and halfperiod wrongtime feedback controls; Source/G6 aliases. No controlpromotion.',
 outputs='Complete predictionsealALL64 before separateCPUfixedofficiallabels. Report t/fullv/fixedspacev, thresholds/boundary/length/shortlong/wins/harms/severe/parentCI andall signed. Describe magnitude anduncertainty; no invented statisticalequivalence/pass orfull-result modelselection.',
 budget=dict(max_GPU_seconds=1800,new_Source_queries=64,new_expert=0,new_observations=0,new_training=0,new_extra_feedback_model_forwards=0,seed=20260920),GT_in_inference=False,exposure='Previously exposed fixedHC32/Vid32. Bounded paired feasibility, not unseen confirmation or crossbackbone proof.')
 p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest();atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k not in ['items','labels','inputs']})
 for f in code:
  dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R/f).read_bytes())
 atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));verify(True);print(p['fingerprint'])

def predict():
 import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 from stvg.native import frozen_native_forward
 from stvg.global6 import read_logits,posterior
 from stvg.temporal import final_mbr
 from stvg.prototype import one_step,inclusive_scores,update
 from stvg.spatial import SpatialGeometry,evidence_dp
 from stvg.representation import extract,weights,pool
 p=verify(True);setup();torch.set_num_threads(1);gpu=lease();start=time.perf_counter();assert not(D/'predictions').exists();atomic(D/'state.json',dict(status='predicting',fingerprint=p['fingerprint'],**identity()));files={};counts=dict(Source_queries=0,new_expert=0,new_observations=0,new_training=0,extra_feedback_model_forwards=0);parser=QuerySubjectParser(R/'checkpoints/stanza')
 try:
  for ds in ['hc2','vidstg']:
   model=load_source(read(R/'configs/default.json')['datasets'][ds])
   for item in [x for x in p['items']if x['dataset']==ds]:
    assert time.perf_counter()-start<p['budget']['max_GPU_seconds'];tick=time.perf_counter();torch.manual_seed(20260920);np.random.seed(20260920);e=item['entry'];key=item['alias'];row=item['row']
    for name,h in [('capture_path','capture_sha256'),('trace_path','trace_sha256'),('path','sha256')]:assert sha(B/e[name])==e[h]
    cap=torch.load(B/e['capture_path'],map_location='cpu',weights_only=False);parent=torch.load(B/e['trace_path'],map_location='cpu',weights_only=False);ids=cap['frame_ids'];g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);aux=evidence_dp(g.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
    frames,physical=decode(row['metadata'],ds);assert physical==ids;rr=dict(input=row['metadata'],frame_ids=ids,parses={'subject':parser(row['metadata']['caption'])['subject']});observed=[]
    def preserve(module,args,kw):
     assert not args;v=kw['videos'];observed.append(dict(appearance=v.tensors.detach().cpu().clone(),motion=kw['vid'].detach().cpu().clone(),valid=(~v.mask).detach().cpu().clone()))
    hook=model.ground_encoder.register_forward_pre_hook(preserve,with_kwargs=True)
    try:
     with torch.no_grad():batch,recs,base=frozen_native_forward(model.cuda(),frames,rr);logs,_=read_logits(model.temp_embed,[x for x in base.temporal_inputs],[x for x in base.zero['logits']])
    finally:hook.remove()
    assert len(observed)==2 and np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source']);q0=posterior(recs,logs,ids);assert np.array_equal(q0,parent['q']['NativeG6'])
    for a,b in zip(logs,cap['logits']):assert torch.equal(a,b),'Full6logits bitwise'
    for off,saved in enumerate(cap['temporal_cache']['offsets']):
     assert torch.equal(base.time_inputs[off]['encoded_feature'].cpu(),saved['inputs']['encoded_feature'])
    features={k:np.empty((len(ids),256))for k in ['Native2D','Native3D','Fused2D','Reflected2D']};poolstats=[]
    for off,o in enumerate(observed):
     x=o['appearance'].permute(0,2,3,1).numpy();y=o['motion'].permute(0,2,3,1).numpy();n,h,w,c=x.shape;assert c==256 and y.shape==x.shape
     spatial=base.spatial_inputs[off]['encoded_feature'][:h*w].permute(1,0,2).cpu().numpy().reshape(n,h,w,c)
     for j in range(n):
      t=off+2*j;support,fallback=weights(aux[t],h,w,o['valid'][j].numpy());reflected=support[::-1,::-1].copy()
      for name,value,weight in [('Native2D',x[j],support),('Native3D',y[j],support),('Fused2D',spatial[j],support),('Reflected2D',x[j],reflected)]:features[name][t]=pool(value,weight)
      poolstats.append(dict(frame=t,fallback=bool(fallback),cells=h*w,area=float(support.sum())))
    features['CurrentRegion'],_=extract(cap,aux);qs={};traces={}
    for a,z in features.items():qs[a],traces[a]=one_step(z,q0)
    assert np.array_equal(qs['CurrentRegion'],parent['q']['RegionKernel']);L=traces['Native2D']['L'].copy()
    for parity in [0,1]:ii=np.arange(parity,len(ids),2);L[ii]=np.roll(L[ii],len(ii)//2)
    qs['WrongTime2D']=update(q0,inclusive_scores(L));qs['NativeG6']=q0;ivs={a:final_mbr(ids,q)['interval']for a,q in qs.items()};arrays={a:evidence_dp(g.for_interval(ivs[a]))['final'].numpy()for a in qs}
    with np.load(B/e['path'],allow_pickle=False)as z:
     for a,b in [('CurrentRegion','RegionKernel'),('NativeG6','NativeG6')]:assert np.array_equal(arrays[a],z[b])and ivs[a]==z[b+'_interval'].tolist()
     arrays['Source']=z['Source'].copy();ivs['Source']=z['Source_interval'].tolist();arrays['FixedSpace']=z['NativeG6'].copy()
    out=D/'predictions'/f'{key}.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays,**{a+'_interval':v for a,v in ivs.items()},frame_ids=ids,positions8=cap['positions8'])
    trace=D/'traces'/f'{key}.pt';trace.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(q=qs,features=features,poolstats=poolstats,feedback=traces,GT_read=False),trace)
    files[key]=dict(path=str(out.relative_to(D)),sha256=sha(out),trace_path=str(trace.relative_to(D)),trace_sha256=sha(trace),Source_native_memory_logits_q_baseline_bitwise=True,seconds=time.perf_counter()-tick);counts['Source_queries']+=1;atomic(D/'progress.json',dict(completed=len(files),total=64,counts=counts,seconds=time.perf_counter()-start,**identity()));print(key,round(files[key]['seconds'],3),flush=True)
    del batch,base,frames,cap,parent,g,observed,arrays,features,qs;gc.collect();torch.cuda.empty_cache()
   del model;gc.collect();torch.cuda.empty_cache()
  assert len(files)==64;verify();atomic(D/'sealed.json',dict(count=64,files=files,counts=counts,GT_read=False,plan_sha256=sha(D/'plan.json'),seconds=time.perf_counter()-start));atomic(D/'state.json',dict(status='sealed',fingerprint=p['fingerprint']))
 finally:gpu.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  atomic(D/f'failure-{time.time_ns()}.json',dict(traceback=traceback.format_exc(),partial_preserved=True,**identity()));atomic(D/'state.json',dict(status='engineering_failed',partial_preserved=True,**identity()));raise
