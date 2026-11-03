"""Three strategies: cache-only spatial processing timing, no new accuracy run."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse,gc,itertools,platform,subprocess,csv
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(R/'studies/spatial_temporal_six_v1'))
import cpu as old
import matching
import numpy as np,torch
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry,evidence_dp
D=R/'outputs/spatial_comparison_focus_v1';P=R/'results/spatial_comparison_focus_v1';PREV=old.D/'matching_pilot'
read=old.read;sha=old.sha;atomic=old.atomic;identity=old.identity;digest=old.digest
ARMS=['PureExpert','Matching','Current'];DATASETS=['hc2','vidstg']
def settings():
 torch.set_num_threads(1)
 import cv2
 cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
def prepare():
 assert not(D/'plan.json').exists();inputs=[D/'intent.json',PREV/'plan.json',PREV/'selection.json',PREV/'raw_inventory.json',old.D/'source_validation/calibration/root_acceptance.json',R/'configs/default.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 for ds in DATASETS:inputs += [PREV/ds/n for n in ['sealed.json','metrics.json','reporting_receipt.json','root_receipt.json']]
 p=dict(schema='three_spatial_strategy_cache128_actual_CPU_timing_v1',methods=ARMS,counts={ds:128 for ds in DATASETS},repeats=3,total_measured_spatial_calls=2304,warmup_calls=6,
  selection='Same existing128queries/128distinctparents/domain. All existing accuracy predictions/metrics independently audited and reused exactly, no new official scores/GT/model/Source/expert/optimizer. Not fulltarget or an unseen panel.',
  common_inputs='Same original8expert masks/boxes/positions and complete Source postfusion feature grid/valid mask/Source boxes and Current event interval. All resident CPU inputs/parameters and references loaded before timer. Matching settings fixed only from source validation (.05,K10,context20). Pure need not use available features.',
  clock='time.perf_counter_ns around a complete actual spatial function call producing an owned CPU box array. Current includes shared E initialization at originalgamma/calibration, SourceEvidence construction, candidate geometry/appearance/links, bank building and final intervalDP; no firstauxDP/time-prototype branch. Matching includes feature normalization, two-direction propagation and native-resolution mask-to-box readout. Pure includes interpolation/nearestexpert tails/exactanchors. Disk IO/experimental SHA/GT/reference equality/array digest/resource guards/input preparation/cleanup outside measured window. Original intrinsic eight-box initialization hash/checks remain inside the unchanged function. These are ready-input CPU spatial clocks, not endtoend deployment or saved components subtraction.',
  ordering='Each query loads once outside clocks, runs3repeats. MetadataSHA bit chooses forward/reverse base arm order; rotate by repeat so every arm appears each order position once. Warmup one firstquery per arm/domain outside formalcalls. No shared normalized-feature/geometry/output cache across calls.',
  resources='singleownedCPUjob, affinity14-15 nice19 idleIO, Torch/BLAS/OpenCV thread1, CUDAhidden; RAM8GiB/disk80GiB/private160GiB/72h. No GPU/model/oldowner signals. No heavy own concurrent compute/IO during measured stage.',
  statistics='Per query first average3repeat, then equal-query means,p50,p90,paired10000parentbootstrap for timingdifference and ratio-of-means seed20261103. All256distinctparents share same8evidence and fixedinterval within query. PointwiseCI. Actual2304call costs/hardware/processstart/threads/warmup separate. No accuracy/interval/seed/setting retuning.',
  caveat='Current unchanged, observed experts are soft candidates rather than fixed output; Pure/Matching preserve expert anchors. Existing observed-frame implementation-contract audit retained; this is as-implemented comparison, no hard-anchor-method correctness claim. Adapter functional/sourcevalidated but weaker than interpolation, not strong learnedVideoWalk reproduction.',
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'controller.py',S/'report.py',R/'studies/spatial_temporal_six_v1/matching.py',R/'studies/pure_expert_interpolation_v1/run.py',R/'stvg/initialization.py',R/'stvg/consensus.py',R/'stvg/predictor.py',R/'stvg/evidence.py',R/'stvg/spatial.py',R/'stvg/_spatial_math.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',p);print(p['fingerprint'],flush=True)
def check():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def rows(ds):return old.fixed(PREV/'selection.json')[ds]
def load(ds,row):
 key=row['id'];entry=old.fixed(PREV/'raw_inventory.json')[ds][key];assert sha(entry['path'])==entry['sha256'];raw=torch.load(entry['path'],map_location='cpu',weights_only=False,mmap=True);cap,binding=old.SEL.suite().load_parent(ds,row);g=SpatialGeometry.from_cached_geometry(cap['geometry']);e=old.fixed(PREV/ds/'sealed.json')['files'][key];assert sha(PREV/e['path'])==e['sha256']and sha(PREV/e['receipt_path'])==e['receipt_sha256']
 with np.load(PREV/e['path'])as z:
  refs={a:z['PureMatched'if a=='PureExpert'else a].copy()for a in ARMS};ids=z['frame_ids'].copy();pos=z['positions8'].copy();iv=z['Current_interval'].copy()
  assert np.array_equal(iv,z['Matching_interval'])and np.array_equal(iv,z['PureMatched_interval'])
 assert list(ids)==raw['frame_ids']==cap['frame_ids']and list(pos)==cap['positions8'];cfg=read(R/'configs/default.json')['datasets'][ds];selected=read(PREV/'plan.json')['source_config'][ds]
 v=dict(ds=ds,ids=ids,pos=pos,Source=np.asarray(cap['Source']).copy(),features=raw['spatial_features'].clone(),valid=raw['spatial_valid'].clone(),seeds=g.side_maps[pos,0].clone(),expert=np.asarray(g.native_boxes).copy(),shape=(g.height,g.width),image_size=(row['metadata']['height'],row['metadata']['width']),interval=iv,cfg=cfg,matching_cfg=selected)
 assert v['features'].shape==(len(ids),g.height*g.width,256)and v['valid'].shape==v['features'].shape[:2]
 return v,refs,dict(raw_sha256=entry['sha256'],reference_sha256=e['sha256'],parent=binding)
def apply(arm,v):
 if arm=='PureExpert':return old.pure.pure_interpolation(v['ids'],v['pos'],v['expert'])
 if arm=='Matching':return matching.propagate(v['features'],v['valid'],v['pos'],v['seeds'],v['expert'],v['shape'],v['image_size'],**v['matching_cfg'])[0]
 assert arm=='Current';shared,_=initialize(v['Source'],v['pos'],v['expert'],v['cfg']['E_calibration']);E=output_writeback(v['Source'],shared,v['cfg']['initialization_gamma']);ev=SourceEvidence.from_arrays(v['features'],v['valid'],v['ids'],v['pos'],v['seeds']);geo=SpatialGeometry(ev,E,v['expert']);return evidence_dp(geo.for_interval(v['interval']))['final'].numpy().copy()
def array_digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def input_digest(v):
 out={}
 for k in ['ids','pos','Source','features','valid','seeds','expert','interval']:
  x=v[k].numpy()if torch.is_tensor(v[k])else np.asarray(v[k]);out[k]=array_digest(x)
 return digest(out)
def preflight():
 plan=check();settings();old.parent.infer_guard();cases=[]
 for ds in DATASETS:
  for row in rows(ds)[:2]:
   v,refs,p=load(ds,row);before=input_digest(v)
   with torch.no_grad():
    for a in ARMS:
     x=apply(a,v);y=apply(a,v);assert np.array_equal(x,refs[a])and np.array_equal(x,y),(ds,a)
   assert input_digest(v)==before;cases.append(dict(dataset=ds,all3strategies_actual_replay_double_exact=True,input_immutable=True,reference_sha256=p['reference_sha256']))
 atomic(D/'preflight.json',dict(status='passed_four_actual_cache_cases_all3_spatial_functions_double_exact',cases=cases,worker=identity(),plan_fingerprint=plan['fingerprint'],new_Source=0,new_expert=0,new_GT=0));atomic(P/'preflight.json',read(D/'preflight.json'));print('three-arm spatial timing cache preflight passed',flush=True)
def hardware():
 import cv2
 cpus=[line.split(':',1)[1].strip()for line in Path('/proc/cpuinfo').read_text().splitlines()if line.startswith('model name')];return dict(CPU=cpus[0],platform=platform.system(),Python=platform.python_version(),Torch=torch.__version__,NumPy=np.__version__,OpenCV=cv2.__version__,torch_threads=torch.get_num_threads(),cv2_threads=cv2.getNumThreads(),affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),GPU_used=False)
def measure():
 plan=check();assert read(D/'preflight.json')['status']=='passed_four_actual_cache_cases_all3_spatial_functions_double_exact';assert not(D/'measure_worker.json').exists();settings();old.parent.infer_guard();start=time.perf_counter();who=identity();atomic(D/'measure_worker.json',who);items=[];warm=[];meta=hardware()
 try:
  with torch.no_grad():
   for ds in DATASETS:
    for qi,row in enumerate(rows(ds),1):
     old.parent.resources(start);assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30;v,refs,binding=load(ds,row);before=input_digest(v)
     if qi==1:
      for arm in ARMS:
       t=time.perf_counter_ns();out=apply(arm,v);ns=time.perf_counter_ns()-t;assert np.array_equal(out,refs[arm]);warm.append(dict(dataset=ds,arm=arm,elapsed_ns=ns,excluded=True));del out
     base=ARMS.copy()
     if hashlib.sha256(f'20261103|{ds}|{row["id"]}'.encode()).digest()[0]%2:base.reverse()
     for repeat in range(3):
      order=base[repeat:]+base[:repeat]
      for oi,arm in enumerate(order):
       gc.collect();clock=time.perf_counter_ns();out=apply(arm,v);elapsed=time.perf_counter_ns()-clock
       assert elapsed>0 and np.array_equal(out,refs[arm]),(ds,qi,repeat,arm,'no fallback/retry')
       item=dict(dataset=ds,query=qi,repeat=repeat+1,arm=arm,order_position=oi+1,elapsed_ns=elapsed,elapsed_seconds=elapsed/1e9,output_digest=array_digest(out),reference_array_digest=array_digest(refs[arm]),reference_prediction_sha256=binding['reference_sha256'],input_digest=before,all_output_arrays_exact=True,worker=who);items.append(item);del out
     assert input_digest(v)==before;atomic(D/'progress.json',dict(completed_calls=len(items),total_calls=2304,dataset=ds,query=qi,GT_read=False,worker=who));atomic(D/'calls_partial.json',dict(items=items,warmup=warm,hardware=meta));del v,refs;gc.collect()
   assert len(items)==2304 and len(warm)==6
  atomic(D/'measurements.json',dict(items=items,warmup=warm,hardware=meta,plan_fingerprint=plan['fingerprint']));atomic(D/'sealed.json',dict(status='all2304_spatial_timing_calls_exact',calls=2304,warmups=6,measurement_sha256=sha(D/'measurements.json'),worker=who,elapsed_experiment_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_official_scores=0));print('all2304 actual CPU spatial calls sealed exact',flush=True)
 except BaseException:
  import traceback
  atomic(D/'failure.json',dict(error=traceback.format_exc(),worker=who,completed_calls=len(items),existing_inputs_and_all_calls_preserved=True));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','measure']);globals()[ap.parse_args().stage]()
