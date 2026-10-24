"""Real uncached query timing, currentMAP/nativeSource, original offload schedule."""
from common import *
import argparse,gc,traceback
G=D/'deployment'
def prepare():
 p=freeze('deployment',dict(cohort='Prior fixed16parents/queries/domain;3pairedrepeats, alternating order; no outcome-based sampling.',arms=['Source','FullMAP'],warmup='One call/arm/domain excluded; source/expertcheckpoint/startup recorded separately; cache_dirNone for residentexpert.',input='Original native224FP32 twooffsets and SAME8 positions/teacher. No interquery visual/RGB/capture/teacher cache. Source weights retained with original CPU/GPU offload/transfers.',time='Wall from RGBdecode/parse to final output plus cleanup, GPU synchronization; scalarverificationIO outside timer. Subcomponents nested as declared, never added to total.',verification='Every result bitwise its saved nativeSource/currentFullMAP.0GT. Any failure preservespartial/attempt/output and doesnotretry.'),[Path(__file__),S/'live_map.py',S/'common.py',R/'stvg/predictor.py',R/'studies/map_full_v1/map_core.py']);atomic(G/'state.json',dict(status='prepared',fingerprint=p['fingerprint']));print(p['fingerprint'])
def run():
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.inputs import validate_input
 from stvg.teacher_bridge import ResidentTeacher
 from stvg.native import frozen_native_forward
 from stvg.readout import prediction
 from stvg.query import QuerySubjectParser
 from live_map import predictor_class,finish
 p=validate('deployment');assert not(G/'worker.json').exists(),'No blind repeat timedmodelwork';G.mkdir(parents=True,exist_ok=True);setup();gpu=lease();who=identity();atomic(G/'worker.json',who);selection=read(D/'timing_selection.json');items=[];loads={};klass=predictor_class()
 try:
  for ds,rows in selection.items():
   model=klass(ds);teacher=ResidentTeacher();assert teacher.cache_dir is None;model.teacher_service=teacher;t=time.perf_counter();model.source=load_source(model.config);model.source.cpu();gc.collect();torch.cuda.empty_cache();loads[ds]=dict(source_checkpoint_load_seconds=time.perf_counter()-t)
   def once(row,arm,repeat,warmup=False):
    attempt=G/'attempts'/f'{ds}_{row["id"]}_{repeat}_{arm}.json';assert not attempt.exists();atomic(attempt,dict(before_model=True,worker=who,arm=arm,warmup=warmup,GT_read=False));begin=time.perf_counter();tick=begin;frames,ids=decode(row['metadata'],ds);decode_s=time.perf_counter()-tick;validate_input(frames,ids,row['metadata']);torch.manual_seed(20260920);np.random.seed(20260920);tick=time.perf_counter()
    if model.parser is None:model.parser=QuerySubjectParser(R/'checkpoints/stanza')
    parses=dict(subject=model.parser(row['metadata']['caption'])['subject']);parse_s=time.perf_counter()-tick
    if arm=='Source':
     tick=time.perf_counter();source=model.source.cuda();torch.cuda.synchronize();transfer=time.perf_counter()-tick;tick=time.perf_counter()
     with torch.no_grad():batch,recs,base=frozen_native_forward(source,frames,dict(input=row['metadata'],frame_ids=ids,parses=parses))
     native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);boxes=native['boxes'].cpu().numpy().copy();iv=list(native['physical_interval']);torch.cuda.synchronize();source_s=time.perf_counter()-tick;del batch,base,native;tick=time.perf_counter();model.source.cpu();gc.collect();torch.cuda.empty_cache();offload=time.perf_counter()-tick;cost=dict(Source_seconds=source_s,Source_to_GPU_seconds=transfer,Source_to_CPU_seconds=offload,expert_forwards=0,expert_seconds=0.)
    else:
     cap=model.capture(frames,ids,row['metadata'],parses=parses,collect_mask_diagnostics=False);out=finish(cap);boxes=out['boxes'].copy();iv=out['interval'];cc=cap['cost'];assert cc['ReferDINO_query_forwards']==1 and not cc['ReferDINO_cache_hit'];cost=dict(Source_seconds=cc['Source224_seconds'],Source_to_GPU_seconds=cc['Source_to_GPU_seconds'],Source_to_CPU_seconds=cc['Source_to_CPU_seconds'],expert_forwards=1,expert_seconds=cc['expert_current_wall_seconds'],expert_forward_seconds=cc['ReferDINO']['forward_seconds'],expert_load_seconds=cc.get('ReferDINO_load_seconds',0.),shared_initialization_seconds=cc['ERef_seconds'],geometry_seconds=cc['evidence_seconds'],**out['cost']);del cap,out
    del frames;gc.collect();torch.cuda.empty_cache();wall=time.perf_counter()-begin;mp,h=current_MAP(ds,row['id'])
    with np.load(mp)as z:assert np.array_equal(boxes,z[arm])and np.array_equal(iv,z[arm+'_interval']),(ds,row['id'],arm,'currentMAPdeployment bridge mismatch')
    rec=dict(arm=arm,query_wall_seconds=wall,RGB_seconds=decode_s,parse_seconds=parse_s,cost=cost,prediction_bitwise_verified=True,current_MAP_prediction_sha256=h,verification_IO_outside_timer=True,GT_read=False,worker=who);atomic(attempt,{**rec,**dict(status='completed_actual_call',warmup=warmup)});return rec
   try:
    loads[ds]['warmup']={a:once(rows[0],a,0,True)for a in ['Source','FullMAP']}
    for rep in range(1,4):
     for i,row in enumerate(rows):
      for arm in (['Source','FullMAP']if(rep+i)%2 else['FullMAP','Source']):
       v=once(row,arm,rep);v.update(dataset=ds,query=i+1,repeat=rep);items.append(v);atomic(G/'partial.json',dict(items=items,loads=loads));atomic(G/'progress.json',dict(completed=len(items),total=192,worker=who,GT_read=False));print(ds,rep,i+1,arm,v['query_wall_seconds'],flush=True)
   finally:teacher.close();del model;gc.collect();torch.cuda.empty_cache()
  assert len(items)==192 and sum(v['cost']['expert_forwards']for v in items)==96;means={}
  for ds in COUNTS:
   means[ds]={}
   for arm in ['Source','FullMAP']:
    rr=[x for x in items if x['dataset']==ds and x['arm']==arm];keys={k for x in rr for k,v in x['cost'].items()if isinstance(v,(int,float))};means[ds][arm]=dict(calls=len(rr),mean_total_seconds=float(np.mean([x['query_wall_seconds']for x in rr])),median_total_seconds=float(np.median([x['query_wall_seconds']for x in rr])),p90_total_seconds=float(np.quantile([x['query_wall_seconds']for x in rr],.9)),mean_RGB_seconds=float(np.mean([x['RGB_seconds']for x in rr])),mean_parse_seconds=float(np.mean([x['parse_seconds']for x in rr])),components={k:float(np.mean([x['cost'][k]for x in rr if k in x['cost']]))for k in keys})
  result=dict(status='completed_real_current_MAP_deployment_timing',queries_per_domain=16,parents_per_domain=16,repeats=3,measured_calls=192,measured_expert_calls=96,warmup_excluded=True,models_resident_original_offload=True,interquery_RGB_visual_capture_teacher_cache=False,means=means,startup_and_warmup=loads,scope='Actual wall includes original CPU/GPU transfers/decode/parse/native224 Source/same8teacher/geometry/auxDP/repool/one-step/MAP/finalDP/cleanup; no MBR performance comparison, no throughput claim; nested subcomponents not additive; shared-host conditional measurement.',fingerprint=p['fingerprint'],GT_read=False)
  atomic(G/'measurements.json',dict(report=result,items=items));atomic(P/'deployment/report.json',result);atomic(G/'sealed.json',dict(status='completed',worker=who,all_current_MAP_outputs_bitwise=True,calls=192,expert_calls=96,warmup_expert_calls=2,GT_read=False,measurement_sha256=sha(G/'measurements.json')));atomic(G/'state.json',dict(status='complete_measured_verified'))
 except BaseException:atomic(G/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),worker=who));raise
 finally:gpu.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);z=a.parse_args();dict(prepare=prepare,run=run)[z.command]()
