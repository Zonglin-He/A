"""Fixed matched cost protocol; production method and full plan remain immutable."""
import os,sys,json,time,gc
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'tools'));sys.path.insert(0,str(R))
from full import check,RUN,PUBLIC,PROVIDER,read,sha,atomic,identity
D=R/'outputs/maintenance/simplepath_fair_cost_20261023'

def main():
 import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.inputs import validate_input
 from stvg.predictor import Predictor
 from stvg.teacher_bridge import ResidentTeacher
 from stvg.native import frozen_native_forward
 from stvg.readout import prediction
 from stvg.temporal import final_mbr,probabilities,mass_table
 from stvg.query import QuerySubjectParser
 plan=read(D/'plan.json');assert sha(Path(__file__))==plan['script_sha256'];assert sha(D/'selection.json')==plan['selection_sha256'];check(True)
 assert all((RUN/ds/'score_receipt.json').exists()for ds in ['hc2','vidstg']);assert read(RUN/'state.json')['status']=='awaiting_cost_qualitative_publication'
 assert not(D/'measurements.json').exists()and not(D/'partial.json').exists(),'Preserve existing timing data; no duplicate run'
 setup();gpu=lease();atomic(D/'worker.json',identity());selection=read(D/'selection.json');items=[];loads={}
 try:
  for ds,rows in selection.items():
   model=Predictor(ds);expert=ResidentTeacher();assert expert.cache_dir is None;model.teacher_service=expert
   t=time.perf_counter();model.source=load_source(model.config);model.source.cpu();gc.collect();torch.cuda.empty_cache();loads[ds]={'Source_checkpoint_seconds':time.perf_counter()-t}
   def once(row,arm):
    tick=time.perf_counter();frames,ids=decode(row['metadata'],ds);decode_s=time.perf_counter()-tick;validate_input(frames,ids,row['metadata']);torch.manual_seed(20260920);np.random.seed(20260920)
    if model.parser is None:model.parser=QuerySubjectParser(R/'checkpoints/stanza')
    t=time.perf_counter();parses={'subject':model.parser(row['metadata']['caption'])['subject']};parse_s=time.perf_counter()-t
    if arm=='Source':
     t=time.perf_counter();source=model.source.cuda();torch.cuda.synchronize();transfer=time.perf_counter()-t;t=time.perf_counter()
     with torch.no_grad():batch,recs,base=frozen_native_forward(source,frames,dict(input=row['metadata'],frame_ids=ids,parses=parses))
     native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);U,q=mass_table(probabilities(recs,base.zero['logits']));iv=final_mbr(U,q)['interval'];torch.cuda.synchronize();source_s=time.perf_counter()-t;boxes=native['boxes'].cpu().numpy().copy();del batch,base,native
     t=time.perf_counter();model.source.cpu();gc.collect();torch.cuda.empty_cache();offload=time.perf_counter()-t
     costs=dict(Source_seconds=source_s,Source_to_GPU_seconds=transfer,Source_to_CPU_seconds=offload,parse_seconds=parse_s,expert_forwards=0,expert_seconds=0.,geometry_seconds=0.,region_pooling_seconds=0.,prototype_seconds=0.,final_DP_seconds=0.)
    else:
     cap=model.capture(frames,ids,row['metadata'],parses=parses,collect_mask_diagnostics=False);out=model.finish(cap);boxes=out['boxes'].copy();iv=out['interval'];cc=cap['cost'];assert cc['ReferDINO_query_forwards']==1 and not cc['ReferDINO_cache_hit'];costs=dict(Source_seconds=cc['Source224_seconds'],Source_to_GPU_seconds=cc['Source_to_GPU_seconds'],Source_to_CPU_seconds=cc['Source_to_CPU_seconds'],parse_seconds=parse_s,expert_forwards=1,expert_seconds=cc['expert_current_wall_seconds'],expert_model_load_seconds=cc.get('ReferDINO_load_seconds',0.),shared_initialization_seconds=cc['ERef_seconds'],geometry_seconds=cc['evidence_seconds'],native_temporal_seconds=cc['native_time_postprocess_seconds'],**out['cost']);del cap,out
    del frames;gc.collect();torch.cuda.empty_cache();wall=time.perf_counter()-tick
    with np.load(RUN/ds/'predictions'/(row['id']+'.npz'),allow_pickle=False)as z:
     expected='Source'if arm=='Source'else'Full';assert np.array_equal(boxes,z[expected])and np.array_equal(iv,z[expected+'_interval']),(ds,row['id'],arm,'deployment-vs-full mismatch')
    return dict(arm=arm,query_wall_seconds=wall,RGB_seconds=decode_s,cost=costs,prediction_bitwise_verified=True,verification_IO_outside_timer=True,GT_read=False)
   try:
    loads[ds]['warmup']={a:once(rows[0],a)for a in ['Source','Full']}
    for repeat in range(3):
     for i,row in enumerate(rows):
      for arm in (('Source','Full')if(repeat+i)%2==0 else('Full','Source')):
       x=once(row,arm);x.update(dataset=ds,query=i+1,repeat=repeat+1);items.append(x);atomic(D/'partial.json',dict(items=items,loads=loads));atomic(D/'progress.json',dict(completed=len(items),total=192,**identity()));print(json.dumps(dict(dataset=ds,query=i+1,repeat=repeat+1,arm=arm,seconds=x['query_wall_seconds'])),flush=True)
   finally:expert.close();del model;gc.collect();torch.cuda.empty_cache()
  report=dict(status='completed_matched_timing',samples_per_domain=16,parents_per_domain=16,repeats=3,warmup_excluded=True,interquery_capture_RGB_visual_cache=False,weights_resident='One persistent Source model and one persistent ReferDINO process per direction. Original CPU/GPU offload retained; transfers included, not simultaneously GPU-resident weights.',startup_and_loads=loads,means={},scope='Total wall includes RGB, parse, original Source, eight-observation expert, initialization, geometry, auxiliary path, Native2D pooling, prototype update, MBR and final DP. Nested component times must not be added. Mask-range/ablation collectors and verification I/O excluded.',plan_sha256=sha(D/'plan.json'))
  for ds in selection:
   report['means'][ds]={}
   for arm in ['Source','Full']:
    rr=[x for x in items if x['dataset']==ds and x['arm']==arm];keys={k for x in rr for k,v in x['cost'].items()if isinstance(v,(int,float))};report['means'][ds][arm]=dict(queries=len(rr),mean_total_seconds=float(np.mean([x['query_wall_seconds']for x in rr])),median_total_seconds=float(np.median([x['query_wall_seconds']for x in rr])),mean_RGB_seconds=float(np.mean([x['RGB_seconds']for x in rr])),components={k:float(np.mean([x['cost'][k]for x in rr if k in x['cost']]))for k in keys})
  assert sum(x['cost']['expert_forwards']for x in items)==96
  atomic(D/'measurements.json',dict(report=report,items=items));atomic(PUBLIC/'cost.json',report);atomic(D/'sealed.json',dict(status='complete',timed_queries=192,expert_query_calls=96,warmup_expert_query_calls=2,all_outputs_bitwise=True,public_sha256=sha(PUBLIC/'cost.json'),GT_read=False,worker=identity()))
 finally:gpu.close()
if __name__=='__main__':main()
