"""Matched resident-weight, uncached Source/Ours cost measurements after full seals."""
import os,sys,time,gc,json,importlib.util
from pathlib import Path
from tastvg_paper_full import ROOT,PROVIDER,RUN,PUBLIC,read,atomic,check,identity,sha

def main():
 sys.path.insert(0,str(PROVIDER));import numpy as np,torch
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.inputs import validate_input
 from stvg.predictor import Predictor
 from stvg.teacher_bridge import ResidentTeacher
 from stvg.native import frozen_native_forward
 from stvg.readout import prediction
 from stvg.temporal import probabilities,mass_table,final_mbr
 from stvg.spatial import evidence_dp
 from scripts.tube_fusion.core import head_logits
 from scripts.boxfixed_tuning.cache import from_native
 from scripts.boxfixed_tuning.core import temporal
 p=check();assert all((RUN/ds/'score_receipt.json').exists()for ds in p['jobs']);selection=read(RUN/'cost_selection.json');setup();gpu=lease();out=RUN/'cost';items=[];loads={}
 try:
  for ds,rows in selection.items():
   model=Predictor(ds);expert=ResidentTeacher();model.teacher_service=expert;t=time.perf_counter();model.source=load_source(model.config);model.source.cpu();gc.collect();torch.cuda.empty_cache();loads[ds]=dict(Source_checkpoint_seconds=time.perf_counter()-t)
   def once(r,arm):
    tick=time.perf_counter();frames,ids=decode(r['metadata'],ds);decode_s=time.perf_counter()-tick;validate_input(frames,ids,r['metadata']);torch.manual_seed(20260920);np.random.seed(20260920);cost={}
    if arm=='Source':
     if model.parser is None:
      from stvg.query import QuerySubjectParser
      model.parser=QuerySubjectParser(PROVIDER/'checkpoints/stanza')
     parse={'subject':model.parser(r['metadata']['caption'])['subject']};t=time.perf_counter();source=model.source.cuda();torch.cuda.synchronize();transfer_s=time.perf_counter()-t;t=time.perf_counter()
     with torch.no_grad():batch,recs,base=frozen_native_forward(source,frames,dict(input=r['metadata'],frame_ids=ids,parses=parse))
     native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);U,q=mass_table(probabilities(recs,native['logits']));interval=final_mbr(U,q)['interval'];torch.cuda.synchronize();source_s=time.perf_counter()-t;output_boxes=native['boxes'].numpy();output_interval=interval;del batch,base,native;model.source.cpu();gc.collect();torch.cuda.empty_cache();cost=dict(Source_seconds=source_s,Source_to_GPU_seconds=transfer_s,ReferDINO_seconds=0.,geometry_seconds=0.,temporal_seconds=0.,path_seconds=0.)
    else:
     cap=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True);tc=cap['temporal_cache'];head=model.source.temp_embed.cuda();logits,hcost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu();compact=from_native(cap,logits);t=time.perf_counter();tm=temporal(compact,1.);temporal_s=time.perf_counter()-t;t=time.perf_counter();path=evidence_dp(cap['geometry'].for_interval(tm['interval']));path_s=time.perf_counter()-t
     output_boxes=path['final'].numpy();output_interval=tm['interval'];cc=cap['cost'];cost=dict(Source_seconds=cc['Source224_seconds'],ReferDINO_seconds=cc['expert_current_wall_seconds'],geometry_seconds=cc['evidence_seconds'],shared_initialization_seconds=cc['ERef_seconds'],native_temporal_seconds=cc.get('native_time_postprocess_seconds',0.),temporal_seconds=temporal_s,path_seconds=path_s,Source_to_GPU_seconds=cc['Source_to_GPU_seconds'],Source_to_CPU_seconds=cc['Source_to_CPU_seconds'],head=hcost,expert_model_load_seconds=cc.get('ReferDINO_load_seconds',0.),expert_forwards=cc['ReferDINO_query_forwards']);assert cc['ReferDINO_query_forwards']==1;del cap,tc,logits,compact,path
    del frames;gc.collect();torch.cuda.empty_cache();seconds=time.perf_counter()-tick
    with np.load(PROVIDER/'outputs/boxfixed_full_v1'/ds/'predictions'/(r['id']+'.npz'),allow_pickle=False)as z:
     name='Source'if arm=='Source'else'Final';assert np.array_equal(output_boxes,z[name])and np.array_equal(output_interval,z[name+'_interval'])
    return dict(arm=arm,query_wall_seconds=seconds,RGB_seconds=decode_s,cost=cost,prediction_bitwise_verified=True,verification_IO_outside_timer=True,GT_read=False)
   try:
    # Initial calls load parser/expert and warm kernels; excluded from repeats.
    warm={a:once(rows[0],a)for a in ('Source','Ours')};loads[ds]['warmup']=warm
    for repeat in range(3):
     for i,r in enumerate(rows):
      # Alternating order limits systematic warm/order bias without changing data.
      for arm in (('Source','Ours')if(repeat+i)%2==0 else('Ours','Source')):
       x=once(r,arm);x.update(dataset=ds,query=i+1,parent_sha256=__import__('hashlib').sha256(str(r['metadata']['source']).encode()).hexdigest(),repeat=repeat+1);items.append(x);atomic(out/'progress.json',dict(completed=len(items),total=192,**identity()));atomic(out/'partial.json',dict(items=items,loads=loads));print(json.dumps(dict(dataset=ds,query=i+1,repeat=repeat+1,arm=arm,seconds=x['query_wall_seconds'])),flush=True)
   finally:expert.close();del model;gc.collect();torch.cuda.empty_cache()
  report=dict(status='completed_matched_timing',samples_per_domain=16,parents_per_domain=16,repeats=3,warmup_excluded=True,interquery_capture_RGB_visual_cache=False,weights_resident='loaded once in persistent Source/ReferDINO processes; original deployment CPU/GPU offload retained and transfers included. Not an all-models-simultaneously-GPU benchmark.',startup_and_loads=loads,means={},scope='Actual total wall includes RGB/parse/transfers/original Source passes/eight-frame expert/geometry/head/temporal/DP. Module times can nest; never sum them. Auxiliary ablation/mask-analysis writes excluded.')
  for ds in selection:
   report['means'][ds]={}
   for arm in ('Source','Ours'):
    rr=[x for x in items if x['dataset']==ds and x['arm']==arm];keys=set(k for x in rr for k,v in x['cost'].items()if isinstance(v,(int,float)))
    report['means'][ds][arm]=dict(queries=len(rr),mean_total_seconds=float(np.mean([x['query_wall_seconds']for x in rr])),median_total_seconds=float(np.median([x['query_wall_seconds']for x in rr])),mean_RGB_seconds=float(np.mean([x['RGB_seconds']for x in rr])),components={k:float(np.mean([x['cost'][k]for x in rr if k in x['cost']]))for k in keys})
  atomic(out/'measurements.json',dict(report,items=items));atomic(PUBLIC/'cost.json',report);atomic(out/'sealed.json',dict(status='complete',queries=192,expert_calls_timed=96,warmup_expert_calls=2,public_sha256=sha(PUBLIC/'cost.json'),GT_read=False,worker=identity()))
 finally:gpu.close()

if __name__=='__main__':main()
