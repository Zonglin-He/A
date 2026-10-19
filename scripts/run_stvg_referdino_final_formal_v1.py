"""Prelocked RF32 full video→Source→DINO4/E/MBR→Refer8→OPD, no GT."""
from pathlib import Path
import sys,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final'
def main(chunk):
 import torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,decode
 from methods.stvg_referdino_opd_v1.predictor import ReferDINOOPDPredictor
 from main_method.runtime import resource_guard
 runtime=read(D/'FORMAL_RUNTIME_LOCK.json');assert all(sha(R/f)==h for f,h in runtime['pins'].items());p=read(D/'PLAN_LOCK.json');sel=read(D/'SELECTION_LOCK.json');cfg=read(R/'methods/stvg_referdino_opd_v1/config.json');assert cfg['tau']==sel['tau'];setup();lock=lease();start=time.perf_counter();files=[];complete=False;used=sum(read(f)['GPU_seconds']for f in(D/'formal_cost').glob('*.json'))
 def check():
  resource_guard();assert time.perf_counter()-start+3<900;assert torch.cuda.mem_get_info()[0]>4*2**30;assert used+time.perf_counter()-start+3<2400;assert used+sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json'))+time.perf_counter()-start+3<3600
 try:
  predictor=ReferDINOOPDPredictor(check=check);source_hash=None;from methods.decota_final_simplified_v1.tensors import state_hash
  rows=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows']
  for r in p['formal_records'][chunk*8:(chunk+1)*8]:
   check();a=r['alias'];raw=rows[r['query_ordinal']];sub=read(R/f"artifacts/decota_paper_experiments_v1/hc2/subjects/{r['query_ordinal']:05}.json");assert not sub['GT_read'];tick=time.perf_counter();t=time.perf_counter();frames,ids=decode(raw['input']);decode_s=time.perf_counter()-t
   cache=predictor.capture(frames,ids,raw['input']);assert cache['original_positions4']==r['positions4']and cache['planned_positions']==r['positions8'];cache['cost']['decode_seconds']=decode_s
   capture_wall=time.perf_counter()-tick
   assert sha(R/r['prior_input'])==r['prior_sha256'];old=torch.load(R/r['prior_input'],map_location='cpu',weights_only=False);assert not old['GT_read'];assert torch.equal(cache['Source'],old['native_boxes'])and cache['native_interval']==old['interval'];del old
   if source_hash is None:source_hash=state_hash(predictor.source.state_dict())
   # One measured deployable seed41 path first, then matched controls share evidence.
   first=predictor.predict(frames,ids,raw['input'],key=a,cache=cache,seed=41);end_to_end=capture_wall+first['cost']['complete_predict_seconds']
   others=predictor.adapt(cache,key=a,arms=('Direct','Fixed','Live'),seeds=[42,43]);fixed41=predictor.adapt(cache,key=a,arms=('Fixed',),seeds=[41])
   z=first;z['fits'].update(others['fits']);z['fits'].update(fixed41['fits']);z['outputs'].update(others['outputs']);z['outputs'].update(fixed41['outputs']);z.update(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],GT_read=False)
   for s in p['seeds']:
    f=z['fits'][f'Fixed_{s}'];l=z['fits'][f'Live_{s}'];assert torch.equal(f['path'][1]['state'],l['path'][1]['state']);assert torch.equal(f['rounds'][0]['rollout']['weights'],l['rounds'][0]['rollout']['weights'])
   costs=dict(cache['cost'])
   costs.update(full_video_to_final_Live41_seconds=end_to_end,all_controls_wall_seconds=time.perf_counter()-tick,first_Live_expert_cache_reused=False,query_parses_reused=False,parse_runtime_not_measured=False,all_fit_seconds={k:v['seconds']for k,v in z['fits'].items()});z['cost']=costs
   save(D/'formal'/'inputs'/f'{a}.pt',cache);files.append(save(D/'formal'/'predictions'/f'{a}.pt',z));put(D/'STATUS.json',dict(stage='formal',done=chunk*8+len(files),total=32,GT_read=False),True);print('sealed',a,'E2E',round(end_to_end,3),'DINO',cache['cost']['DINO_image_forwards'],flush=True)
   del frames,cache,z,first,others,fixed41;gc.collect();torch.cuda.empty_cache()
  assert state_hash(predictor.source.state_dict())==source_hash;assert all(not x.requires_grad and x.grad is None for x in predictor.source.parameters())
  put(D/'formal'/f'CHUNK_{chunk}_BARRIER.json',dict(files=files,count=8,GT_read=False));complete=True
  if all((D/'formal'/f'CHUNK_{i}_BARRIER.json').exists()for i in range(4)):put(D/'formal'/'GLOBAL_PREDICTION_BARRIER.json',dict(count=32,files=[f for i in range(4)for f in read(D/'formal'/f'CHUNK_{i}_BARRIER.json')['files']],all_before_new_GT=True,selected_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json')))
 finally:put(D/'formal_cost'/f'chunk{chunk}_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,complete=complete,count=len(files)));lock.close()
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except BaseException as e:put(D/f'FORMAL_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e)));raise
