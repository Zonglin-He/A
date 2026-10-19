"""Finite cached native-pass and auxiliary head knowledge probe, no new pixels."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_mc_dropout128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_mc_dropout128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';HR=R/'artifacts/stvg_temporal_hires128_v1'
STAGE=R/'artifacts/stvg_temporal_native_stage128_v1'
ARMS=['Native','FullMC','HeadOnlyMC','OutputOnlyMC','HiResAuxMix']

def prepare():
 records=[dict(r,native_stage_sha256=sha(STAGE/'predictions'/f"{r['alias']}.pt"))for r in read(STAGE/'PLAN_LOCK.json')['records']]
 assert len(records)==128 and len({r['parent']for r in records})==128 and len({r['alias']for r in records})==128
 own=['protocols/stvg_temporal_mc_dropout128_v1.md','vg_tta/stvg_temporal_mc_dropout128_v1.py','scripts/run_stvg_temporal_mc_dropout128_v1.py','scripts/score_stvg_temporal_mc_dropout128_v1.py','scripts/publish_stvg_temporal_mc_dropout128_v1.py','methods/stvg_referdino_opd_v1/config.json','vg_tta/stvg_posterior_decision_tree_v1.py','scripts/stvg_temporal_marginal_decode_util256_v1.py','methods/stvg_referdino_opd_v1/runtime.py','methods/stvg_referdino_opd_v1/_capture.py','methods/decota_final_simplified_v1/tensors.py','methods/decota_final_simplified_v1/_tastvg_load.py','main_method/runtime.py','external/TA-STVG/models/pipeline.py','external/TA-STVG/models/grounding_model/query_decoder.py','external/TA-STVG/models/net_utils.py','external/TA-STVG/config/defaults.py','external/TA-STVG/experiments/vidstg.yaml','scripts/run_stvg_joint_event_critic_v1.py','scripts/score_stvg_where_when_event128_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/stvg_opd_paper_common_v1.py','scripts/decota_matrix_common_v1.py','scripts/decota_opd_tuning_common_v1.py','vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py','external/TA-STVG/utils/box_utils.py','external/TA-STVG/engine/evaluate.py','external/TA-STVG/datasets/evaluation/vidstg_eval.py','external/TA-STVG/datasets/evaluation/hcstvg_eval.py','artifacts/decota_paper_experiments_v1/hc2/PLAN.json','artifacts/decota_paper_experiments_v1/DESIGN_LOCK.json']
 p=dict(records=records,count=128,source_checkpoint='checkpoints/TASTVG_VidSTG.pth',source_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',seeds=[41,42,43],draws_per_stream=16,total_draws=48,seed_formula='seed*10**10+ordinal*10**6+offset*10**5+draw*1000; decoder+3, head+100,+101',arms=ARMS,primary='FullMC',GPU_seconds_cap=900,CPU_seconds_cap=900,decoder_offset_calls_cap=12808,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,new_student_updates=0,spatial_frozen=True,not_OPD=True,conditional_on_fixed_native_second_query=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});print('locked128 conditional MC, no GT')

def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_mc_dropout128_v1 import one_draw,integrate,stochastic_hidden,paired_head
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();files=[];times=[];nativechecks=zerochecks=replaychecks=0;counter=dict(decoder_attempts=0,decoder_completed=0,head_attempts=0,head_completed=0);hooks=[];partial=None;success=False
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30;assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<900 and counter['decoder_attempts']<=p['decoder_offset_calls_cap']
 try:
  torch.cuda.set_per_process_memory_fraction(.8)
  free,total=torch.cuda.mem_get_info();assert free>4*2**30;check();m=load_source();assert len(m.ground_decoder.time_decoder.layers)==6 and m.temp_embed.num_layers==2;mh=state_hash(m.state_dict());flags={n:x.training for n,x in m.named_modules()};assert not any(flags.values());loading=time.perf_counter()-beg
  def entered(kind):
   def f(module,args):
    counter[kind+'_attempts']+=1
    if kind=='decoder':assert counter['decoder_attempts']<=p['decoder_offset_calls_cap']and time.perf_counter()-beg+30<900
   return f
  def completed(kind):
   def f(module,args,out):counter[kind+'_completed']+=1
   return f
  for module,kind in [(m.ground_decoder.time_decoder,'decoder'),(m.temp_embed,'head')]:
   hooks.extend([module.register_forward_pre_hook(entered(kind)),module.register_forward_hook(completed(kind))])
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');cache=load(MEM/'temporal_inputs'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');strong=load(STAGE/'predictions'/f'{a}.pt');native_h=[];native_logits=[];draws=[dict(stream=seed,draw=d,offsets=[])for seed in p['seeds']for d in range(16)];partial=dict(alias=a,query_ordinal=r['query_ordinal'],draws=draws,native_h=native_h,native_logits=native_logits,GT_read=False,only_completed_CPU_outputs=True,inflight_failed_outputs_not_claimed=True)
   for off,kw0 in enumerate(cache['inputs']):
    check();kw=transfer(kw0,'cuda');kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt'])
    with torch.no_grad():nh=m.ground_decoder.time_decoder(**kw);nl=m.temp_embed(nh)
    assert torch.equal(nl[-1].cpu(),old['logits'][off]);nativechecks+=1;native_h.append(nh.cpu());native_logits.append(nl.cpu())
    if len(files)<2:
     zh,c=stochastic_hidden(m.ground_decoder.time_decoder,kw,41,zero=True);zl,meta=paired_head(m.temp_embed,zh,42,zero=True);assert torch.equal(zl,nl);zerochecks+=1
    for d in draws:
     check();seed=d['stream']*10**10+r['query_ordinal']*10**6+off*10**5+d['draw']*1000;v=one_draw(m,kw,nh,nl,seed);v['seed']=seed;d['offsets'].append(v);assert v['calls']==dict(MHA=12,Dropout=24)
     if len(files)<2 and d['stream']==41 and d['draw']==0:
      again=one_draw(m,kw,nh,nl,seed)
      for k in ['FullMC','HeadOnlyMC','OutputOnlyMC','output_mask','full_pre_output','head_pre_output']:assert torch.equal(v[k],again[k])
      assert v['head_masks']==again['head_masks'];replaychecks+=1
    with torch.no_grad():after=m.temp_embed(m.ground_decoder.time_decoder(**kw))
    assert torch.equal(after,nl);nativechecks+=1;del kw,nh,nl,after;gc.collect();torch.cuda.empty_cache()
   qs,drop=integrate(inp['records'],draws,old['q']);qs.update(Native=old['q'],HiResAuxMix=strong['outputs']['HiResAuxMix']['q']);out={k:dict(q=q,interval=final_mbr(inp['frame_ids'],q)['interval'])for k,q in qs.items()};assert out['Native']['interval']==inp['interval']and out['HiResAuxMix']['interval']==strong['outputs']['HiResAuxMix']['interval'];trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});assert all(np.array_equal(v,strong['trajectories'][k])for k,v in trajectories.items());assert {n:x.training for n,x in m.named_modules()}==flags
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=inp['frame_ids'],records=inp['records'],outputs=out,trajectories=trajectories,draws=draws,native_h=native_h,native_logits=native_logits,removed_mass=drop,GT_read=False,weights_unchanged=True,no_student_updates=True)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('conditional_MC',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,native_before_after_exact=nativechecks,zero_rate_exact=zerochecks,same_seed_draw_exact=replaychecks,GT_read=False));assert estimate<900
   del inp,sp,cache,old,strong,draws,native_h,native_logits,qs,out,trajectories;gc.collect();torch.cuda.empty_cache()
  check();assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters())and {n:x.training for n,x in m.named_modules()}==flags;assert all(sha(R/f)==h for f,h in p['pins'].items());check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_and_flags_unchanged=True,native_logits_exact=nativechecks,zero_rate_exact=zerochecks,same_seed_replay_exact=replaychecks));success=True;print('sealed128 stochastic time, no GT')
 finally:
  for hdl in hooks:hdl.remove()
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,call_accounting=dict(counter)))
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,time_decoder_offset_calls=counter['decoder_completed'],projection_head_calls=counter['head_completed'],call_attempts_and_completed=counter,native_logits_before_after_exact=nativechecks,zero_rate_exact=zerochecks,same_seed_replay_exact=replaychecks,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else None,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else None,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,new_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 import fcntl
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=bool(list((D/'partials').glob('*.pt'))or list((D/'predictions').glob('*.pt'))),failed_inflight_outputs_not_claimed=True));raise
