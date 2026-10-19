"""Finite cross-offset temporal SA capability probe, cached native input only."""
from pathlib import Path
import sys,os,time,gc,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_cross_offset128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_cross_offset128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';STAGE=R/'artifacts/stvg_temporal_native_stage128_v1'
ARMS=['Native','CrossOffset','WrongBinding','HiResAuxMix']

def prepare():
 from vg_tta.stvg_temporal_cross_offset128_v1 import properties,decoder_properties
 previous=read(R/'artifacts/stvg_temporal_mc_dropout128_v1/PLAN_LOCK.json');records=previous['records'];assert len(records)==128 and len({r['parent']for r in records})==128
 own=[f.replace('mc_dropout','cross_offset')for f in previous['pins']if 'mc_dropout'in f]+[f for f in previous['pins']if 'mc_dropout'not in f]
 p=dict(records=records,count=128,source_checkpoint=previous['source_checkpoint'],source_checkpoint_sha256=previous['source_checkpoint_sha256'],seeds=[41,42,43],arms=ARMS,primary='CrossOffset',GPU_seconds_cap=900,CPU_seconds_cap=900,SA_CA_calls_cap=6168,native_decoder_calls_cap=516,head_calls_cap=1028,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_Source_encoder=0,new_RGB=0,new_experts=0,new_training=0,new_student_updates=0,spatial_frozen=True,not_OPD=True,conditional_on_original_second_query=True,wrong_binding='roll_other_content_and_value_by_floor_half; original_other_position_fixed')
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',dict(binding=properties(),decoder=decoder_properties()));print('locked128 cross-offset SA only, no GT')

def measure():
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_cross_offset128_v1 import decode,audit_attention
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup()
 def forbid_GT(event,args):
  if event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('measure forbids actual HC GT')
 sys.addaudithook(forbid_GT);p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();files=[];times=[];nativechecks=disabledchecks=custom=sa_audits=0;counter={k:0 for k in ['decoder_attempts','decoder_completed','SA_attempts','SA_completed','CA_attempts','CA_completed','head_attempts','head_completed']};hooks=[];partial=None;success=False;maxaudit=0.
 def check():
  resource_guard();free,total=torch.cuda.mem_get_info();assert free>5*2**30;assert time.time()<p['deadline_epoch']and time.perf_counter()-beg+30<900
 try:
  torch.cuda.set_per_process_memory_fraction(.8);check();m=load_source();mh=state_hash(m.state_dict());flags={n:x.training for n,x in m.named_modules()};assert not any(flags.values());decoder=m.ground_decoder.time_decoder;assert len(decoder.layers)==6 and decoder.norm is not None;loading=time.perf_counter()-beg
  def entered(kind):
   def f(module,args):
    counter[kind+'_attempts']+=1;cap=p['SA_CA_calls_cap']if kind in ['SA','CA']else p['native_decoder_calls_cap']if kind=='decoder'else p['head_calls_cap'];assert counter[kind+'_attempts']<=cap and time.perf_counter()-beg+30<900
   return f
  def completed(kind):
   def f(module,args,out):counter[kind+'_completed']+=1
   return f
  modules=[(decoder,'decoder'),(m.temp_embed,'head')]+[(getattr(l,a),k)for l in decoder.layers for a,k in [('self_attn','SA'),('cross_attn_image','CA')]]
  for module,kind in modules:hooks.extend([module.register_forward_pre_hook(entered(kind)),module.register_forward_hook(completed(kind))])
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias']
   for path,h in [(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
   inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');cache=load(MEM/'temporal_inputs'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');strong=load(STAGE/'predictions'/f'{a}.pt');kws=[transfer(v,'cuda')for v in cache['inputs']];native=[]
   partial=dict(alias=a,query_ordinal=r['query_ordinal'],native_logits=native,logits={},traces={},GT_read=False,completed_CPU_outputs_only=True)
   for off,kw in enumerate(kws):
    assert torch.equal(kw['query_tgt'],kw['query_tgt'][:1].expand_as(kw['query_tgt']))
    kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt'])
    assert kw['query_tgt'].shape[1]==1 and kw['encoded_pos']is not None and kw['query_time']is not None
    with torch.no_grad():nl=m.temp_embed(decoder(**kw)).cpu()
    assert torch.equal(nl[-1],old['logits'][off]);native.append(nl);nativechecks+=1
   partial=dict(alias=a,query_ordinal=r['query_ordinal'],native_logits=native,logits={},traces={},GT_read=False,completed_CPU_outputs_only=True)
   if len(files)<2:
    closed,_=decode(decoder,kws,cross=False)
    with torch.no_grad():cl=[m.temp_embed(v).cpu()for v in closed]
    assert all(torch.equal(x,y)for x,y in zip(native,cl));disabledchecks+=2;del closed,cl
   logits={};traces={}
   for arm,wrong in [('CrossOffset',False),('WrongBinding',True)]:
    check();hs,tr=decode(decoder,kws,wrong=wrong,capture=len(files)<2);custom+=1
    with torch.no_grad():ls=[m.temp_embed(v).cpu()for v in hs]
    logits[arm]=ls;traces[arm]=tr;partial['logits'][arm]=ls;partial['traces'][arm]=tr
    if len(files)<2:
     for row in tr:err=audit_attention(row);maxaudit=max(maxaudit,max(err.values()));sa_audits+=1
    assert len(tr)==12 and all(0<=v['cross_mass_min']<=v['cross_mass_max']<=1.000001 for v in tr);del hs
   for off,kw in enumerate(kws):
    with torch.no_grad():after=m.temp_embed(decoder(**kw)).cpu()
    assert torch.equal(after,native[off]);nativechecks+=1
   qs={arm:mass_table(probabilities(inp['records'],[v[-1]for v in ls]))for arm,ls in logits.items()};assert all(ids==inp['frame_ids']for ids,q in qs.values());out={arm:dict(q=q,interval=final_mbr(inp['frame_ids'],q)['interval'])for arm,(ids,q)in qs.items()};out.update(Native=dict(q=old['q'],interval=inp['interval']),HiResAuxMix=strong['outputs']['HiResAuxMix']);assert set(out)==set(ARMS)
   trajectories=dict(E=np.asarray(inp['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});assert all(np.array_equal(v,strong['trajectories'][k])for k,v in trajectories.items())and {n:x.training for n,x in m.named_modules()}==flags
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=inp['frame_ids'],records=inp['records'],outputs=out,trajectories=trajectories,logits=logits,native_logits=native,traces=traces,GT_read=False,weights_unchanged=True,no_student_updates=True)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('cross_offset',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+60;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,estimated_GPU_seconds=estimate,native_before_after_exact=nativechecks,disabled_exact=disabledchecks,independent_actual_SA_audits=sa_audits,max_attention_scaled_error=maxaudit,GT_read=False));assert estimate<900
   partial=None;del inp,sp,cache,old,strong,kws,native,logits,traces,qs,out,trajectories,ls,tr,after;gc.collect();torch.cuda.empty_cache()
  check();assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters())and {n:x.training for n,x in m.named_modules()}==flags;assert all(sha(R/f)==h for f,h in p['pins'].items());assert all(counter[k+'_attempts']==counter[k+'_completed']for k in ['decoder','SA','CA','head']);check();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_and_flags_unchanged=True,native_logits_exact=nativechecks,disabled_exact=disabledchecks,actual_SA_audits=sa_audits));success=True;print('sealed128 cross-offset, no GT')
 finally:
  for h in hooks:h.remove()
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",dict(partial,call_accounting=dict(counter)))
  put(D/'COST.json',dict(GPU_seconds=time.perf_counter()-beg+3,native_logits_before_after_exact=nativechecks,disabled_exact=disabledchecks,custom_two_stream_decoder_layer_groups=custom,custom_offset_equivalent_decodes=custom*2,call_attempts_and_completed=counter,actual_attention_audits=sa_audits,max_attention_scaled_error=maxaudit,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else None,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else None,torch_version=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),new_Source_encoder=0,new_RGB=0,new_DINO=0,new_ReferDINO=0,new_training=0,spatial_updates=0,new_student_updates=0));lock.close()

if __name__=='__main__':
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=bool(list((D/'partials').glob('*.pt'))or list((D/'predictions').glob('*.pt')))));raise
 finally:lock.close()
