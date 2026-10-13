"""Finite technical qualification and actual complete live four-arm trajectories."""
import collections,gc,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_real_forward_task_common_v1 import *

def science_geometry(x):
 if isinstance(x,dict):return {k:science_geometry(v) for k,v in x.items() if k!='candidate_seconds'}
 if isinstance(x,list):return [science_geometry(v) for v in x]
 return x

def clean_fit(f):
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 return {k:v for k,v in bare(f).items() if k!='policy_arm'}

def fit_one(base,row,expert,initial,reference,qualification,arm,visited):
 import torch
 from vg_tta.stvg_opd_p4_joint_precision001 import Installation
 from vg_tta.stvg_opd_real_forward_task_v1 import Policy,objective_audit,math_audit
 from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
 rec=Installation(qualification=qualification);hook=Policy(rec,arm,visited)
 try:
  fit=fit_variant(base,initial,expert,row['frame_ids'],row['key'],'on_policy',reference['config'])
  fit['geometry']=dict(rounds=hook.rows,GT_read=False);fit['policy_arm']=arm
  assert prior.old.digest(fit['initial'])==prior.old.digest(initial)
  assert not hook.rows or hook.rows[0]['Adam_before']['state']=={}
  n,c=prior.old.noise(fit,row['key']);a=objective_audit(fit,expert);m=math_audit(fit)
  return fit,dict(status='pass',objective=a,Adam=m,native_head_VJP=rec.last_vjp,noise=n,sample_coordinates=c,GT_read=False)
 finally:hook.close();rec.uninstall();base.restore(base.initial)

def gpu_fits(qualification):
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();verify();prior.standby()
 from scripts.run_stvg_opd_expanded_after_p8_priority001 import gpu
 handle=gpu()
 from scripts.run_decota_paper_main_v1 import model_for,read_row,frames_for,unpack_expert
 from scripts.stvg_opd_paper_inputs_v1 import observation
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from scripts.decota_paper_common_v1 import budget
 d=read(BASE/'DESIGN_LOCK.json');selected=[r for r in d['records'] if r['index'] in d['qualification_indices']] if qualification else d['records']
 if not qualification:assert read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
 count=0;captures=0;rounds=0;bindings={};model=None;dataset=None;cache=collections.OrderedDict();start=time.time()
 try:
  for r in selected:
   budget();prior.standby()
   if dataset!=r['dataset']:
    if model is not None:
     assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;dataset=r['dataset']
   pair=prior.old.pair_for(r);initials={a:prior.old.states(pair)['Inherited_order1'] for a in ARMS};reference=load(ROOT/r['A_path'])['fit']
   inputs=[normalized(load(ROOT/x['input_path'])) for x in r['originals']];inp=inputs[0]
   assert prior.old.digest(inp)==prior.old.digest(inputs[1]) and inp['source_model_state_sha256']==mh==r['source_model_sha256']
   expert=unpack_expert(inp['expert']);assert prior.old.digest(inp['expert'])==pair['evidence_sha256'][0]
   for repeat in ([0,1] if qualification else [0]):
    row=read_row(dataset,r['query_ordinal']);frames,ids,_=frames_for(dataset,row,cache);shifted,pixel,spec=observation(dataset,row,'clean',frames)
    batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids);captures+=1
    assert prior.old.digest(base.initial)==prior.old.digest(pair['source_initial_state'])
    assert pixel==inp['pixel_sha256']==r['pixel_sha256'] and ids==inp['frame_ids'] and prior.old.digest(ids)==pair['frame_ids_sha256'] and spec is None
    assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval']
    del frames,shifted,batch
    for arm in ARMS:
     budget();prior.standby();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
     fit,checked=fit_one(base,row,expert,initials[arm],reference,qualification,arm,r['visited'])
     compare=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
     if arm=='A':equal(clean_fit(fit),bare(reference),compare)
     assert fit['positions']==reference['positions']
     assert checked['noise']==pair['random_traces'][0] or [dict(round=n['round'],seed=n['seed'],noise_shape=n['noise_shape'],noise_sha256=n['noise_sha256']) for n in checked['noise']]==[{k:v for k,v in n.items() if k!='samples_replay_exact'} for n in pair['random_traces'][0]]
     z=dict(index=r['index'],parent_id=r['parent_id'],dataset=dataset,arm=arm,repeat=repeat,fit=fit,audit=checked,
      original_full_complete_comparison=compare,source_model_sha256=mh,pixel_sha256=pixel,frame_ids=ids,Native_interval=inp['interval'],
      original_input_binding=r['originals'][0],pair_path=r['pair_path'],pair_sha256=r['pair_sha256'],evidence_sha256=prior.old.digest(inp['expert']),
      runtime_sha256=sha(BASE/'RUNTIME.json'),Query_Residual_reset=True,Adam_reset=True,no_cross_query_state_writeback=True,GT_read=False,
      compute=dict(actual_new_DINO_calls=0,fit_seconds=time.perf_counter()-tick,CUDA_peak_allocated=torch.cuda.max_memory_allocated()))
     phase='qualification' if qualification else 'formal';p=BASE/phase/f"{r['index']:05}"/(arm+f'_repeat{repeat}.pt')
     c=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
     if repeat:
      q=load(p.parent/(arm+'_repeat0.pt'));equal(clean_fit(fit),clean_fit(q['fit']),c);equal(science_geometry(fit['geometry']),science_geometry(q['fit']['geometry']),c)
      z['whole_repeat_bitwise']=c
     elif not qualification and r['index'] in d['qualification_indices']:
      q=load(BASE/'qualification'/f"{r['index']:05}"/(arm+'_repeat0.pt'));equal(clean_fit(fit),clean_fit(q['fit']),c);equal(science_geometry(fit['geometry']),science_geometry(q['fit']['geometry']),c)
      z['qualified_formal_bitwise']=c
     save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(status='accepted_complete_live_forward_task_fit',sha256=h,bytes=p.stat().st_size,
      dataset=dataset,arm=arm,index=r['index'],repeat=repeat,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME.json'),time=time.time()))
     bindings[str(p.relative_to(ROOT))]=h;count+=1;rounds+=len(fit['rounds'])
     status(BASE/'GPU_STATUS.json',dict(status='running_actual_qualification' if qualification else 'running_actual_formal',dataset=dataset,done=count,
      total=32 if qualification else 256,captures=captures,pid=os.getpid(),GT_read=False,new_DINO_calls=0,time=time.time()))
     print('LIVE_FORWARD_TASK',phase,count,32 if qualification else 256,dataset,arm,flush=True);del fit,checked,z
    del base,row;gc.collect();torch.cuda.empty_cache()
   del reference,inputs,inp,pair,expert,initials
  assert count==(32 if qualification else 256) and captures==(8 if qualification else 64)
  assert rounds==(800 if qualification else 6400) and state_hash(model.state_dict())==mh
  verify();prior.standby()
  write(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'),dict(status='pass' if qualification else 'sealed',
   files=bindings,actual_complete_GPU_fits=count,actual_complete_rounds=rounds,actual_frozen_captures=captures,source_state_restored=True,
   runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,actual_new_DINO_calls=0,elapsed_seconds=time.time()-start,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()


if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   qualification=sys.argv[1]=='qualification';gpu_fits(qualification)
   phase='qualification' if qualification else 'formal'
   write(BASE/(phase+'_worker_exit.json'),dict(status='pass',scope='finite complete fit worker; source restored and cancelled baseline same T identity',GT_read=False,time=time.time()))
 except BaseException:
  p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc());raise
