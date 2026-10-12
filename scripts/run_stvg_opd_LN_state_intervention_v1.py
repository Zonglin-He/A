"""Finite serial genuine GPU qualification/formal controlled LN transplants."""
import collections,gc,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_LN_state_intervention_common_v1 import *

def standby():
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 for expected in read(BASE/'BASELINE_STANDBY_IDENTITY.json')['processes']:
  actual=process(expected['pid']);assert actual['state']=='T' and actual['start_ticks']==expected['start_ticks'] and actual['command']==expected['command']

def fit_one(base,row,expert,initial,reference,qualification):
 import torch
 from vg_tta.stvg_opd_p4_joint_precision001 import Installation
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import Geometry,mathematical_readback
 from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
 from scripts.stvg_opd_design_ablation_common_v1 import audit
 rec=Installation(qualification=qualification);hook=Geometry(rec,'LN_state_intervention',reference)
 try:
  fit=fit_variant(base,initial,expert,row['frame_ids'],row['key'],'on_policy',reference['config'])
  fit['geometry']=dict(rounds=hook.rows,LN_initialization_intervention=True,GT_read=False)
  assert digest(fit['initial'])==digest(initial) and not torch.count_nonzero(fit['initial'][Q]).item()
  assert not hook.rows or hook.rows[0]['Adam_before']['state']=={}
  n,c=noise(fit,row['key']);a=audit(fit,expert);m=mathematical_readback(fit)
  return fit,dict(status='pass',objective=a,Adam=m,native_head_VJP=rec.last_vjp,noise=n,sample_coordinates=c,GT_read=False)
 finally:hook.close();rec.uninstall();base.restore(base.initial)

def gpu_fits(qualification):
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();verify();standby()
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
   budget();standby()
   if dataset!=r['dataset']:
    if model is not None:
     assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;dataset=r['dataset']
   pair=pair_for(r);initials=states(pair);originals=[load(ROOT/x['path']) for x in r['originals']]
   inputs=[normalized(load(ROOT/x['input_path'])) for x in r['originals']];inp=inputs[0]
   assert digest(inp)==digest(inputs[1]) and inp['source_model_state_sha256']==mh==r['source_model_sha256']
   expert=unpack_expert(inp['expert']);assert digest(inp['expert'])==pair['evidence_sha256'][0]
   for repeat in ([0,1] if qualification else [0]):
    row=read_row(dataset,r['query_ordinal']);frames,ids,_=frames_for(dataset,row,cache);shifted,pixel,spec=observation(dataset,row,'clean',frames)
    batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids);captures+=1
    assert digest(base.initial)==digest(pair['source_initial_state'])
    assert pixel==inp['pixel_sha256']==r['pixel_sha256'] and ids==inp['frame_ids'] and digest(ids)==pair['frame_ids_sha256'] and spec is None
    assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval']
    del frames,shifted,batch
    for arm in ARMS:
     budget();standby();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
     fit,checked=fit_one(base,row,expert,initials[arm],originals[0]['fit'],qualification)
     compare=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
     if arm=='Source':assert torch.equal(fit['before'],inp['native_boxes'])
     else:
      old=originals[ARMS.index(arm)-1]['fit'];equal(bare(fit),bare(old),compare)
     assert fit['positions']==originals[0]['fit']['positions']==originals[1]['fit']['positions']
     assert checked['noise']==pair['random_traces'][0] or [dict(round=n['round'],seed=n['seed'],noise_shape=n['noise_shape'],noise_sha256=n['noise_sha256']) for n in checked['noise']]==[{k:v for k,v in n.items() if k!='samples_replay_exact'} for n in pair['random_traces'][0]]
     z=dict(index=r['index'],parent_id=r['parent_id'],dataset=dataset,arm=arm,repeat=repeat,fit=fit,audit=checked,
      original_full_complete_comparison=compare,source_model_sha256=mh,pixel_sha256=pixel,frame_ids=ids,Native_interval=inp['interval'],
      original_input_binding=r['originals'][0],pair_path=r['pair_path'],pair_sha256=r['pair_sha256'],evidence_sha256=digest(inp['expert']),
      runtime_sha256=sha(BASE/'RUNTIME.json'),Query_Residual_reset=True,Adam_reset=True,no_cross_query_state_writeback=True,GT_read=False,
      compute=dict(actual_new_DINO_calls=0,fit_seconds=time.perf_counter()-tick,CUDA_peak_allocated=torch.cuda.max_memory_allocated()))
     phase='qualification' if qualification else 'formal';p=BASE/phase/f"{r['index']:05}"/(arm+f'_repeat{repeat}.pt')
     save(p.parent/('UNACCEPTED_'+p.name),z);c=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
     if repeat:
      q=load(p.parent/(arm+'_repeat0.pt'));equal(bare(fit),bare(q['fit']),c);equal(fit['geometry'],q['fit']['geometry'],c)
      z['whole_repeat_bitwise']=c
     elif not qualification and r['index'] in d['qualification_indices']:
      q=load(BASE/'qualification'/f"{r['index']:05}"/(arm+'_repeat0.pt'));equal(bare(fit),bare(q['fit']),c);equal(fit['geometry'],q['fit']['geometry'],c)
      z['qualified_formal_bitwise']=c
     save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(status='accepted_complete_controlled_LN_fit',sha256=h,bytes=p.stat().st_size,
      dataset=dataset,arm=arm,index=r['index'],repeat=repeat,GT_read=False,runtime_sha256=sha(BASE/'RUNTIME.json'),time=time.time()))
     bindings[str(p.relative_to(ROOT))]=h;count+=1;rounds+=len(fit['rounds'])
     status(BASE/'GPU_STATUS.json',dict(status='running_actual_qualification' if qualification else 'running_actual_formal',dataset=dataset,done=count,
      total=24 if qualification else 192,captures=captures,pid=os.getpid(),GT_read=False,new_DINO_calls=0,time=time.time()))
     print('LN_STATE_PROGRESS',phase,count,24 if qualification else 192,dataset,arm,flush=True);del fit,checked,z
    del base,row;gc.collect();torch.cuda.empty_cache()
   del originals,inputs,inp,pair,expert,initials
  assert count==(24 if qualification else 192) and captures==(8 if qualification else 64)
  assert rounds==(600 if qualification else 4800) and state_hash(model.state_dict())==mh
  verify();standby()
  write(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'),dict(status='pass' if qualification else 'sealed',
   files=bindings,actual_complete_GPU_fits=count,actual_complete_rounds=rounds,actual_frozen_captures=captures,source_state_restored=True,
   runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,actual_new_DINO_calls=0,elapsed_seconds=time.time()-start,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

def qualification_root():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old()
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare,mathematical_readback
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from scripts.run_decota_paper_main_v1 import unpack_expert,read_row
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.stvg_opd_design_ablation_common_v1 import audit
 b=read(BASE/'QUALIFICATION_BARRIER.json');assert b['status']=='pass' and b['actual_complete_GPU_fits']==24
 comparisons=dict(tensors=0,tensor_coordinates=0,scalar_values=0);checks=0;vjp=0;empty=0;episodes=0
 lookup={r['index']:r for r in read(BASE/'DESIGN_LOCK.json')['records']}
 for rel,h in b['files'].items():
  assert sha(ROOT/rel)==h;z=load(ROOT/rel);r=lookup[z['index']];pair=pair_for(r);ss=states(pair);assert digest(z['fit']['initial'])==digest(ss[z['arm']])
  inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert']);a=audit(z['fit'],ex);m=mathematical_readback(z['fit'])
  assert a==z['audit']['objective'] and m==z['audit']['Adam'];checks+=m['scalar_checks'];empty+=z['fit']['empty'];episodes+=1
  if z['fit']['empty']:assert not z['audit']['native_head_VJP'] and not z['fit']['gradient_calls']
  else:assert len(z['audit']['native_head_VJP'])==r['config']['steps'];vjp+=len(z['audit']['native_head_VJP'])
  first=load((ROOT/rel).parent/(z['arm']+'_repeat0.pt'));equal(bare(z['fit']),bare(first['fit']),comparisons);equal(z['fit']['geometry'],first['fit']['geometry'],comparisons)
  if z['arm']!='Source':equal(bare(z['fit']),bare(load(ROOT/r['originals'][ARMS.index(z['arm'])-1]['path'])['fit']),comparisons)
  else:assert torch.equal(z['fit']['before'],inp['native_boxes'])
  assert noise(z['fit'],read_row(r['dataset'],r['query_ordinal'])['key'])== (z['audit']['noise'],z['audit']['sample_coordinates'])
  assert sha(ROOT/rel)==h
 assert episodes==24 and empty==6 and vjp==540
 write(BASE/'ROOT_QUALIFICATION.json',dict(status='pass',scope='all24 actual full three-LN-state fits/two fresh captures/exact inherited originals/source Native/empty Adam/objective/state/math/common noise/actual native head VJP',
  actual_complete_GPU_fits=24,independent_Adam_scalar_checks=checks,full_comparisons=comparisons,actual_native_head_VJP_rounds=vjp,
  empty_complete_episodes=empty,independent_full_decoder_Jacobian=False,CPU_only=True,GT_read=False,time=time.time()))

def controller():
 verify();standby();py=ROOT/'.conda/tubedetr/bin/python';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
 for mode in ['qualification','qualification_root','formal','score','root']:
  if mode in ['qualification','formal']:script=Path(__file__).name
  elif mode=='qualification_root':script=Path(__file__).name
  else:script=('score' if mode=='score' else 'audit')+'_stvg_opd_LN_state_intervention_v1.py'
  args=[str(py),'-B','scripts/'+script]+([mode] if script==Path(__file__).name else [])
  with (BASE/(mode+'.log')).open('xb') as log:
   worker=subprocess.Popen(args,cwd=ROOT,env=env if mode in ['qualification','formal'] else {**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
   status(BASE/'STATUS.json',dict(status='running_finite_'+mode,stage=mode,controller_pid=os.getpid(),worker_pid=worker.pid,GT_read=mode in ['score','root'],time=time.time()))
   result=worker.wait();assert result==0,(mode,result)
  write(BASE/'worker_exit'/(mode+'.json'),dict(status='pass',stage=mode,pid=worker.pid,exit_code=result,time=time.time()))
 status(BASE/'STATUS.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',done=192,total=192,CPU_only=True,GT_read=True,time=time.time()))

if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='prepare':prepare()
  elif mode=='controller':controller()
  elif mode=='qualification_root':qualification_root()
  else:gpu_fits(mode=='qualification')
 except BaseException:
  p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',stage=sys.argv[1] if len(sys.argv)>1 else 'unknown',evidence=str(p.relative_to(ROOT)),time=time.time()));raise
