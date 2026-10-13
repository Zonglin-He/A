"""Finite complete supervision intervention fits and explicit exact-supervision aliases."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_admission_association_common_v1 import *

def clean_fit(f):
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 return bare(f)

def fit_one(base,row,expert,initial,reference,qualification,fixed,h_positions):
 from vg_tta.stvg_opd_p4_joint_precision001 import Installation
 from vg_tta.stvg_opd_admission_association_v1 import FixedInstallation,objective_audit,math_audit,noise_audit
 rec=Installation(qualification=qualification);hook=FixedInstallation(rec,h_positions,row['frame_ids'])
 try:
  fit=hook.fit(base,initial,expert,row['frame_ids'],row['key'],'on_policy',config=reference['config'],fixed_evidence=fixed,noise_override=hook.noise)
  fit['geometry']=dict(rounds=hook.rows,GT_read=fixed['GT_assisted'])
  assert prior.old.digest(fit['initial'])==prior.old.digest(initial) and (not hook.rows or hook.rows[0]['Adam_before']['state']=={})
  return fit,dict(status='pass',objective=objective_audit(fit,expert,fixed),Adam=math_audit(fit),native_head_VJP=rec.last_vjp,
   noise=noise_audit(fit,row['key'],h_positions,row['frame_ids']),GT_read=fixed['GT_assisted'])
 except BaseException:
  p=BASE/'failures'/('fit_audit_'+str(time.time_ns()));p.mkdir(parents=True,exist_ok=False)
  (p/'traceback.txt').write_text(traceback.format_exc())
  if 'fit' in locals():save(p/'complete_fit_unaccepted.pt',dict(fit=fit,fixed_evidence=fixed,runtime_sha256=sha(BASE/'RUNTIME.json')))
  raise
 finally:rec.uninstall();base.restore(base.initial)

def gpu_fits(phase):
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();verify();prior.standby();qualification=phase=='qualification';oracle=phase=='oracle'
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
 d=read(BASE/'DESIGN_LOCK.json');selected=[r for r in d['records'] if not qualification or r['index'] in d['qualification_indices']]
 if not qualification:assert read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
 if oracle:
  b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert b['status']=='sealed' and b['logical_complete_fits']==384 and not b['GT_read']
  for n,h in b['files'].items():assert sha(ROOT/n)==h
  od=read(BASE/'ORACLE_DESIGN.json');oracle_records={r['index']:r for r in od['records']};assert len(oracle_records)==64
  oracle_sha=sha(BASE/'ORACLE_DESIGN.json')
 count=captures=rounds=actual=actual_rounds=0;bindings={};model=None;dataset=None;cache=collections.OrderedDict();start=time.time()
 try:
  for r in selected:
   budget();prior.standby();sel=read(ROOT/r['selection_path'])['arms'];sel={'Oracle':oracle_records[r['index']]['selection']} if oracle else sel
   h_positions=[x['position'] for x in read(ROOT/r['selection_path'])['arms']['H_Top1']['rows']]
   pair=prior.old.pair_for(r);initial=prior.old.states(pair)['Inherited_order1'];reference=load(ROOT/r['A_path'])['fit'];inp=normalized(load(ROOT/r['originals'][0]['input_path']));expert=unpack_expert(inp['expert'])
   assert prior.old.digest(inp['expert'])==pair['evidence_sha256'][0]
   previous_groups={}
   if oracle:
    for arm in ARMS:
     rr=read(logical_path(r['index'],arm));previous_groups.setdefault(evidence_digest(rr['selection']),(ROOT/rr['payload_path'],rr['payload_sha256']))
   for repeat in ([0,1] if qualification else [0]):
    groups=dict(previous_groups);needs_capture=any(evidence_digest(fixed) not in groups for fixed in sel.values())
    if needs_capture:
     if dataset!=r['dataset']:
      if model is not None:assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
      torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;dataset=r['dataset']
     assert mh==r['source_model_sha256']==inp['source_model_state_sha256']
     row=read_row(dataset,r['query_ordinal']);assert row['key']==r['key'];frames,ids,_=frames_for(dataset,row,cache);shifted,pixel,spec=observation(dataset,row,'clean',frames)
     batch,records,base=frozen_forward(model,shifted,row);native=prediction(base.zero['logits'],base.zero['boxes'],records,ids);captures+=1
     assert prior.old.digest(base.initial)==prior.old.digest(pair['source_initial_state'])
     assert pixel==inp['pixel_sha256']==r['pixel_sha256'] and ids==inp['frame_ids'] and prior.old.digest(ids)==pair['frame_ids_sha256'] and spec is None
     assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval'];del frames,shifted,batch
    for arm,fixed in sel.items():
     budget();prior.standby();dig=evidence_digest(fixed);reused=dig in groups
     if reused:p,h=groups[dig]
     else:
      torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();fit,checked=fit_one(base,row,expert,initial,reference,qualification,fixed,h_positions)
      comp=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
      if arm=='H_Top1':
       try:equal(clean_fit(fit),bare(reference),comp)
       except BaseException:
        p=BASE/'failures'/('H_reproduction_'+str(time.time_ns()));p.mkdir(parents=True);save(p/'complete_fit_unaccepted.pt',dict(fit=fit,fixed_evidence=fixed,index=r['index'],repeat=repeat));raise
      z=dict(index=r['index'],parent_id=r['parent_id'],dataset=r['dataset'],arm=arm,repeat=repeat,fit=fit,audit=checked,fixed_evidence=fixed,
       original_full_complete_comparison=comp,source_model_sha256=mh,pixel_sha256=pixel,frame_ids=ids,Native_interval=inp['interval'],intervals=r['intervals'],
       original_input_binding=r['originals'][0],pair_path=r['pair_path'],pair_sha256=r['pair_sha256'],evidence_sha256=prior.old.digest(inp['expert']),
       runtime_sha256=sha(BASE/'RUNTIME.json'),Query_Residual_reset=True,Adam_reset=True,no_cross_query_state_writeback=True,GT_read=oracle,
       compute=dict(actual_new_DINO_calls=0,fit_seconds=time.perf_counter()-tick,CUDA_peak_allocated=torch.cuda.max_memory_allocated()))
      if repeat or (phase=='formal' and r['index'] in d['qualification_indices']):
       q=resolve(r['index'],arm,'qualification',0);c=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
       equal(clean_fit(fit),clean_fit(q['fit']),c);equal(fit['geometry'],q['fit']['geometry'],c);z['whole_repeat_or_qualified_formal_bitwise']=c
      p=BASE/phase/f"{r['index']:05}"/(arm+f'_repeat{repeat}.pt');save(p,z);h=sha(p);groups[dig]=(p,h);actual+=1;actual_rounds+=len(fit['rounds']);del fit,checked,z
     assert sha(p)==h;bindings[str(p.relative_to(ROOT))]=h
     q=logical_path(r['index'],arm,phase,repeat);write(q,dict(status='accepted_complete_fixed_supervision_fit' if not reused else 'exact_fixed_supervision_complete_fit_reference',
      payload_path=str(p.relative_to(ROOT)),payload_sha256=h,bytes=p.stat().st_size,selection=fixed,evidence_sha256=dig,canonical_reused=reused,
      index=r['index'],dataset=r['dataset'],arm=arm,repeat=repeat,GT_assisted=oracle,runtime_sha256=sha(BASE/'RUNTIME.json'),time=time.time()))
     bindings[str(q.relative_to(ROOT))]=sha(q);count+=1;rounds+=r['config']['steps'];total=60 if qualification else 64 if oracle else 384
     status(BASE/'GPU_STATUS.json',dict(status='running_actual_'+phase,dataset=r['dataset'],done=count,total=total,actual_complete_GPU_fits=actual,captures=captures,pid=os.getpid(),GT_read=oracle,new_DINO_calls=0,time=time.time()))
     print('ADMISSION_ASSOCIATION',phase,count,total,r['dataset'],arm,'reuse' if reused else 'new',flush=True)
    if needs_capture:del base,row;gc.collect();torch.cuda.empty_cache()
   del pair,initial,reference,inp,expert
  expected=60 if qualification else 64 if oracle else 384;assert count==expected
  assert rounds==(1320 if qualification else 1600 if oracle else 9600)
  if model is not None:assert state_hash(model.state_dict())==mh
  verify();prior.standby()
  if oracle:assert sha(BASE/'ORACLE_DESIGN.json')==oracle_sha
  write(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'ORACLE_PREDICTION_BARRIER.json' if oracle else 'GLOBAL_PREDICTION_BARRIER.json'),dict(status='pass' if qualification else 'sealed',files=bindings,
   logical_complete_fits=count,logical_complete_rounds=rounds,actual_complete_GPU_fits=actual,actual_complete_rounds=actual_rounds,actual_frozen_captures=captures,
   source_state_restored=True,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=oracle,actual_new_DINO_calls=0,elapsed_seconds=time.time()-start,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:
   gpu_fits(sys.argv[1]);write(BASE/(sys.argv[1]+'_worker_exit.json'),dict(status='pass',scope='finite complete fits and exact aliases; source restored/cancelled baseline T',time=time.time()))
 except BaseException:
  p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc());raise
