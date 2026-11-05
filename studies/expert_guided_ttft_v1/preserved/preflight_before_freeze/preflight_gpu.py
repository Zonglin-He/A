"""Bounded two SOURCE-domain actual Source/gradient probes, no target labels."""
from pathlib import Path
import os,sys,json,time,hashlib,gc,subprocess,traceback,importlib.util
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/expert_guided_ttft_v1';P=R/'results/expert_guided_ttft_v1';sys.path[:0]=[str(R),str(R/'tools')]
import numpy as np,torch
from full import atomic,sha,identity,read
from model_utils import cpu_source
from core import adapt,decoder_boxes,to_device
from stvg.runtime import setup,lease
from stvg.native import frozen_native_forward
from stvg.representation import observe_native_grid,extract
from stvg.tensors import detached
from stvg.prior import native_prior
from stvg.prototype import one_step
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def resources(start):
 import shutil
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'))>=8*2**30
def foreign():
 spec=importlib.util.spec_from_file_location('protected_service',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);out=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid)
  if pid==os.getpid():continue
  out.append(m.identity_from_proc(pid,name.strip()))
 return out
def prepare():
 assert not(D/'source_gradient_preflight_plan.json').exists();assert read(R/'outputs/shared_time_spatial_recalc_v2/ROOT_completion.json')['all6actualchild_wait0'];assert read(D/'CPU_doctor_complete.json')['optimizer_steps']==0
 files=[*S.glob('*.py'),R/'stvg/native.py',R/'stvg/representation.py',R/'stvg/prototype.py',R/'stvg/prior.py',R/'stvg/runtime.py',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py',R/'outputs/spatial_temporal_six_v1/source_validation_cohort_v1.json',D/'calibration_intent_v1.json']
 for target in ['hc2','vidstg']:files+=[R/'outputs/spatial_temporal_six_v1/source_validation'/target/'sealed.json']
 plan=dict(scope='Only first metadata-preselected SOURCE validation query per direction, correct source-trained model; actual pos_fc/full spatial decoder gradient and bitwise Source/reset/ROI compatibility. Not target pilot or calibration config selection.',new_Source_cap=2,new_expert=0,optimizer_steps_cap=10,config=dict(lr=1e-5,steps=5,lambda_l1=5,lambda_giou='official source ratio HCtarget3/Vidtarget4',clip_norm=.1),code_and_input_sha256={str(f.relative_to(R)):sha(f)for f in files},failures='Full paid native/context/adapted checkpoints preserved before check; hardfail/no fallback/no blind new Source.',priority_gate_sha256=sha(R/'outputs/shared_time_spatial_recalc_v2/ROOT_completion.json'))
 plan['fingerprint']=digest(plan);atomic(D/'source_gradient_preflight_plan.json',plan);atomic(P/'source_gradient_preflight_protocol.json',plan);print(plan['fingerprint'])
def run():
 plan=read(D/'source_gradient_preflight_plan.json');assert digest({k:v for k,v in plan.items()if k!='fingerprint'})==plan['fingerprint']
 for path,h in plan['code_and_input_sha256'].items():assert sha(R/path)==h,path
 assert not(D/'source_gradient_preflight_worker.json').exists();start=time.perf_counter();resources(start);allowed=foreign();lock=lease();setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;atomic(D/'source_gradient_preflight_worker.json',dict(**identity(),allowed_services=allowed));counts=dict(Source=0,expert=0,optimizer_steps=0);reports={};model=None
 from stvg.query import QuerySubjectParser
 from stvg.decode import decode
 parser=QuerySubjectParser(R/'checkpoints/stanza');cohort=read(R/'outputs/spatial_temporal_six_v1/source_validation_cohort_v1.json');base=R/'outputs/spatial_temporal_six_v1/source_validation'
 try:
  for target in ['hc2','vidstg']:
   resources(start);foreign();row=cohort[target]['rows'][0];ds=cohort[target]['source_domain'];key=row['id'];dest=D/'source_gradient_preflight'/target/key;assert not dest.exists();dest.mkdir(parents=True);old=read(base/target/'sealed.json')['files'][key]
   assert sha(base/old['native_path'])==old['native_sha256']and sha(base/old['input_path'])==old['input_sha256'];reference=torch.load(base/old['native_path'],map_location='cpu',weights_only=False,mmap=True);matching=torch.load(base/old['input_path'],map_location='cpu',weights_only=False,mmap=True);tick=time.perf_counter();model,meta=cpu_source(target);atomic(dest/'model_load.json',dict(worker=identity(),seconds=time.perf_counter()-tick,checkpoint_load=meta.get('checkpoint_load')))
   frames,ids=decode(row['metadata'],ds);assert ids==reference['frame_ids'];parsed={'subject':parser(row['metadata']['caption'])['subject']};atomic(dest/'attempt.json',dict(before_Source=True,worker=identity(),input_video_sha256=row['metadata']['video_sha256'],RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),old_native_reference_sha256=old['native_sha256'],GT_read=False));classes=[]
   hook=model.ground_decoder.pos_fc.register_forward_pre_hook(lambda m,a:classes.append(detached(a[0],'cpu')));model.cuda();torch.set_num_threads(4);tick=time.perf_counter()
   try:
    with torch.no_grad(),observe_native_grid(model)as grids:batch,recs,native=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
   finally:hook.remove()
   torch.cuda.synchronize();source_seconds=time.perf_counter()-tick;counts['Source']+=1
   paid=dict(Source=native.zero['boxes'].cpu().numpy(),native_logits=detached(native.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(native.spatial_inputs,'cpu'),frames_cls_calls=classes,native_local_grid=grids,frame_ids=ids,GT_read=False)
   save(dest/'paid_Source_complete.pt',paid);atomic(dest/'paid_Source_checkpoint.json',dict(sha256=sha(dest/'paid_Source_complete.pt'),checkpoint_before_validation=True,Source_seconds=source_seconds,worker=identity()))
   assert np.array_equal(paid['Source'],reference['Source']);assert all(torch.equal(a,b)for a,b in zip(paid['native_logits'],reference['native_logits']));assert recs==reference['records'];assert len(classes)>=4
   ctx={k:paid[k]for k in ['Source','spatial_inputs','native_local_grid','frame_ids','native_logits','records']};ctx['frames_cls']=[classes[-3],classes[-1]];ctx['q0']=native_prior(recs,paid['native_logits'],ids)
   for c,kw in zip(ctx['frames_cls'],ctx['spatial_inputs']):
    with torch.no_grad():v=model.ground_decoder.pos_fc(c.cuda()).sigmoid().unsqueeze(1)
    assert torch.equal(v.cpu(),kw['pred_boxes'])
   save(dest/'context.pt',ctx);atomic(dest/'context_checkpoint.json',dict(sha256=sha(dest/'context.pt'),GT_read=False,source_original_exact=True,frames_cls_actual_readonly_observer=True))
   del batch,native,classes,frames,paid,reference;gc.collect();torch.cuda.empty_cache();model.cpu();torch.cuda.empty_cache()
   cfg=dict(lr=1e-5,lambda_l1=5,lambda_giou=3 if target=='hc2'else4,clip_norm=.1)
   result=adapt(model,ctx,matching['native_boxes'],matching['positions8'],cfg,checkpoints=(5,),scratch=dest/'adapted_checkpoints');counts['optimizer_steps']+=5;out=result[5];features,pool=extract(ctx,out['boxes']);q,proto=one_step(features,ctx['q0'],kappa=0,beta=1);assert np.isfinite(q).all()
   initial_features,_=extract(ctx,ctx['Source']);zeroq,zero_proto=one_step(initial_features,ctx['q0'],kappa=0,beta=1)
   save(dest/'adapted_complete.pt',dict(result=result,features=features,q1=q,prototype=proto,pooling=pool,initial_SourceTR_q1=zeroq));report={k:v for k,v in out.items()if k not in ['boxes','parameter_update_norms']};report.update(target=target,source_domain=ds,context_sha256=sha(dest/'context.pt'),paid_Source_sha256=sha(dest/'paid_Source_complete.pt'),adapted_complete_sha256=sha(dest/'adapted_complete.pt'),new_Source=1,new_expert=0,GT_read=False,worker=identity());atomic(dest/'passed.json',report);reports[target]=report;atomic(D/'source_gradient_preflight_progress.json',dict(completed=len(reports),counts=counts,worker=identity()));model.cpu();del model,ctx,result;model=None;gc.collect();torch.cuda.empty_cache()
  atomic(D/'source_gradient_preflight_complete.json',dict(status='passed_actual_source_full_spatial_gradient_parameter_allframes_reset_two_directions',reports=reports,counts=counts,worker=identity(),elapsed_seconds=time.perf_counter()-start,plan_fingerprint=plan['fingerprint']));print('actual source gradient preflight complete',flush=True)
 except BaseException:
  atomic(D/'source_gradient_preflight_failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
if __name__=='__main__':
 assert sys.argv[1]in ['prepare','run'];globals()[sys.argv[1]]()
