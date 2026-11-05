"""No-label inventory, real mask-interface preflight and independent freeze."""
from common import *
from studies.review_strengthening_v1.sam2_input_adapter_v1 import selected_prompts,mask_box,lossless_rgb_loader
import argparse
def inventory():
 assert not(D/'propagation_plan.json').exists();torch.set_num_threads(1)
 from stvg.runtime import guard
 sys.addaudithook(guard)
 selection={ds:fixed(R/'outputs/paper_review_suite_v1/selection.json')[ds]['confirm']for ds in ['hc2','vidstg']}
 refs={ds:fixed(F/ds/'sealed.json')['files']for ds in selection};
 for path,value in [(D/'selection.json',selection),(D/'references.json',refs)]:
  if path.exists():assert read(path)==value
  else:atomic(path,value)
 masks=[];counts=dict(queries=0,new_Source=0,new_expert=0,new_optimizer=0,source_native_mask_payloads_present=0,empty_expert_masks=0,historical_reused_expert=0,previously_new_expert=0)
 for ds,rr in selection.items():
  assert len(rr)==128 and len({r['id']for r in rr})==128
  for row in rr:
   a,t,e=reference(ds,row);h,w=row['metadata']['height'],row['metadata']['width'];m=selected_prompts(t,h,w)
   assert m.shape==(8,h,w)and np.array_equal(a['positions8'],e['input_binding']['positions8']);assert tuple(t['selected_frame_scores'].shape)in [(8,1),(8,)]
   counts['queries']+=1;counts['source_native_mask_payloads_present']+=1;counts['empty_expert_masks']+=int((~m.reshape(8,-1).any(1)).sum());counts['previously_new_expert']+=int(e['new_current_expert']);counts['historical_reused_expert']+=not e['new_current_expert']
   masks.append(dict(dataset=ds,id=row['id'],payload_sha256=e['current_teacher']['payload_sha256'],mask_prompt_sha256=hashlib.sha256(m.tobytes()).hexdigest(),empty=int((~m.reshape(8,-1).any(1)).sum()),input_shape=list(t['input_shape']),global_selected_track=int(t['selected_query']),confidence_available=True,matching_Full_prediction_sha256=e['sha256']))
 atomic(D/'mask_inventory.json',dict(status='all256_exact_original_selected_native_masks_available',counts=counts,cases=masks,GT_read=False,worker=identity()))
 print(json.dumps(counts),flush=True)
def preflight():
 torch.set_num_threads(1);assert not(D/'CPU_mask_preflight.json').exists()
 from stvg.runtime import guard
 sys.addaudithook(guard);proof=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   frames,a,t,b=data(ds,row);h,w=frames.shape[1:3];m=selected_prompts(t,h,w);again=selected_prompts(t,h,w);assert np.array_equal(m,again)
   # Independent bilinear pixel map, same align_corners=False resize contract.
   logits=np.asarray(t['native_mask_logits'],np.float32);ih,iw=logits.shape[1:];yy=np.maximum((np.arange(h)+.5)*ih/h-.5,0);xx=np.maximum((np.arange(w)+.5)*iw/w-.5,0);y0=np.floor(yy).astype(int);x0=np.floor(xx).astype(int);y1=np.minimum(y0+1,ih-1);x1=np.minimum(x0+1,iw-1);wy=yy-y0;wx=xx-x0
   interp=(logits[:,y0[:,None],x0[None,:]]*(1-wy)[None,:,None]*(1-wx)[None,None,:]+logits[:,y1[:,None],x0[None,:]]*wy[None,:,None]*(1-wx)[None,None,:]+logits[:,y0[:,None],x1[None,:]]*(1-wy)[None,:,None]*wx[None,None,:]+logits[:,y1[:,None],x1[None,:]]*wy[None,:,None]*wx[None,None,:])
   # floating resize rounding can differ at tiny distance from zero; hard reject
   # binary mismatch, never waive or change prompt threshold.
   assert np.array_equal(m,interp>0)
   images,_,_=lossless_rgb_loader(frames[a['positions8']],1024,True)
   proof.append(dict(dataset=ds,id=row['id'],input_binding=b,mask_prompt_sha256=hashlib.sha256(m.tobytes()).hexdigest(),preprocessed_eight_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),independent_mask_resize_binary_exact=True,repeat_exact=True))
 # Real API routing checked without constructing/inferencing a model.
 import model
 predictor,_,_=model.modules();src=Path(predictor.__file__).read_text();assert 'point_inputs_per_obj[obj_idx].pop(frame_idx, None)'in src
 assert not torch.cuda.is_initialized();atomic(D/'CPU_mask_preflight.json',dict(status='four_real_teacher_RGB_mask_resize_and_native_API_contracts_passed',cases=proof,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_GT=0));print('four exact real mask prompts passed',flush=True)
def freeze():
 assert not(D/'propagation_plan.json').exists()
 assert read(D/'mask_inventory.json')['counts']['queries']==256 and len(read(D/'CPU_mask_preflight.json')['cases'])==4
 src=fixed(R/'outputs/review_strengthening_v1/sam2_source_lock.json')
 code=list(S.glob('*.py'))+[R/'studies/review_strengthening_v1/sam2_input_adapter_v1.py',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py',R/'stvg/decode.py',R/'stvg/runtime.py',R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'stvg/metrics_hc.py',R/'studies/shared_time_spatial_recalc_v2/score.py']+[V/f for f in src['files_sha256']]
 inputs=[D/'selection.json',D/'references.json',D/'mask_inventory.json',D/'CPU_mask_preflight.json',D/'paused_owner_contract.json',R/'outputs/source_sam21_tr_v1/CPU_doctor.json',R/'outputs/review_strengthening_v1/sam2_source_lock.json',R/'outputs/edgetam_comparison_v1/extension_reuse_receipt.json',WEIGHT,EXT]
 for ds in ['hc2','vidstg']:inputs += [F/ds/'sealed.json',F/ds/'current_audit_receipt_v1.json']
 plan=dict(schema='same_ReferDINO8_selected_masks_SAM21_L_fixedFullTime_128_v1',authorization='2026-11-05 那就按照你说的做: matched original expert propagation and source-only correction reliability',scope='Only matched propagation original128 distinctparents per domain; no automatic full expansion or other paused experiment resume',arms=ARMS,pairs=PAIRS,counts=dict(hc2=128,vidstg=128),model=dict(repository=src['repository'],commit=src['commit'],checkpoint_sha256=sha(WEIGHT),config='configs/sam2.1/sam2.1_hiera_l.yaml',weights='originalFP32 strict903tensors doctor reused',compute='pinned official BF16 CUDA autocast and TF32True; no compile; strict CUDA fillholes8 kernel mandatory'),evidence='Exact original selected ReferDINO track nativefloat masks, actual8RGB, physicalpositions and Source grid/prediction references. Native no-padding bilinear logits to original size >0; empty mask retained. All8 masks in fresh official SAM2 state before frame0 native forward; no boxprompt, Sourcebox/E/DP overwrite, prompt-frame output is SAM2 native too. New SAM2 backbone feature encoder means same expert evidence, not same representation or model/compute budget.',time='Identical matching Full Ours_interval perquery; Full Ours arrays already independently audited with same teacherpayload, including6 prior genuine expert cache misses. No new temporal inference and no approximation matching different payloads.',outputs='Every native float score checkpoint+SHA BEFORE validation; binary masks losslessly packed/compressed; independent halfopen bbox emptyzero; retain all first2 raw until actual independent CPU bridge; only own NEW raw scratch disposable after exact readback. Failed state always preserved; no retry/skip/fallback.',metrics='vIoU strict R>.3/R>.5; common Full tIoU exact; conditional dense GT annotated frames in Full predicted event intersection GTevent, exclude exact8observed physical times. Queryequal means with empty support null, whole query primaryv retains all128. ObservedIoU separately. Parentcluster10000 seed20261022 pointwise exposed panel; no target tuning.',lease='Original Source-prompt full remains USER_PAUSED T with in-flight paid state and original globalflock retained. Independent exclusive cooperative borrow lock plus exact oldPID/start/argv/T checks before every query; precise foreigncompute gate; no oldsignals/resume or service changes. Only newworker actively computes.',resources='GPU1+CPU1, CPU12-15 nice19 ionice3 thread1, RAM8GiB/disk80GiB/private160GiB/72h. Original Source/ReferDINO cost inherited nonzero, newSource/expert/optimizer0. Experimental checkpoints included in clock; not matched deployment latency.',code_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(code)},input_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(inputs)},new_Source=0,new_ReferDINO=0,new_optimizer=0,new_SAM2_queries_cap=256)
 plan['fingerprint']=digest(plan);atomic(D/'propagation_plan.json',plan);atomic(P/'propagation_protocol.json',{k:v for k,v in plan.items()if k not in ['code_sha256','input_sha256']});print(plan['fingerprint'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('task',choices=['inventory','preflight','freeze']);x=a.parse_args();globals()[x.task]()
