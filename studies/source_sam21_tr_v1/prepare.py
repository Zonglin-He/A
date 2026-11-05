"""Read-only extraction of already audited Source predictions into narrow inputs."""
from common import *
import argparse
OLD=R/'outputs/reviewer_key_controls_v1'
def prepare_inputs():
 assert not(D/'source_inputs_sealed.json').exists()
 sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'));import common_v1 as old
 torch.set_num_threads(1);rosters={};domains={};pilot=read(R/'outputs/matched_pipeline_timing_v1/selection.json');atomic(D/'pilot_selection.json',{ds:[r['id']for r in rr]for ds,rr in pilot.items()})
 allowed=['width','height','frame_count','fps','duration','video_path','video_sha256','frame_ids','original_video_id','index']
 for ds in COUNTS:
  rr=old.selected.records(ds);seal=read(OLD/'temporal'/ds/'sealed.json');ack=read(OLD/'temporal'/ds/'reporting_receipt.json');files={};out=[];assert len(rr)==COUNTS[ds]
  for i,row in enumerate(rr,1):
   e=seal['files'][row['id']];assert sha(OLD/e['path'])==e['sha256']
   with np.load(OLD/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','SourceTR_interval','frame_ids','positions8']}
   assert a['Source'].shape==(len(a['frame_ids']),4)and len(set(a['positions8']))==8
   meta={k:row['metadata'][k]for k in allowed if k in row['metadata']};meta['frame_ids']=a['frame_ids'].tolist();assert 'caption'not in meta
   narrow=dict(id=row['id'],dataset=ds,metadata=meta);path=source_path(ds,row['id']);path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb')as f:np.savez_compressed(f,**a)
   with np.load(path)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
   files[row['id']]=dict(path=str(path.relative_to(D)),sha256=sha(path),parent_temporal_prediction_sha256=e['sha256']);out.append(narrow)
   if i%100==0:atomic(D/'input_preparation_progress.json',dict(dataset=ds,completed=i,total=COUNTS[ds],new_Source=0,new_expert=0,new_GT=0))
  domains[ds]=dict(files=files,queries=len(out),source_temporal_seal_sha256=sha(OLD/'temporal'/ds/'sealed.json'),source_temporal_audit_sha256=sha(OLD/'temporal'/ds/'reporting_receipt.json'));rosters[ds]=out
 atomic(D/'source_roster.json',rosters);atomic(D/'source_inputs_sealed.json',dict(domains=domains,queries=12205,Source_forward_calls=0,expert_calls=0,optimizer_steps=0,GT_read=False,caption_or_ReferDINO_payload_in_SAM_inputs=False,worker=identity()));print('all12205 narrow Source-only inputs sealed, no model',flush=True)
def preflight():
 torch.set_num_threads(1);sys.path.insert(0,str(V));from sam2.utils.misc import _load_img_as_tensor
 from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader
 from model import mask_box
 from PIL import Image
 import tempfile
 cases=[]
 for ds in COUNTS:
  for row in rows('pilot',ds)[:2]:
   frames,a,binding=data(ds,row);actual,h,w=lossless_rgb_loader(frames[[0,-1]],1024,True);expected=torch.zeros_like(actual)
   with tempfile.TemporaryDirectory(prefix='SAM21-CPU-lossless-')as temp:
    for i,rgb in enumerate(frames[[0,-1]]):
     p=Path(temp)/f'{i}.png';Image.fromarray(rgb).save(p);im,hh,ww=_load_img_as_tensor(str(p),1024);assert(h,w)==(hh,ww);expected[i]=im
   expected-=torch.tensor([.485,.456,.406])[:,None,None];expected/=torch.tensor([.229,.224,.225])[:,None,None];assert torch.equal(expected,actual)
   assert np.array_equal(mask_box(np.zeros((3,5),bool)),np.zeros(4))and np.array_equal(mask_box(np.ones((3,5),bool)),[.5,.5,1,1]);cases.append(dict(dataset=ds,input_binding=binding,Source_native_positions_exact=True,official_preprocessing_pixel_exact=True))
 assert not torch.cuda.is_initialized();atomic(D/'CPU_input_preflight.json',dict(status='four_real_Source_only_RGB_prompt_time_preprocessing_cases_passed',cases=cases,new_Source=0,new_expert=0,new_model=0,new_GT=0));print('four real Source-only input contracts passed',flush=True)
def freeze():
 assert not(D/'plan.json').exists();assert len(read(D/'CPU_input_preflight.json')['cases'])==4;assert read(D/'CPU_doctor.json')['proof']['strict_all_keys_tensors_exact']
 source=read(R/'outputs/review_strengthening_v1/sam2_source_lock.json')
 for f,h in source['files_sha256'].items():assert sha(V/f)==h,f
 ck=read(R/'outputs/review_strengthening_v1/vendor/sam2.1_hiera_large.download.json');assert sha(WEIGHT)==ck['sha256']
 code=list(S.glob('*.py'))+[R/'studies/review_strengthening_v1/sam2_input_adapter_v1.py',R/'stvg/decode.py',R/'stvg/positions.py',R/'stvg/runtime.py',R/'stvg/metrics.py',R/'stvg/metrics_base.py',R/'stvg/metrics_hc.py',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py']+[V/f for f in source['files_sha256']if Path(f).suffix in ['.py','.yaml','.cu','.cpp','.h']]
 inp=[D/'source_inputs_sealed.json',D/'source_roster.json',D/'pilot_selection.json',D/'CPU_input_preflight.json',D/'CPU_doctor.json',R/'outputs/review_strengthening_v1/sam2_source_lock.json',R/'outputs/review_strengthening_v1/vendor/sam2.1_hiera_large.download.json',WEIGHT,EXT,R/'outputs/edgetam_comparison_v1/extension_reuse_receipt.json',R/'outputs/expert_guided_ttft_v1/target_pilot_v1/ROOT_completion.json']
 scoreinputs=[]
 for ds in COUNTS:
  inp += [OLD/'temporal'/ds/'reporting_receipt.json',OLD/'temporal'/ds/'sealed.json']
  scoreinputs += [OLD/'temporal'/ds/'metrics.json']
 plan=dict(schema='Source_prompt_SAM21_L_plus_Source_box_temporal_refinement_pilot_then_full_v1',authorization='2026-11-05 user: 那就这样做，先做128 然后全量',primary='Frozen Source original eight boxes/physical times prompt SAM2.1 Hiera-L, all native sampled RGB frames, then original Source-box TR interval. Same Source-TR rule and intervals already audited, no new Source/features/ReferDINO/optimizer.',stages=['original128queries_and_distinctparents_per_domain','same_frozen_method_full_HC1902_and_Vid10303_reusing_all256pilot_outputs'],full_release='Automatic only after BOTH128 prediction seals, separate input audits/official scoring/independent all-mask/manual-score/meansaudits and export complete. No target accuracy threshold, parameter/precision/seed/model search or manual ACK wait. Predeclared same model/protocol. Existing completed TTFT128 results retained; TTFT full not launched.',model=dict(repository=source['repository'],commit=source['commit'],checkpoint=ck['url'],checkpoint_sha256=ck['sha256'],config='configs/sam2.1/sam2.1_hiera_l.yaml',image_size=1024,weights='strict_FP32_checkpoint_all_tensor_exact',compute='official BF16 CUDA autocast, TF32 allowed as official Ampere+ notebook; eval, deterministic flags, no torch.compile',postprocessing='official fillholes8/stability switches unchanged; exact CUDA extension mandatory, strict wrapper removes silent exception fallback, original kernel/math unchanged'),prompts='Exactly original Source nativeMAPhull-selected8 physical positions/raw Source boxes. No ReferDINO boxes/masks/caption/E/current time/candidates/DP input to tracker. All8box conditioning before official frame0 forward; no custom reverse/fusion. SAM output is native logit>0 halfopen mask-to-box including observed frames, emptyzero/no Source overwrite or fallback.',temporal='SourceTR primary interval comes only from original Source boxes/native pretext features/prior, one pair kappa0/beta1/context1.5/MAP; no temporal update on SAM2 trajectory and no expert input to temporal branch. SAM2Native secondary uses Source original nativeMAPhull event, same tracking output/no extra forward.',budget='One external model per pipeline, but SAM2 encodes all sampled frames, so neither eight-expert-forward compute nor knowledge/precision equality is claimed.',arms=ARMS,pairs=PAIRS,new_Source=0,new_ReferDINO=0,new_optimizer=0,new_SAM_video_queries_cap=12205,pilot_video_queries=256,full_additional_video_queries_cap=11949,raw='Every full native-resolution float score checkpoint+SHA BEFORE validation; packed masks compressed losslessly and all boxes cold rederived before only own transient float disposal. First2 NEW requests/domain/stage independently reread all raw scores/RGB/preprocessing/Source fields/masks before disposal, coordinator invokes actual CPU bridge and records wait0 automatically. No manual marker/skip/fallback/blind retry.',evaluation='Each whole stage/domain seals before separate CPU input audit, new SAM2Native/SAM2TR official scores, other arms audited scalar reuse, separate manual dense pixel-IoU/allmask/InputSHA/allmeans/parent10kCI/severe uniononce/rescues/shortlong audit/export. PointwiseCI, exposed cohorts, no unknown confirmation claim.',resources='1 GPU+at most1 CPU child; CPU12-15nice19idleIO, Source inherited nonzero cost, RAM8GiB/disk80GiB/private160GiB/domain72h, exact foreign-service gate/globallease, no unrelated service signal. Resident model loads uniquePIDstart, raw checkpoint/IO timers nested; cached experiment wall not deployment.',code_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(code)},input_sha256={str(f.relative_to(R)):sha(f)for f in dict.fromkeys(inp)})
 plan['score_input_sha256']={str(f.relative_to(R)):sha(f)for f in scoreinputs};plan['fingerprint']=digest(plan);atomic(D/'plan.json',plan);atomic(P/'protocol.json',{k:v for k,v in plan.items()if k not in ['code_sha256','input_sha256','score_input_sha256']});print(plan['fingerprint'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare_inputs','preflight','freeze']);x=a.parse_args();globals()[x.stage]()
