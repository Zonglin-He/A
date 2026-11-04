from common import *
import argparse

def prepare():
 assert not(D/'plan.json').exists();torch.set_num_threads(1);cases=[]
 for ds in COUNTS:
  for r in rows(ds)[:2]:
   a,p=derive(ds,r);b,q=derive(ds,r,reference=True);assert p==q and all(np.max(abs(v-b[k]))<2e-10 for k,v in a.items());cases.append(dict(dataset=ds,raw8_E_consensus_and_Full_DP_and_plain_linear_independent_math_passed=True))
 code=list(S.glob('*.py'))+[R/'studies/pure_linear_standalone_v1/pure.py',R/'stvg/initialization.py',R/'stvg/consensus.py',R/'stvg/predictor.py',R/'stvg/spatial.py',R/'stvg/metrics.py',R/'tools/full_score.py'];inputs=[CHECK/'teacher_inventory.json',R/'configs/default.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
 for ds in COUNTS:inputs += [CHECK/ds/'sealed.json',old.selected.PREV/'spatial'/f'{ds}_sealed.json']
 p=dict(schema='independent_full_three_path_shared_time_spatial_recalculation_v2',counts=COUNTS,authorization='2026-11-04 user asks first independently redo full shared-time spatial comparison before expert-guided spatial-decoder TTFT baseline',paths=ARMS,one_shared_time='Immutable actual current Full interval per query; checked against current Full reference, never legacy Source/MBR alias. All tIoUs identical.',PureExpertLinear='Independent plain raw8pixelcorner/time/video-frame function; all inputsegment frames/nearest raw tails/exactanchors. No E/features/posterior/candidates/DP input.',SharedCorrection='Actually reconstruct singleton consensus residual from native Source/raw8/source-calibration/gamma; no boxtransport orDP.',Full='Actually execute exact final intervalDP from immutable exact sufficient-state main candidates/unary/link, raw8/context/nativeSource/E match, all boxes bitwise previouscurrentFull. No new full-feature transport fit claimed.',metrics='Official dense v/t/R03/R05 plus event-unobserved IoU on EXACT SAME annotated frames inside common predicted interval and GT support, excluding raw8physical expert observations; common frame/querycounts for all3. Do not mix dense versus native unobserved supports or all-video versus inside-event groups. Empty support null; v0 rows retained.',independent_audit='Cold all-input SHA/parent/physicaltime/raw8; E independent256subset solver; Full independent NumPy DP; plain linear independent bracket formula; independent normalized denseIoU/means.',original_Full_hardanchor_issue_unchanged=True,old_method_or_paper_or_frozen_files_changed=False,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_optimizer=0,whole_domain_seal_before_officialscore=True,source_TTFT_GPU_gate='Only after this priority full comparison complete/root interpreted',stats='queryequal10000parentbootstrap seed20261022, all signed/severeuniononce>5pp/rescue/short<=5s-long, pointwise exposed/no target tuning',code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'});atomic(P/'preflight.json',dict(cases=cases,new_model=0,new_GT_score=0));print(p['fingerprint'],flush=True)

def predict(ds):
 p=validate();old.infer_guard();torch.set_num_threads(1);start=time.perf_counter();dest=D/ds;assert not(dest/'sealed.json').exists();files={}
 for i,r in enumerate(rows(ds),1):
  resources(start);a,proof=derive(ds,r);f=dest/'predictions'/(r['id']+'.npz');f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists()
  with f.open('wb')as out:np.savez_compressed(out,**a)
  with np.load(f)as z:assert all(np.array_equal(v,z[k])for k,v in a.items())
  rec=dict(path=str(f.relative_to(D)),sha256=sha(f),proof=proof,worker=identity(),GT_read=False);rp=dest/'receipts'/(r['id']+'.json');atomic(rp,rec);files[r['id']]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
  if i%100==0:atomic(dest/'progress.json',dict(completed=i,total=COUNTS[ds],GT_read=False,worker=identity()))
 atomic(dest/'sealed.json',dict(files=files,queries=len(files),GT_read=False,worker=identity(),fingerprint=p['fingerprint'],elapsed_wall_seconds=time.perf_counter()-start));print(ds,len(files),'sealed',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','predict']);a.add_argument('--dataset',choices=COUNTS);x=a.parse_args();globals()[x.stage](*([]if x.stage=='prepare'else[x.dataset]))
