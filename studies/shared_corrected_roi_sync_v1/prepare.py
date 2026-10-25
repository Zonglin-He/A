"""Freeze independent CPU-only synchronization and repeat real-cache preflight."""
from sync_common import *
import argparse
def prepare():
 assert not(D/'plan.json').exists();selected.validate()
 code=[*S.glob('*.py'),R/'studies/shared_corrected_roi_v1/shared_common.py',R/'studies/no_shrink_main_v1/ns_common.py',R/'studies/map_components_v1/run.py',R/'tools/full_score.py',*sorted((R/'stvg').glob('*.py'))]
 inputs=[selected.D/'plan.json',selected.D/'preflight_receipt.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',R/'outputs/map_components_v1/teacher_index.json',R/'outputs/paper_review_suite_v1/supports_sealed.json',R/'outputs/paper_review_suite_v1/budget_v2_confirm_sealed.json',*[parent.D/f'spatial_{ds}_{suffix}.json'for ds in COUNTS for suffix in ['sealed','score_receipt','reporting_receipt']]]
 plan=dict(schema='shared_corrected_ROI_structural_ownK_sync_v1',authorization='User conditional simplification: synchronize structural controls and own-K budgets to direct shared-corrected-box ROI pooling; first auxiliary DP absent, final DP retained.',science=dict(kappa=0,beta=1,context=1.5,Source='Original TA224 FP32 two-offset nativeMAP hull',new_Source=0,new_expert=0,new_optimizer=0,structural_cohorts=COUNTS,budget='Original fixed128parents/domain, each own Hybrid/Uniform K2/4/8/16 exact teacher context, never subsets of eight.'),structural='Full/Past/OneWay use same saved exact E-ROI features and their respective final geometry. NoShared rebuilds E=Source with uncorrected geometry and saved actual Source-ROI features. AuxIndependent is a redundant identity control after first DP removal; AuxSource uses Source ROIs, AuxE is Full. NoSpatial removes final DP and propagation; NoTemporal restores native MAP hull. E-fixed factorial recomputed, explicit interaction.',budget='Exact original old per-K MBR arrays checked under original Torch4; this compatibility check executes historical auxiliary path only. Selected output then actually repools each K-specific corrected E from raw native grid, updates kappa0 posterior, selects own jointMAP interval and re-solves final DP. Own-K teacher acquisition costs retained. CPU verification wall includes compatibility overhead, not deployment.',gate='Wait initial clean owner actual exit and both successful independent score/audits; max2 compute jobs, no GPU/corruption launch.',limits=dict(CPU_jobs=2,CPU=[14,15],nice=19,threads=1,io='idle3',minimum_RAM=8*2**30,minimum_disk=80*2**30,private_bytes=160*2**30,domain_seconds=259200),evaluation='Whole task/domain sealed before independent CPU official scoring, separate all-array cold replay/SHA/allmeans audit. All signed positives/negatives/severe union/rescues/shortlong/parent95CI/cost; full data historically exposed, no retuning.',failure='Preserve all source/plans/logs/partial predictions; no skip/fallback/hash waiver/blind retry.',corruption_release_allowed=False,code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 plan['fingerprint']=digest(plan);atomic(D/'plan.json',plan);atomic(P/'protocol.json',{k:v for k,v in plan.items()if k!='input_sha256'})
 for f in code:
  p=D/'source_snapshot'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f.read_bytes())
 print(plan['fingerprint'])
def preflight():
 import torch,numpy as np
 from sync_core import derive
 from stvg.runtime import guard
 validate();sys.addaudithook(guard);torch.set_num_threads(1);proof=[]
 for task in ['spatial','budget']:
  for ds in COUNTS:
   item=rows(task,ds)[0];a,m=derive(task,ds,item);b,n=derive(task,ds,item);assert m==n and all(np.array_equal(a[k],b[k])for k in a)
   proof.append(dict(task=task,dataset=ds,query_hash=hashlib.sha256(item['key'].encode()).hexdigest(),repeat_all_arrays_exact=True,proof_digest=digest(m)))
 assert not torch.cuda.is_initialized();rec=dict(status='passed',cases=proof,code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),GT_read=False,new_model=0,new_Source=0,new_expert=0,new_optimizer=0);atomic(D/'preflight_receipt.json',rec);atomic(P/'preflight_receipt.json',rec);print('four real structural/ownK cases repeat exact')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight']);globals()[p.parse_args().command]()
