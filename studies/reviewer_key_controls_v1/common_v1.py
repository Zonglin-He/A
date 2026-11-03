"""Independent reviewer controls; old studies/configurations remain immutable."""
import os,sys,json,hashlib,time,csv,shutil,argparse
from pathlib import Path
from functools import lru_cache
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/shared_corrected_roi_v1'),str(R),str(R/'tools')]
import shared_common as selected
read=selected.read;sha=selected.sha;atomic=selected.atomic;digest=selected.digest;identity=selected.identity
fixed=lru_cache(None)(read)
D=R/'outputs/reviewer_key_controls_v1';P=R/'results/reviewer_key_controls_v1'
COUNTS={'hc2':1902,'vidstg':10303}
ARMS_A=['Source','SourceJointMAP','SourceTR','ExpertInitTR','Ours','FixedE_SourceTR','Interpolation_SourceTR','Interpolation_ExpertTR']
ARMS_B=['Source','SourceROI','CorrectedROI','GlobalAppearance','BackgroundAppearance','ShuffledBoxes']
PAIRS_A=[('SourceTR','Source'),('SourceTR','SourceJointMAP'),('ExpertInitTR','SourceTR'),('ExpertInitTR','FixedE_SourceTR'),('Ours','ExpertInitTR'),('Ours','SourceTR'),('Ours','Interpolation_SourceTR'),('Interpolation_ExpertTR','Interpolation_SourceTR'),('Ours','Interpolation_ExpertTR')]
PAIRS_B=[('CorrectedROI',a)for a in ARMS_B if a!='CorrectedROI']+[(a,'SourceROI')for a in ['GlobalAppearance','BackgroundAppearance','ShuffledBoxes']]
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
def deny_gpu_media(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
  x=os.fsdecode(args[0]);assert not x.startswith('/dev/nvidia')and Path(x).suffix.lower()not in ['.mp4','.avi','.mov'],x
def infer_guard():
 from stvg.runtime import guard
 sys.addaudithook(guard);sys.addaudithook(deny_gpu_media)
def validate():
 p=read(D/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for name in ['code_sha256','input_sha256']:
  for f,h in p[name].items():assert sha(R/f)==h,f
 return p
def rows(task,ds):
 rr=selected.records(ds)
 if task=='regions':
  keys=set(fixed(R/'outputs/review_response_fast_v1/cohort_v1.json')[ds]);rr=[r for r in rr if r['id']in keys];assert len(rr)==192
 return rr
def source(ds,row):
 root=selected.PREV/'spatial';entry=fixed(root/f'{ds}_sealed.json')['files'][row['id']];c,proof=selected.checked_payload(root,entry)
 assert not c['GT_read']and c['parent_capture_sha256']==entry['parent_capture_sha256']
 assert entry['input_binding']==selected.binding(row,fixed(selected.RUN/'plan.json')['provider_runtime_fingerprint'])
 return c,proof
def prepare():
 assert not(D/'plan_v1.json').exists();inputs=[R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',R/'outputs/review_response_fast_v1/cohort_v1.json',selected.PREV/'raw_reuse_index.json']
 for ds in COUNTS:
  inputs += [selected.PREV/'spatial'/f'{ds}_sealed.json',selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json',R/f'outputs/review_strengthening_v1/matched_cpu/{ds}/reporting_receipt.json']
  j=read(selected.RUN/'plan.json')['jobs'][ds];inputs += [selected.PROVIDER/j['manifest'],selected.PROVIDER/j['labels']]
 p=dict(schema='reviewer_key_controls_cached_cpu_v1',authorization='2026-11-03 all pasted reviewer experiments; GPU missing-cache work allowed only after prior GPU task finishes. CPU first; no paper/profile changes; no EdgeTAM extra expert.',tasks=dict(temporal='Full HC1902/Vid10303: original Source, joint-readout control, no-expert SourceROI TR with original boxes; expert E-ROI TR with E boxes, Ours final DP; fixed-E SourceTR and expert TR; original interpolation with each own TR interval.',regions='Retained192/domain only: same main spatial geometry and q0, independently refit temporal prototypes/posterior/legalMAP; primary fixed same E trajectory. SourceROI/E-ROI/all-valid-global/outside expanded-E background/SHA fixed box permutation. No model or new acquisition.'),params=dict(kappa=0,beta=1,context_scale=1.5,expert_positions=8,rounds=1,first_auxiliary_DP=False),background='valid native grid support minus original expanded E ROI weights, pooled directly; empty complement uses all valid grid and is counted, no row dropped',shuffle='SHA256 UTF8 20261103|dataset|query_id|source_index order, index tie break, identity N>1 leftrotate1; permute whole E boxes, preserve exact box multiset and sizes; fixed E output path and spatial geometry are never shuffled',statistics='Query equal,10000 parent-cluster percentile95% bootstrap seed20261022; short<=5s GT duration/long; signed severe t/v loss>5pp union once/rescues; negative10pp supplementary. Previously exposed targets, no new unknown confirmation or full-label parameter selection.',prediction_labels=False,whole_each_task_domain_seal_before_separate_official_score=True,separate_all_arrays_cold_rederive_SHA_means_audit=True,new_Source=0,new_expert=0,new_optimizer=0,new_model=0,GPU=False,resources='max2 own low priority CPU14-15 nice19 idleIO thread1 CUDAhidden, RAM8GiB/disk80GiB/private160GiB/domain72h; do not signal unrelated services',full_regions_and_full_uniform_gamma='new independent GPU plan pending raw inventory; not represented by retained192',code_sha256={str(f.relative_to(R)):sha(f)for f in [S/'common_v1.py',S/'core_v1.py',S/'run_v1.py',S/'execute_v1.py',R/'stvg/prototype.py',R/'stvg/representation.py',R/'stvg/spatial.py',R/'stvg/current_shared_roi.py',R/'tools/full_score.py']},input_sha256={str(f):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(D/'plan_v1.json',p);atomic(P/'protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])

