"""Independent saved coverage scalar/denominator readback; no raw labels."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import BASE,read,sha,write
def main():
 p=BASE/'POOL_CANDIDATE_COVERAGE_POSTSEAL.json';d=read(p);rows=d['cases']
 assert d['status']=='completed_cpu_postseal' and d['GT_read_after_complete_prediction_seal'] is True
 assert d['bindings']['global_prediction_barrier_sha256']==sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 assert d['bindings']['roster_lock_sha256']==sha(BASE/'ROSTER_LOCK.json')
 assert d['bindings']['tracking_global_barrier_sha256']==sha(BASE/'TRACKING_GLOBAL_BARRIER.json')
 assert d['bindings']['coverage_tool_sha256']==sha(ROOT/'scripts/score_stvg_native_instance_pool_coverage_v1.py')
 assert d['thresholds']==[.25,.5] and len(rows)==8
 scored=read(BASE/'POSTSEAL_SCORE_ROWS.json')['rows'];counts={}
 for r in scored:
  cid=r['case_id'];alias=('HC' if cid.startswith('hc2') else 'V')+f'{int(cid[-3:])+1:02d}'
  count=r['support']['dense_gt_known_frame_count']
  assert alias not in counts or counts[alias]==count
  counts[alias]=count
 checks=0
 def eq(a,b):
  nonlocal checks
  assert abs(a-b)<=2e-12;checks+=1
 for r in rows:
  n=r['known_gt_frames'];assert n==counts[r['alias']] and n>0
  assert r['gt_frames_in_canonical_map']+r['gt_frames_outside_canonical_map']==n
  assert r['gt_frames_with_empty_filtered_pool']==r['canonical_gt_frames_with_empty_filtered_pool']+r['gt_frames_outside_canonical_map']
  for k in ['0_25','0_50']:eq(r['fraction_gt_frames_max_iou_ge_'+k],r['gt_frames_max_iou_ge_'+k]/n)
  assert 0<=r['mean_max_bbox_iou']<=1 and r['gt_frames_max_iou_ge_0_50']<=r['gt_frames_max_iou_ge_0_25']<=n
 for group,part in [('all',rows),('hc2',[r for r in rows if r['dataset']=='hc2']),('vidstg',[r for r in rows if r['dataset']=='vidstg'])]:
  s=d['summary']['all'] if group=='all' else d['summary']['by_dataset'][group];n=sum(r['known_gt_frames'] for r in part)
  assert s['case_count']==len(part)
  for field in ['known_gt_frames','gt_frames_in_canonical_map','gt_frames_outside_canonical_map','canonical_gt_frames_with_empty_filtered_pool','gt_frames_with_empty_filtered_pool','gt_frames_max_iou_ge_0_25','gt_frames_max_iou_ge_0_50']:eq(s[field],sum(r[field] for r in part))
  eq(s['frame_weighted_mean_max_bbox_iou'],math.fsum(r['mean_max_bbox_iou']*r['known_gt_frames'] for r in part)/n)
  eq(s['case_macro_mean_max_bbox_iou'],math.fsum(r['mean_max_bbox_iou'] for r in part)/len(part))
  for k in ['0_25','0_50']:eq(s['fraction_gt_frames_max_iou_ge_'+k],s['gt_frames_max_iou_ge_'+k]/n)
 write(BASE/'ROOT_POOL_COVERAGE_READBACK.json',dict(status='pass',scope='independent saved pool coverage scalar aggregation and source binding readback; no new label read or prediction changes',coverage_sha256=sha(p),appendix_sha256=sha(ROOT/'docs/STVG_NATIVE_INSTANCE_POOL_COVERAGE_APPENDIX.md'),CPU_only=True,raw_GT_read_by_this_reader=False))
 print(dict(status='pass',case_count=8,checks=checks))
if __name__=='__main__':main()
