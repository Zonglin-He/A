"""Read-only paired audit of already scored Strong/Global6 HC64.

No model/head/RGB/decoder/expert/scoring/label access. Reaggregate saved metrics
and inspect content-bound cached ERef box extent as a descriptive size proxy.
"""
import os,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np,torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);os.nice(15)
 fp=ROOT/'results/tube_fusion_hc64/report.json';cp=ROOT/'results/joint_boundary_readout_v2/report.json';r=read(fp);pool=read(cp);rows=r['queries_anonymous'];candidates=[x for x in pool['queries_anonymous']if x['cohort']=='HC64'];lock=read(ROOT/'outputs/tube_temporal_hc64/lock.json');assert len(rows)==len(candidates)==len(lock['ids'])==64
 B=np.asarray([x['values']['Global6']['t']for x in rows]);S=np.asarray([x['values']['Strong_Reference']['t']for x in rows]);best=np.asarray([x['candidate_best_tIoU']for x in candidates]);assert np.allclose(B,[x['values']['Baseline']['t']for x in candidates],atol=1e-12,rtol=0)
 details=[];areas=[];grids=[];sourcehashes={str(fp.relative_to(ROOT)):sha(fp),str(cp.relative_to(ROOT)):sha(cp)}
 for i,(item,row)in enumerate(zip(lock['ids'],rows)):
  receipt=read(ROOT/'outputs/tube_temporal_hc64/receipts'/f'{item}.json');path=ROOT/'outputs/tube_temporal_hc64'/receipt['capture_path'];assert sha(path)==receipt['capture_sha256'];cap=torch.load(path,map_location='cpu',weights_only=False);hw=tuple(cap['temporal_cache']['offsets'][0]['grid_size']);E=np.asarray(cap['E']);area=float(np.median(E[:,2]*E[:,3]));areas.append(area);grids.append(hw)
  base=row['values']['Global6'];strong=row['values']['Strong_Reference'];details.append(dict(query=row['query'],baseline=base,Strong=strong,delta_tIoU_pp=100*(strong['t']-base['t']),delta_vIoU_pp=100*(strong['v']-base['v']),candidate_best_tIoU=float(best[i]),ERef_median_normalized_box_area=area,feature_grid=list(hw)))
 groups={}
 for name,mask in dict(GT_event_lt5_seconds=np.asarray([x['values']['Global6']['GT_length_seconds']<5 for x in rows]),GT_event_5to10_seconds=np.asarray([5<=x['values']['Global6']['GT_length_seconds']<10 for x in rows]),GT_event_ge10_seconds=np.asarray([x['values']['Global6']['GT_length_seconds']>=10 for x in rows]),baseline_wrong_tIoU_half=B<.5,baseline_correct_tIoU_half=B>=.5,ERef_area_lower_half=np.asarray(areas)<=np.median(areas),ERef_area_upper_half=np.asarray(areas)>np.median(areas)).items():
  rr=[x for x,m in zip(details,mask)if m];groups[name]=dict(queries=len(rr),delta_tIoU_pp=float(np.mean([x['delta_tIoU_pp']for x in rr])),delta_vIoU_pp=float(np.mean([x['delta_vIoU_pp']for x in rr])))
 paired={}
 for key in ['paired','fixed_spatial_paired']:
  paired[key]={}
  for field,x in r[key]['Global6-Strong_Reference'].items():
   if 'mean'not in x:continue
   paired[key][field]=dict(mean=-x['mean'],CI95=[-x['CI95'][1],-x['CI95'][0]],unit=x['unit'],Strong_higher=x['decreased'],Strong_lower=x['increased'],unchanged=x['unchanged'])
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert len(formal['sources'])==434
 for path,h in formal['sources'].items():assert sha(ROOT/path)==h
 assert not torch.cuda.is_initialized()
 result=dict(status='completed_saved_evidence_analysis',scope='SameHC64 already exposed and scored. No new prediction, model invocation, metric scoring or labels read.',branch_identity='q_Global6=mean_6(q_224_layer);q_Strong=.5*q_448_last+.5*q_Global6; same physical grid, offset merge and original MBR. Strong addition is a conditional 448-posterior intervention; it does not isolate 448-only accuracy or detail from multiscale calibration.',means={a:r['means'][a]for a in ['Global6','Strong_Reference']},paired=paired,groups_posthoc_descriptive=groups,limited_candidate_comparison=dict(Strong_better_than_best13=int((S>best+1e-10).sum()),equal=int((abs(S-best)<=1e-10).sum()),best13_better=int((S<best-1e-10).sum()),best13_minus_Strong_tIoU_pp=float(100*(best-S).mean()),Strong_rescues_half=int(((B<.5)&(S>=.5)).sum()),Strong_harms_half=int(((B>=.5)&(S<.5)).sum()),Strong_rescues_not_reachable_by13=int(((B<.5)&(S>=.5)&(best<.5)).sum())),size_proxy=dict(grids=[list(g)for g in sorted(set(grids))],ERef_median_area=float(np.median(areas)),area_delta_t_spearman=float(spearmanr(areas,100*(S-B)).statistic),warning='ERef normalized full-box size is not GT target-part size; no measured small-object causal attribution.'),examples_largest_gain=sorted(details,key=lambda x:x['delta_tIoU_pp'],reverse=True)[:5],examples_largest_harm=sorted(details,key=lambda x:x['delta_tIoU_pp'])[:3],all_signed_queries=details,proof=dict(source_reports_sha256=sourcehashes,capture_hashes_checked=64,new_GT_reads=0,new_metric_scores=0,new_model_head_GPU_expert_calls=0,formal_sources_unchanged=434),limitations=['Strong improves36,harm22,same6 in tIoU; not uniformly better.','Posterior arrays for a full matched 448-only accuracy/calibration audit are not part of this saved metric comparison.','22 above finite-candidate oracle does not prove all224 hidden/posterior support lacks correct event information.','Posthoc duration/size grouping is descriptive, not new independent confirmation or a deployment gate.'])
 dest=ROOT/'results/strong_vs_global6_audit';dest.mkdir(exist_ok=False);(dest/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['limited_candidate_comparison','size_proxy','proof']},indent=2))
if __name__=='__main__':main()
