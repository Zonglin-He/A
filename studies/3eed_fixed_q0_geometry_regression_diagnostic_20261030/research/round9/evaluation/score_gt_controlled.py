"""R9 oracle-only head-capacity CPU evaluation; never unlabeled TTA.
Uses already-sealed normal GT7D sidecar and oracle13 predictions. No new raw
annotation reads, no model/GPU. Parent separate post-oracle-seal release required.
"""
from pathlib import Path
import argparse, json, math, collections
import numpy as np
from score_normal import sha,read,dump,bounded
from geometry_metrics import eval_box,candidate_ious,coordinate_errors
from capacity_metrics import MASKS,review_capacity_case
GROUPS=('C','S','CS');KINDS=('direct_target','param1','param3','exactfit')
ARMS=('source',)+tuple('GT_'+g+'__'+k for g in GROUPS for k in KINDS)

def main(a):
 assert sha(a.release)==a.release_sha256;g=read(a.release)
 assert g.get('CPU_SCORE_RELEASE') is True and g.get('stage')=='GT_controlled_NOT_TTA' and g.get('normal_sealed_before_GT_injection') is True
 for k,p in [('GT_prediction_seal_sha256',a.predictions/'SEALED.json'),('normal_scored_seal_sha256',a.normal_scored/'SEALED.json'),('source_checkpoint_sha256',a.source_checkpoint),('scorer_program_sha256',Path(__file__)),('capacity_metrics_sha256',Path(__file__).with_name('capacity_metrics.py')),('geometry_metrics_sha256',Path(__file__).with_name('geometry_metrics.py')),('normal_scorer_program_sha256',Path(__file__).with_name('score_normal.py'))]:assert g[k]==sha(p),k
 normal=read(a.normal_scored/'SEALED.json');s=read(a.predictions/'SEALED.json');assert s['n_samples']==normal['n_samples']==160 and s['n_arms']==13
 assert s['state']=='sealed_R9_GTcontrolled_samecache_coordinate_capacity_diagnostic' and s['normal_prediction_seal_sha256']==normal['normal_prediction_seal_sha256'] and s['source_checkpoint_sha256']==g['source_checkpoint_sha256']
 for k in ('cudnn_allow_tf32','matmul_allow_tf32','cudnn_benchmark'):assert s['strict_FP32_flags'][k] is False
 for root,seal in [(a.normal_scored,normal),(a.predictions,s)]:
  for name,digest in seal['files'].items():assert sha(bounded(root,name))==digest
 for name,digest in s['program_sha256'].items():assert sha(bounded(a.program_root,name))==digest
 assert s['all_query_indices_same_normal_q0'] and s['no_GT_yaw_RGB_PC_G_or_backbone_forward_in_GTphase'] and s['GTcontrolled_not_unlabelled_TTA_or_taskIoU_upperbound'] and s['source_model_parameters_buffers_grads_modes_restored']
 for rel,digest in g['geometry_dependencies_sha256'].items():assert sha(bounded(Path(__file__).resolve().parents[3],rel))==digest
 # Numerical GT read starts here, only from independently sealed normal score artifacts.
 side=read(a.normal_scored/'PRIVATE_GT7D_SIDECAR.json');targets=read(a.normal_scored/'GT_CONTROL_TARGET_INDEX_PRIVATE.json')
 assert sha(a.normal_scored/'GT_CONTROL_TARGET_INDEX_PRIVATE.json')==s['target_index_sha256']
 gt={r['id']:r for r in side['rows']};ix={r['id']:r for r in targets};normrows={r['id']:r for r in read(a.normal_scored/'RESULTS_PRIVATE.json')['rows_PRIVATE']};traces={r['id']:r for r in read(a.predictions/'trace.json')};assert set(gt)==set(ix)==set(normrows)==set(traces) and len(gt)==160
 a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'SCORER_LOCK.json',{'state':'all_GTcontrolled_and_normal_score_files_verified_before_derived_GT_read','release_sha256':sha(a.release),'GTseal_sha256':sha(a.predictions/'SEALED.json'),'normal_scored_seal_sha256':sha(a.normal_scored/'SEALED.json'),'oracle_only_NOT_TTA':True})
 rows=[];all_exactfit=True
 for sid,r in gt.items():
  with np.load(a.predictions/(sid+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert tuple(d['arm_names'].tolist())==ARMS and d['arm_boxes_raw6'].shape==(13,6) and np.isfinite(d['arm_boxes_raw6']).all()
  q=int(d['q_source']);assert np.array_equal(d['arm_query_index'],np.full(13,q)) and q==normrows[sid]['q_source']
  source=d['source_boxes_raw6'][q];target=np.asarray(ix[sid]['target6_modelcoords'],np.float32);GT7=np.asarray(r['GT7D_model'],float)
  assert np.array_equal(d['target6_modelcoords'],target) and np.array_equal(target,GT7[:6]) and np.array_equal(d['arm_boxes_raw6'][0],source)
  assert np.array_equal(d['arm_boxes_eval6'],eval_box(d['arm_boxes_raw6'])) and np.array_equal(d['coordinate_residual6_by_arm'],d['arm_boxes_raw6']-d['constructed_target6_by_arm'])
  amap={k:dict(direct='GT_'+k+'__direct_target',param1='GT_'+k+'__param1',param3='GT_'+k+'__param3',exactfit='GT_'+k+'__exactfit') for k in GROUPS}
  arrays=dict(zip(ARMS,d['arm_boxes_raw6']));cap=review_capacity_case(source,target,arrays,amap,exactfit_tolerance=5e-5)
  for k,m in MASKS.items():
   for kind in KINDS:
    j=ARMS.index('GT_'+k+'__'+kind);assert np.array_equal(d['arm_coordinate_masks6'][j],m.astype(np.float32)) and np.array_equal(d['constructed_target6_by_arm'][j],np.where(m,target,source))
   checked=cap[k]['exactfit_within_registered_tolerance'];assert checked==traces[sid]['groups'][k]['exactfit_atol5e_minus5_pass'];all_exactfit&=checked
  vals,cost=candidate_ious(eval_box(d['arm_boxes_raw6']),GT7);assert np.isfinite(vals).all()
  rows.append({'id':sid,'platform':r['platform'],'scene':r['scene'],'identity_diagnostic':normrows[sid]['identity_diagnostic'],'oracle_only_NOT_TTA':True,'capacity':cap,'actual_head_update_trace':traces[sid]['groups'],'arms':{n:{'iou':float(vals[j]),'Acc25':bool(vals[j]>.25),'Acc50':bool(vals[j]>.5),'coordinate_errors':coordinate_errors(d['arm_boxes_raw6'][j],GT7),'status':str(d['arm_status'][j])} for j,n in enumerate(ARMS)}})
 assert bool(all_exactfit)==s['all_exactfit_coordinate_checks_pass']
 summary=[]
 for platform in ('drone','quad'):
  for pop in ('all','annotated_spatially_confirmed'):
   pp=[r for r in rows if r['platform']==platform and (pop=='all' or r['identity_diagnostic']['annotated_spatially_confirmed'])]
   for arm in ARMS:
    n=len(pp);summary.append({'platform':platform,'population':pop,'arm':arm,'n':n,'scene_clusters':len({r['scene'] for r in pp}),'mean_iou':math.fsum(r['arms'][arm]['iou'] for r in pp)/n if n else None,'Acc25':sum(r['arms'][arm]['Acc25'] for r in pp)/n if n else None,'Acc50':sum(r['arms'][arm]['Acc50'] for r in pp)/n if n else None,'center_L2_m_mean':math.fsum(r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in pp)/n if n else None,'size_L2_m_mean':math.fsum(r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in pp)/n if n else None,'oracle_only_NOT_TTA':True})
    if arm!='source':
     group=arm.split('__')[0][3:];kind=arm.split('__')[1];key='direct' if kind=='direct_target' else kind
     summary[-1]['mean_supervised_Linf_m_vs_direct_target']=math.fsum(r['capacity'][group]['controls'][key]['supervised_coordinate_Linf_m'] for r in pp)/n if n else None
     summary[-1]['max_supervised_Linf_m_vs_direct_target']=max((r['capacity'][group]['controls'][key]['supervised_coordinate_Linf_m'] for r in pp),default=None)
 result={'state':'GTcontrolled_samecache_coordinate_capacity_scored_NOT_TTA','n':len(rows),'all_exactfit_within_registered5e5':bool(all_exactfit),'summary':summary,'rows_PRIVATE':rows,'scope':'Groundtruth controls, fixedq0 same normal cached forward. Nativeyaw0; directGT dimensions NOT maximalIoU upperbound. Primary evidence coordinate fit/true gradient/parameterchange/source restoration; no normal method adoption from oracle results.'};dump(a.output/'RESULTS_PRIVATE.json',result);dump(a.output/'SCALAR_SUMMARY.json',{k:v for k,v in result.items() if k!='rows_PRIVATE'});dump(a.output/'SEALED.json',{'files':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()},'n_samples':len(rows),'oracle_only_NOT_TTA':True})
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('predictions','normal-scored','release','source-checkpoint','program-root','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--release-sha256',required=True);main(p.parse_args())
