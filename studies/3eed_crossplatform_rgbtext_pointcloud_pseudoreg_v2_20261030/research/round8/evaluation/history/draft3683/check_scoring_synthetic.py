"""Synthetic R8 q0/q1/geometry/counter fixture, no real model/data/GT/GPU."""
from pathlib import Path
import tempfile,json
import numpy as np
import score_crossdomain as S

def main():
 A=Path(__file__).resolve().parent
 with tempfile.TemporaryDirectory() as td:
  p=Path(td);samples=[];rows=[]
  for platform in ('drone','quad'):
   sid='synthetic_'+platform;z=1.8 if platform=='drone' else 0.;pool=np.tile([100.,100.,z,2.,2.,2.],(256,1)).astype(np.float32);pool[0]=[4,0,z,4,4,4];pool[2]=[.5,0,z,2,2,2];pool[3]=[-.1,0,z,2,2,2];pool[6]=[0,0,z,2,2,2];scores=(1-np.arange(256)*.001).astype(np.float32);q1=np.array([2,0,0]);membership=np.full(256,-1);membership[0]=0;membership[[2,3]]=1;r=np.array([.5,0,0]);raw=pool[q1,:3].astype(float);raw[0,0]=.2;target64=pool[q1].astype(float);target64[0,0]=.35;w=np.zeros((3,6));w[0,0]=1;arms=np.tile(pool[0],(19,1));aq=[0];am=['source'];tr={'G_modes':{}}
   for mi,mode in enumerate(S.MODES):
    anchor=pool[q1[mi]];t32=target64[mi].astype(np.float32);block={};v=anchor.copy()
    for kind in S.KINDS:
     ix=S.ARMS.index(mode+'__'+kind);aq.append(int(q1[mi]));am.append(mode)
     if kind=='association_only':arms[ix]=anchor
     elif kind=='direct_G':arms[ix]=t32
     else:
      n=1 if kind.endswith('1') else 3;v=anchor.copy()
      for step in range(n):v=v+.25*(t32-v)
      arms[ix]=v
    for n in (1,3):block[str(n)]={'trace':[{'parameter_gradient_norm':1.,'nonzero_gradient_parameter_tensors':2,'parameter_delta_max_by_name':{'synthetic_parameter':.01}} for _ in range(n if mi==0 else 0)]}
    tr['G_modes'][mode]={'steps':block}
   fp=p/(sid+'.npz');np.savez_compressed(fp,source_boxes_raw6=pool,source_boxes_eval6=S.eval_box(pool),source_text_scores256=scores,source_native_top10=np.arange(10),q_source=np.array(0),q_association=q1,arm_names=np.asarray(S.ARMS),arm_modes=np.asarray(am),arm_query_index=np.asarray(aq),arm_status=np.asarray(['synthetic']*19),arm_boxes_raw6=arms,arm_boxes_eval6=S.eval_box(arms),modes=np.asarray(S.MODES),raw_consensus_center=raw,r_global=r,pseudo_target6_float64=target64,pseudo_target6=target64.astype(np.float32),pseudo_reliability6=w,dedup_membership=membership,roi_group_ids=np.array([0,1]))
   loaded=S.load_prediction(fp);assert loaded['q0']==0 and loaded['q1'][0]==2
   meta=p/(sid+'_SYNTHETIC.json');S.dump(meta,{'ground_info':[{'bbox_3d':[0.,0.,0.,2.,2.,2.,0.,777.,888.],'class':'must_not_be_used','positive_map':'must_not_be_used'}]});sample={'id':sid,'platform':platform};index={'meta_path':str(meta),'metadata_container_sha256':S.sha(meta),'object_index':0,'scene':'synthetic_'+platform};row,cost,err=S.score_sample(sample,index,loaded,tr);assert err<1e-12;assert row['arms']['rgb_fulltext__association_only']['iou']>row['source_q0_iou'];assert row['arms']['rgb_fulltext__param3']['delta_vs_association']>0;assert row['arms']['rgb_fulltext__param3']['delta_vs_directG']<0;assert row['cluster_oracles_PRIVATE']['rgb_fulltext']['selected_group_best_iou_GTassisted']>row['arms']['rgb_fulltext__association_only']['iou'];rows.append(row)
  summaries=S.summaries(rows);assert any(r['population']=='common_eligible_all_modes' and r['n']==0 and r['mean_iou'] is None for r in summaries)
  rejected=False
  try:S.validate_strict_flags({'cudnn_allow_tf32':True,'matmul_allow_tf32':False,'cudnn_benchmark':True})
  except ValueError:rejected=True
  assert rejected
 result={'state':'synthetic_R8_q0_q1_scoring_contract_pass_no_real_targetGT','scorer_sha256':S.sha(Path(S.__file__)),'fixture_sha256':S.sha(Path(__file__)),'q1_nonargmax_accepted_and_q0_baseline_preserved':True,'q1_shape_prior_not_q0':True,'association_geometry_and_parameter_gains_separate':True,'parameter_vs_direct_negative_preserved':True,'cluster_oracle_offline_targetcoverage_not_instance_identity':True,'trust_targetspace_shrink_actual_target32_bridge':True,'old_R7_precision_rejected':True,'empty_common_mask_null':True,'no_real_model_GPU_data_or_GT_scoring':True};S.dump(A/'SYNTHETIC_SCORING_FIXTURE.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
