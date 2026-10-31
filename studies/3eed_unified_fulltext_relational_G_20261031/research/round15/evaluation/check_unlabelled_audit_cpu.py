"""Synthetic recorded-array checks, no G solver/model/sensor or real outputs."""
from pathlib import Path
import copy,hashlib,json
import numpy as np
import audit_direct_unlabelled as a
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def must_reject(fn):
 try:fn()
 except AssertionError:return
 raise AssertionError('Deliberate contract violation not rejected')
def main():
 out=D/'UNLABELLED_AUDITOR_CPU_FIXTURE.json'
 if out.exists():raise FileExistsError(out)
 anchor=np.array([0,0,10,2,2,2],np.float32);boxes=np.tile(anchor,(4,1));boxes[1,0]=.1;boxes[2,0]=.05;boxes[3]=anchor+np.float32(.5)*(boxes[1]-anchor)
 active=np.ones((4,6),bool);active[0]=False
 g={'arm_boxes_raw6':boxes.tolist(),'arm_query_index':[1]*4,'actual_changed_axes':(boxes!=anchor).tolist(),'active_mask6':active.tolist(),'acceptance_mask6':active.tolist(),'point_support_sha256':['a'*64]*4,'mask_sha256_by_arm':['b'*64]*4,'arm_status':['source','converged_bounded_result','converged_bounded_result','fixed_half_of_direct_full'],'diagnostics':{'fixed_q0':1,'same_support_gate_active_bounds_full_noRGB':True,'config':{'center_bound_fraction':.25,'center_bound_cap_m':.5,'max_size_expansion':1.2}}}
 a.geometry_arrays(g,anchor,1);checks={'synthetic_valid_center_bounds_half_and_same_support':True}
 bad=copy.deepcopy(g);bad['arm_boxes_raw6'][1][0]=1.;must_reject(lambda:a.geometry_arrays(bad,anchor,1));checks['out_of_bound_center_rejected']=True
 bad=copy.deepcopy(g);bad['arm_boxes_raw6'][1][3]=2.7;must_reject(lambda:a.geometry_arrays(bad,anchor,1));checks['size_above1point2_rejected']=True
 bad=copy.deepcopy(g);bad['arm_status'][1]='source_fallback:synthetic';must_reject(lambda:a.geometry_arrays(bad,anchor,1));checks['failed_geometry_must_return_same_anchor']=True
 bad=copy.deepcopy(g);bad['arm_query_index'][2]=0;must_reject(lambda:a.geometry_arrays(bad,anchor,1));checks['geometry_must_keep_q1_not_q0']=True
 pool=np.tile(anchor,(256,1));scores=np.zeros(256);scores[0]=1.;scores[1]=.5;scores[2]=.3
 count=np.zeros((256,1),int);count[1:3]=8;pos=count>=8
 groups=[[1,2]]+[[q] for q in range(256) if q not in (1,2)];member=np.empty(256,int)
 for j,qs in enumerate(groups):member[qs]=j
 mapping={'inside_exclusive_counts':count.tolist(),'foreign_exclusive_counts':np.zeros_like(count).tolist(),'pollution_rejected':np.zeros_like(count,bool).tolist(),'positive_feasible':pos.tolist(),'feasible':pos.tolist(),'groups':[{'query_indices':x} for x in groups],'membership':member.tolist(),'mask_mappings':[{'mask_id':'M00001','canonical_query_index':1}]}
 branch={'semantic_proposed_target_mask_id':'M00001','association_accepted':True,'mapping':mapping,'q_association':1,'selected_target_mask_id':'M00001'};sem={'selected_target_mask_id':'M00001'};meta={'objects':[{'mask_id':'M00001','n_exclusive':8}]}
 a.mapping_readout_check(branch,sem,meta,pool,scores);checks['all256_groups_readout_uses_originalscore_inside_unique_group_not_q0']=True
 bad=copy.deepcopy(branch);bad['q_association']=2;must_reject(lambda:a.mapping_readout_check(bad,sem,meta,pool,scores));checks['wrong_canonical_member_rejected']=True
 bad=copy.deepcopy(branch);bad['mapping']['inside_exclusive_counts'][1][0]=9;must_reject(lambda:a.mapping_readout_check(bad,sem,meta,pool,scores));checks['count_above_exclusive_support_rejected']=True
 # Exercise the actual eight-arm reader, including complete-NA fallback and
 # the nonapplicable flip bridge, without running any G solver.
 source=np.tile(anchor,(8,1));zero=np.zeros((8,6),bool);queries=np.zeros(8,np.int64)
 branch0={'arm_boxes_raw6':np.tile(anchor,(5,1)).tolist(),'arm_query_index':[0]*5,'q_association':0,'arm_status':['source']+['source_fallback:synthetic']*4,'association_accepted':False}
 trace={'geometry_trace':{'branches':{k:copy.deepcopy(branch0) for k in ('proposal','graph','relation_disabled','relation_flip')},'relation_flip_applicable':False,'nonapplicable_flip_exact_fullgraph_array_reuse':True}}
 data={'arm_names':np.asarray(a.ARMS),'q_source':np.asarray(0,np.int64),'arm_query_index':queries,'arm_boxes_raw6':source,'arm_boxes_eval6':a.evalcopy(source),'source_boxes_raw6':pool,'source_text_scores256':scores,'arm_geometry_anchor_raw6':source.copy(),'actual_changed_axes':zero,'geometry_increment_axes':zero.copy(),'arm_support_sha256':np.asarray(['0'*64]*8),'arm_mask_sha256':np.asarray(['0'*64]*8),'arm_association_accepted':np.zeros(8,bool),'arm_status':np.asarray(['source']+['source_fallback:synthetic']*7)}
 a.audit_arrays(data,pool,scores,trace);checks['actual_eight_arm_reader_allNA_keeps_all_source_and_flip_bridge']=True
 result={'state':'PASS_R15_UNLABELLED_AUDITOR_SYNTHETIC_RECORDED_ARRAY_CONTRACTS_ONLY','checks':checks,'check_count':len(checks),'auditor_sha256':sha(D/'audit_direct_unlabelled.py'),'fixture_sha256':sha(__file__),'actual_predictions_semantics_captures_GT_model_GPU_sensor_Gsolver_or_taskIoU_read_or_executed':False}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':len(checks),'sha256':sha(out)}))
if __name__=='__main__':main()
