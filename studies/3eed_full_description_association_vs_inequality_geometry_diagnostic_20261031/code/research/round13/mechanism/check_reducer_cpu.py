"""Synthetic reducer structures only; no actual A/B partial or sensor payloads."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reduce_unlabelled as r


def check():
 source=np.array([0,0,5,2,2,2],np.float32)
 abox=np.tile(source,(5,1));abox[1,0]=.1
 aq=np.array([0,1,0,0,0],np.int64)
 ap={'arm_names':np.array(r.A_ARMS),'arm_boxes_raw6':abox,'arm_query_index':aq,'source_q0':np.asarray(0)}
 cs={};cons={}
 for cond in r.CONDITIONS:
  cs[cond]=[{'parsed':{'parse_status':'valid_candidate_ID' if k==0 else 'explicit_UNKNOWN'},
      'raw_answer_PRIVATE':'C001' if k==0 else 'UNKNOWN','input_token_n':100,'generated_token_n':2,
      'image_visualtoken_n':[4,4],'ended_with_eos':True,'max_new_tokens':12,'seconds_model_generate':.5} for k in range(2)]
  cons[cond]={'two_numberings_same_physical_mask':False,'selected_mask_index':None,
      'selected_mask_index_by_order':[0,None],'readout_status':'fallback_q0_two_numberings_disagree_or_UNKNOWN',
      'q_selected':0,'query_changed':False}
 at={'mapping':{'groups':[{'group_id':0,'query_indices':list(range(256))}],
       'membership256':[0]*256,'source_group':0,
       'masks':[{'mapping_status':'mapped_unique_geometry_hypothesis','eligible_group_n':1},
                {'mapping_status':'NA_no_eligible_querygroup','eligible_group_n':0}]},
       'calls':cs,'consensus':cons,'full_order0_raw':{'readout_status':'mapped_original_query','selected_mask_index':0},
       'deranged_same_caption_as_full':False}
 aa=r.reduce_A(at,ap)
 assert aa['group_n']==1 and aa['mapped_mask_n']==1 and len(aa['calls'])==6
 assert aa['arms']['full_order0_raw']['box_changed'] and not aa['arms']['full_two_numbering_consensus']['box_changed']
 raw=np.tile(source,(5,1));raw[1,0]=.2;raw[3]=source+np.float32(.5)*(raw[1]-source)
 bp={'arm_names':np.array(r.B_ARMS),'arm_boxes_raw6':raw,'arm_query_index':np.zeros(5,np.int64),
      'q_source':np.asarray(0),'active_coordinate_mask':np.ones((5,6),bool),
      'actual_changed_axes':raw!=source,'arm_status':np.array(['source','finite_nonconverged_bounded_result','source_already_satisfies_positive_constraints','fixed_half_of_direct_full','source'])}
 init={'point_containment':0.,'RGB_positive_containment':1.,'point_violation_fraction':0.,'RGB_boundary_violation_fraction':.5}
 bt={'support':{'segmentation_valid':True,'status':'unique','diagnostics':{'selected_unique_model_XYZ':8,'candidates':[0],'visible_in_source_ROI':10}},
     'inequality':{'gate_passed':True,'gate_reason':'shared','initial_terms':init,
       'solver':{'direct_full':{'success':False,'iterations':64,'finite_nonconverged_retained':True},'direct_no_rgb':{'success':True,'iterations':0}}}}
 bb=r.reduce_B(bt,bp)
 assert bb['initial_violation_class']=='RGB_only_violated'
 assert bb['arms']['B_inequality_half']['half_derived_from_nonconverged_full']
 rows=[{'case_role':'new_content108','platform':'drone','A':aa,'B':bb},
       {'case_role':'regression8','platform':'quad','A':aa,'B':bb}]
 summary=r.summarize(rows)
 assert len(summary)==4 and all(s['n']==1 for s in summary)
 assert summary[0]['A_association']['conditions']['full']['call_denominator']==2
 assert summary[0]['B_geometry']['shared_gate_passed_n']==1
 cost=r.call_cost([aa]);assert cost['n_calls']==6 and cost['ended_with_eos_n']==6 and cost['model_generate_seconds']['sum']==3.
 # An NA gate retains full case denominator and missing initial constraint terms.
 nt={'support':{'segmentation_valid':False,'status':'NA_multiple','diagnostics':{'candidates':[0,1]}},
     'inequality':{'gate_passed':False,'gate_reason':'invalid_or_empty_independent_mask'}}
 raw0=np.tile(source,(5,1));np0=dict(bp,arm_boxes_raw6=raw0,actual_changed_axes=raw0!=source)
 na=r.reduce_B(nt,np0);assert na['point_already_satisfied'] is None and not na['min8_met']
 checks={'all256_group_coverage_and_mapping_counts':True,'six_strict_parse_calls_and_consensus_fallback':True,
       'raw_query_change_separate_from_consensus':True,'B_RGB_only_violation_class':True,
       'half_no_new_solver_nonconvergence_lineage':True,'NA_constraints_not_counted_satisfied':True,
       'new_and_regression_domain_denominators_separate':True,'actual_call_token_EOS_cost_reduction':True}
 return {'state':'passed_synthetic_reducer_structure_checks','checks':checks,
     'program_sha256':{str(p.relative_to(r.ROOT)):r.sha(p) for p in [Path(r.__file__),Path(__file__)]},
     'actual_A_partial_B_GT_score_sensor_model_GPU_accessed':False}


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=check()
 if a.output.exists():raise FileExistsError(a.output)
 r.dump(a.output,out);print(json.dumps({'state':out['state'],'n_checks':len(out['checks'])}))
