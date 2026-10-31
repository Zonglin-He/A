"""Array-only R15 8-arm composition and pre-execution plan guards.

No real input registries, sealed predictions, GT, sensor files, model or GPU.
"""
from pathlib import Path
import argparse,hashlib,json,sys,tempfile
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_geometry_cpu import fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from crossplatform.round15.assemble_direct import compose_direct_arrays,prepare,ARM_NAMES,sha

def full(mid='M00020',mode='literal'):
 return {'status':'unique_supported','selected_target_mask_id':mid,'search_complete':True,'relation_mode':mode,'literal_relations_satisfied':mode=='literal','relation_check':mode!='disabled'}
def semantics(mid='M00020'):
 g=full(mid);f={**g,'relation_mode':'flip_diagnostic','flip_applied':False,'nonapplicable_policy':'exact_original_fullgraph_output_not_source_cancellation'}
 return {'proposal_consensus':{'status':'proposal_control_unique','selected_target_mask_id':mid,'search_complete':True,'relation_mode':'proposal_control'},'graph_match':g,'graph_relation_disabled':full(mid,mode='disabled'),'graph_relation_flip':f}
def pool256(boxes,scores):
 rest=np.array([[100.+j,0,10,1,1,1] for j in range(256-len(boxes))],np.float32)
 return np.r_[boxes,rest],np.r_[scores,np.full(len(rest),-100.)]
def check():
 c={};_,_,scene,small,ss=fixture();pool,scores=pool256(small,ss);q=int(scores.argmax())
 a=compose_direct_arrays(scene=scene,source_pool=pool,text_scores=scores,q_source=q,semantics=semantics())
 assert tuple(a['arm_names'])==ARM_NAMES and a['arm_boxes_raw6'].shape==(8,6) and a['arm_boxes_raw6'].dtype==np.float32
 assert a['arm_query_index'][0]==q and np.array_equal(a['arm_boxes_raw6'][0],pool[q])
 assert a['arm_query_index'][3]==a['arm_query_index'][4]==a['arm_query_index'][5]==2
 assert np.array_equal(a['arm_boxes_raw6'][3],pool[2]) and np.array_equal(a['arm_support_sha256'][3:6],np.repeat(a['arm_support_sha256'][3],3))
 assert np.array_equal(a['arm_mask_sha256'][3:6],np.repeat(a['arm_mask_sha256'][3],3))
 assert a['trace']['nonapplicable_flip_exact_fullgraph_array_reuse'] and np.array_equal(a['arm_boxes_raw6'][4],a['arm_boxes_raw6'][7])
 c['eight_literal_arms_all_direct_common_source_and_assoc_geo_q1_support_bridge']=True
 c['nonapplicable_flip_keeps_literalG_exact_no_source_cancellation']=True
 changed=semantics();changed['graph_relation_flip']={**full('M00010',mode='flip_diagnostic'),'flip_applied':True}
 b=compose_direct_arrays(scene=scene,source_pool=pool,text_scores=scores,q_source=q,semantics=changed)
 assert b['arm_query_index'][7]==0 and not b['trace']['nonapplicable_flip_exact_fullgraph_array_reuse']
 assert np.array_equal(a['arm_boxes_raw6'][:7],b['arm_boxes_raw6'][:7])
 c['active_relation_flip_only_diagnostic_arm_changes']=True
 bad=semantics();bad['graph_relation_flip']['selected_target_mask_id']='M00010'
 try:compose_direct_arrays(scene=scene,source_pool=pool,text_scores=scores,q_source=q,semantics=bad)
 except ValueError:pass
 else:raise AssertionError('Nonapplicable flip silently changed identity')
 c['nonapplicable_mismatched_flip_rejected']=True
 # Distinct selected query q1 accepted, geometric corner projection invalid.
 failpool=pool.copy();failpool[1,5]=30;failpool[2,5]=30
 f=compose_direct_arrays(scene=scene,source_pool=failpool,text_scores=scores,q_source=q,semantics=semantics())
 assert np.all(f['arm_query_index'][3:6]==2) and np.array_equal(f['arm_boxes_raw6'][3:6],np.tile(failpool[2],(3,1)))
 assert all('projection' in str(v) for v in f['arm_status'][4:6])
 c['geometry_fail_delivers_associated_anchor_in_all_main_coordinate_controls']=True
 # Every missing/unknown branch remains in complete eight-arm output.
 unknown={'status':'NA_unknown','selected_target_mask_id':None,'search_complete':False,'relation_mode':'literal'}
 uk={'proposal_consensus':unknown,'graph_match':unknown,'graph_relation_disabled':unknown,'graph_relation_flip':{**unknown,'flip_applied':False,'nonapplicable_policy':'exact_original_fullgraph_output_not_source_cancellation'}}
 z=compose_direct_arrays(scene=scene,source_pool=pool,text_scores=scores,q_source=q,semantics=uk)
 assert np.array_equal(z['arm_boxes_raw6'],np.tile(pool[q],(8,1))) and not z['arm_association_accepted'].any()
 c['allNA_cases_keep_eight_source_fallbacks_no_filtering']=True
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'plan.json';p.write_text(json.dumps({'stage':'R15_direct8_from_sealed_semantics','GT_read_permitted':True,'CPU_geometry_execution_permitted':True,'model_execution_permitted':False}))
  try:prepare(SimpleNamespace(plan=p,plan_sha256=sha(p),output=Path(tmp)/'out'))
  except ValueError as e:assert 'scope' in str(e)
  else:raise AssertionError('GT permitted plan entered assembler')
  assert not (Path(tmp)/'out').exists()
 c['actual_plan_scope_rejects_before_any_real_input_or_output']=True
 assert 'torch' not in sys.modules and 'transformers' not in sys.modules
 c['CPUimport_does_not_construct_or_import_neural_runtime']=True
 return {'state':'PASS_R15_PURE_SYNTHETIC_EIGHT_ARM_COMPOSITION_AND_NOEXECUTION_GUARDS','checks':c,'check_count':len(c),'actual_files_GT_scores_raw_sensors_predictions_model_or_GPU_read':False,'program_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['assemble_direct.py','check_assembly_cpu.py']}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':r['state'],'checks':r['check_count']}))
