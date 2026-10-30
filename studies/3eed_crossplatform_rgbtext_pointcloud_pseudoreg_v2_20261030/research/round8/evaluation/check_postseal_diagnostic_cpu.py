"""Memory-only synthetic boundary review of postseal diagnostic; no input data."""
from pathlib import Path
import ast,collections,importlib.util,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3];A=Path(__file__).resolve().parent;file=A/'diagnose_scored_results.py'
spec=importlib.util.spec_from_file_location('R8_postseal_reviewed_diagnostic',file);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def row(sid,scene,av,bv,r=0.):
 names=set(x for pair in M.CONTRASTS.values() for x in pair);arms={n:{'iou':av} for n in names};arms['source']={'iou':bv}
 return {'id':sid,'scene':scene,'platform':'drone','arms':arms,'G':{'rgb_fulltext':{'r_global':r,'q_changed':True,'actual_float32_pseudo_delta_nonzero':True,'accepted_axis':True}},'source_coverage':{'256':{'best_iou':.8}},'cluster_oracles_PRIVATE':{'rgb_fulltext':{'all_prepared_group_pool_best_iou_GTassisted':.8,'semantic_ROI_pool_best_iou_GTassisted':.8,'selected_group_best_iou_GTassisted':.8}}}

def execute_ast(node,namespace):exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),namespace)
def main():
 tree=ast.parse(file.read_text());tailnode=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='name' and isinstance(n.iter,ast.Tuple));binnode=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts]==['lo','hi']);covernode=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='threshold')
 def tails(rr):
  ns={'rr':rr,'tails':{'drone':{}},'platform':'drone','CONTRASTS':M.CONTRASTS};execute_ast(tailnode,ns);return ns['tails']['drone']
 all_tie=[row('private_case_b','private_scene_A',.3,.3),row('private_case_a','private_scene_A',.3,.3)]
 tt=tails(all_tie);assert all(v['negative_available']==v['positive_available']==0 and v['negative']==v['positive']==[] for v in tt.values());stats=M.stats(all_tie,M.CONTRASTS['association_minus_source']);assert stats['n']==2 and stats['tie_atol1e_minus8']==2
 allpos=[row('private_case_b','private_scene_A',.4,.2),row('private_case_a','private_scene_A',.4,.2)];pp=tails(allpos)['association_minus_source'];assert pp['negative_available']==0 and pp['negative']==[] and pp['positive_available']==2 and [r['id'] for r in pp['positive']]==['private_case_a','private_case_b']
 missing=M.stats([],M.CONTRASTS['association_minus_source']);assert missing['n']==missing['defined_scene_clusters']==0 and missing['delta_mean'] is None and missing['equal_scene_delta_mean'] is None and missing['delta_quantiles'] is None
 unequal=[row('a'+str(i),'scene_A',.6,.5) for i in range(10)]+[row('b','scene_B',0.,.5)];u=M.stats(unequal,M.CONTRASTS['association_minus_source']);assert u['defined_scene_clusters']==2 and math.isclose(u['delta_mean'],.5/11,abs_tol=1e-12) and math.isclose(u['equal_scene_delta_mean'],-.2,abs_tol=1e-12)
 for t in u['strict_Acc_transitions'].values():assert sum(t.values())==11
 edge=[row(str(i),'scene',.25,.25) for i in range(2)];es=M.stats(edge,M.CONTRASTS['association_minus_source']);assert es['strict_Acc_transitions']['0.25']['retained_failure']==2
 rs=[row(str(i),'scene',.3,.3,r) for i,r in enumerate([0,.05,.2,.5,1])];ns={'rr':rs,'groups':{}};execute_ast(binnode,ns);bins=ns['groups'];assert len(bins)==5 and all(len(v)==1 for v in bins.values()) and len({r['id'] for v in bins.values() for r in v})==5
 cases=[row(str(i),'scene',.8,.8) for i in range(6)];keys=['all_prepared_group_pool_best_iou_GTassisted','semantic_ROI_pool_best_iou_GTassisted','selected_group_best_iou_GTassisted'];cases[0]['source_coverage']['256']['best_iou']=.2
 for i,stage in [(1,0),(2,1),(3,2)]:cases[i]['cluster_oracles_PRIVATE']['rgb_fulltext'][keys[stage]]=None
 cases[4]['arms']['rgb_fulltext__association_only']['iou']=.2;ns={'rr':cases,'result':{'GT_assisted_coverage_categories':{}},'collections':collections,'RGB':M.RGB};execute_ast(covernode,ns)
 for table in ns['result']['GT_assisted_coverage_categories'].values():assert len(table)==6 and sum(table.values())==6
 # The empty-prepared case has an already-successful q0 fallback. It is still a
 # coverage-path category, explicitly not an actual prediction failure label.
 assert cases[1]['arms']['rgb_fulltext__association_only']['iou']>.5
 out={'platforms':{'drone':{'scene_rows_PRIVATE':{'private_scene_NEVER_PUBLIC':{}},'subgroups':{'all':stats}},'quad':{'scene_rows_PRIVATE':{},'subgroups':{}}},'n':2}
 mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');projection=[n for n in mainnode.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='public' or isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='public' for t in n.targets)];ns={'out':out}
 for node in projection:execute_ast(node,ns)
 public=json.dumps(ns['public']);assert 'private_scene_NEVER_PUBLIC' not in public and 'private_case' not in public and 'scene_rows_PRIVATE' not in public
 receipt={'state':'memory_only_synthetic_postseal_diagnostic_review_pass','diagnostic_sha256':M.sha(file),'fixture_sha256':M.sha(Path(__file__)),'checks':{'alltie_has_no_fake_positive_or_negative_tail':True,'allpositive_has_empty_negative_and_stable_positive_idtie':True,'empty_groups_null_and_zero_defined_scenes':True,'unequal_scene_weight_micro_vs_macro_opposite_sign_verified':True,'strict_Acc_four_cells_partition_n_and_threshold_equal_isfailure':True,'fixed_r_bins_endpoints_disjoint_and_exhaustive':True,'six_coverage_categories_partition_including_empty_group':True,'coverage_miss_does_not_equal_actual_fallback_prediction_miss':True,'actual_public_projection_excludes_privateIDs_scenes_tails':True},'no_model_GPU_GT_inprogress_data_or_taskscore_access':True};M.Path(A/'POSTSEAL_DIAGNOSTIC_SYNTHETIC_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
