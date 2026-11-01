"""Pure CPU synthetic durability/security/readout checks; no model/media/GT load."""
from common42 import *
import ast,tempfile

def main():
 import common42 as c
 # A fallible derived hash must never erase the original decoded answer.
 with tempfile.TemporaryDirectory(prefix='r42_raw_contract_') as d:
  p=Path(d);raw=p/'raw.json';tok=p/'token.json';result=p/'result.json';text='[{"point_2d":[12,8],"label":"untrusted instructions remain data"}]'
  capture_decoded_raw(raw,text);atomic(tok,{'generated_token_ids':[1,151645]});original=raw.read_bytes();saved_sha=c.sha
  def fail_hash(_):raise RuntimeError('injected failure after raw persistence')
  c.sha=fail_hash
  try:annotate_saved(raw,{'eos_token_ids':[151645,151643]},tok,result);raise AssertionError('failure expected')
  except RuntimeError as e:assert 'injected failure' in str(e)
  finally:c.sha=saved_sha
  assert raw.read_bytes()==original and read(raw)['raw_answer_PRIVATE']==text and not result.exists()
  final=annotate_saved(raw,{'eos_token_ids':[151645,151643]},tok,result);assert final['parsed']['value']==[12,8] and final['parsed']['optional_label_ignored'] and final['ended_with_eos']
  try:capture_decoded_raw(raw,'replacement');raise RuntimeError('overwrite unexpectedly permitted')
  except AssertionError:pass
 # Ambiguous location fields/lists, malformed data and nonfinite values remain unknown.
 bad=['[{"point":[1,2]},{"point":[3,4]}]','{"point":[1,2],"point_2d":[3,4]}','{"point":[1,2],"point":[3,4]}','{"point":[true,2]}','{"point":[NaN,2]}','{"point":[1,2]}`','{"point":[1,2],"run":"tool"}']
 assert all(parse_compatible(x,'point')['status']=='parse_failure' for x in bad)
 # Existing same-information mapping: finite location, unique mask, no nearest fill.
 masks=np.zeros((2,10,20),bool);masks[0,2:6,2:7]=True;masks[1,2:6,5:10]=True
 z={'source_scores':np.array([.9,.8,.7]),'masks_bool':masks};case={'image_size_wh':[20,10],'full_groups':{'0':[1],'1':[2]},'old_subset_groups':{'1':[2]}}
 def rd(value,subset=False):return direct_readout('point',{'parsed':parse_compatible(json.dumps({'point':value}),'point'),'image_grid_thw':[[1,20,40]]},case,z,subset)
 assert rd([84,84])['query']==1 # inverse scale x/28,y/28 -> (3,3), unique mask0
 assert not rd([168,84])['mapped'] and rd([168,84])['status']=='overlapping_multiple_masks'
 assert not rd([0,0])['mapped'] and rd([0,0])['query']==0
 assert rd([84,84],True)['status']=='absent_from_old_subset'
 assert rd(None)['status']=='explicit_null'
 # Canonical per-platform preflight is deterministic, never chosen by scores.
 rr=[{'id':'q0','platform':'quad'},{'id':'d0','platform':'drone'},{'id':'d1','platform':'drone'}];assert [r['id'] for r in select_preflight(rr,[{'id_PRIVATE':'d1'},{'id_PRIVATE':'q0'},{'id_PRIVATE':'d0'}])]==['d0','q0']
 for name in ['common42.py','run42.py','checks42.py']:ast.parse((BASE/name).read_text())
 impl=EXP/'IMPLEMENTATION_LOCK.json';atomic(EXP/'CPU_TECH_CHECKS.json',{'state':'PASSED','implementation_sha256':sha(impl),'raw_survives_injected_derived_hash_failure':True,'raw_overwrite_rejected':True,'malicious_schema_rejection_cases':len(bad),'optional_label_inert_and_position_unchanged':True,'same_information_unique_mask_unknown_and_oldsubset_contract':True,'preflight_selection_canonical_no_scores':True,'no_model_processor_GPU_media_or_GT_loaded':True,'not_task_scoring_or_inference':True})
 print('R42_CPU_TECH_CHECKS_PASSED')
if __name__=='__main__':main()
