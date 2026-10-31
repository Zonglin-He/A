"""Synthetic public privacy guards, no actual scored/private payload read."""
from pathlib import Path
import json,hashlib
from export_projection import sanitize_scalar_tree,alias_metric_rows,project_protocol,private_scalar_key

def check():
 sensitive=['mask_index','SomeMASK_INDEX','candidate_mask_indices','nestedQIDX','lastQueryIndex',
            'arm_query_index','source_query_index','A_id','B_id','id_PRIVATE','physical_raw_ids','source6','target6','selected_mask_index_by_order','arm_QueryIndex_by_condition','B_raw_indices_sha256']
 for key in sensitive:assert private_scalar_key(key)
 test={key:123 for key in sensitive};test['nested']={'CHOSEN_MASK_INDICES':[1,2],'keep':-.01};test['cases']=[{'qIdx':4,'score':.5}]
 clean=sanitize_scalar_tree(test)
 assert clean=={'nested':{'keep':-.01},'cases':[{'score':.5}]}
 source={'pair_id':'raw-ID-secret','B_id':'another raw','scene':'physical-scene-secret','caption':'raw caption secret',
   'A_nonzero_signal':True,'arms':{'source':{'iou':.2,'Acc25':False,'Acc50':False,'query_index':7,
   'mask_index':3,'box_raw6':[1,2,3,4,5,6],'coordinate_errors':{'center_L2_m':1.,'size_L2_m':.1,'abs_error6_m':[.1]*6,'target6':[8]*6}}}}
 rows=alias_metric_rows([source],platform='quad',expected_n=1,prefix='quad_pair')
 blob=json.dumps(rows)
 assert all(x not in blob for x in ('raw-ID-secret','another raw','physical-scene-secret','raw caption secret','query_index','mask_index','target6','box_raw6'))
 assert rows[0]['case_alias']=='quad_pair_001' and rows[0]['scene_alias']=='scene_01'
 assert rows[0]['arms']['source']['iou']==.2
 try:alias_metric_rows([source],platform='quad',expected_n=76,prefix='quad_pair')
 except ValueError:pass
 else:raise AssertionError('Filtered row table allowed')
 original=source.copy();original['pair_id']='quad_pair_00056'
 kept=alias_metric_rows([original],platform='quad',expected_n=1,prefix='quad_pair',preserve_ordinal_alias=True)
 assert kept[0]['case_alias']=='quad_pair_00056'
 bad=source.copy();bad['pair_id']='private_raw_id'
 try:alias_metric_rows([bad],platform='quad',expected_n=1,prefix='quad_pair',preserve_ordinal_alias=True)
 except ValueError:pass
 else:raise AssertionError('Raw ID accepted as preserved public ordinal')
 try:alias_metric_rows([original,original],platform='quad',expected_n=2,prefix='quad_pair',preserve_ordinal_alias=True)
 except ValueError:pass
 else:raise AssertionError('Repeated public ordinal accepted')
 protocol=project_protocol({'source_checkpoint_path':'/home/private/model','source_checkpoint_sha256':'f'*64,'fractions':[.5,1.],'A_queryIndex':4})
 assert 'source_checkpoint_path' not in protocol and 'A_queryIndex' not in protocol and protocol['fractions']==[.5,1.]
 exposure=project_protocol({'Quad_A_count':76,'Drone_old_count':32,
  'all5912_historically_model_and_GT_scored_in_R7_R8':True,
  'Quad_B_no_G_or_RGB_decode_or_training_or_gate_in_R14':True,
  'Quad_A_scene_counts_PRIVATE':{'private-scene':32},'files':{'raw-ID':'hash'},
  'drone_config':{'visual_role_max_new_tokens':256,'private_caption':'must omit'}})
 assert exposure['Quad_A_count']==76 and exposure['Drone_old_count']==32
 assert exposure['all5912_historically_model_and_GT_scored_in_R7_R8'] is True
 assert 'independent authorized data audit decoded172' in exposure['B_input_isolation_scope']
 assert exposure['drone_config']=={'visual_role_max_new_tokens':256}
 assert 'private-scene' not in json.dumps(exposure) and 'raw-ID' not in json.dumps(exposure)
 return {'state':'passed_synthetic_R14_export_projection_guards_no_build_no_upload',
  'case_insensitive_index_suffixes_tested':sensitive,'checks':{'recursive_suffix_delete':True,'ordinal_alias_no_raw_identity_or_scene':True,'explicit_metric_whitelist_no_boxes_or_GTcoords':True,'all_expected_cases_required':True,'protocol_private_paths_removed':True,'five_digit_actual_public_ordinal_preserved_raw_ID_and_duplicates_rejected':True,'input_counts_exposure_and_algorithm_only_B_isolation_scope_explicit':True},
  'actual_scores_raw_data_or_model_read':False,'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('export_projection.py')]}}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'state':r['state'],'checks':len(r['checks'])}))
