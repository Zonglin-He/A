from common73 import *
import collections
from parser73 import classify,ALLREL,DIRECTION,text72
from parser_tests73 import main as tests
co=cohort_map();rr=rows();counts=collections.Counter();coverage=[];eligible={};semantic_simple_old=semantic_simple_new=attribute_recovered=0
for r in rr:
 s=r['caption'];c=classify(s);old=text72.ALLREL.search(s);new=ALLREL.search(s);e=DIRECTION.search(s);first=text72.ENT_RE.search(s)
 def req(m):
  end=m.start() if m and first and m.start()>first.end() else len(s)
  return text72.request(s[:end],first.group() if first else '')
 a,b=req(old),req(new);recovered=a['color'] is None and b['color'] is not None;attribute_recovered+=recovered
 counts[c['kind']]+=1;counts['UNKNOWN_reason_'+c['reason']]+=c['kind']=='UNKNOWN';span_repaired=bool(e and new and e.span()==new.span() and (old is None or e.span()!=old.span()));counts['shared_direction_span_recovered']+=span_repaired
 if c['kind']=='ENTITY_RELATIVE':
  semantic_simple_new+=1;semantic_simple_old+=bool(e and old and e.span()==old.span())
 coverage.append({'row':'PARSE_'+str(len(coverage)+1).zfill(5),'cohort':co[r['id']],'platform':r['platform'],'kind':c['kind'],'reason':c['reason'],'target_color_recovered':recovered,'direction_span_repaired':span_repaired})
 if co[r['id']] not in ['VAL1280','TRAIN1024'] or c['kind']!='IMAGE_ABSOLUTE':continue
 a0=prep(r)
 if np.count_nonzero(a0['rect_valid']&(a0['project_reliability']>=.99))<2:continue
 eligible.setdefault((co[r['id']],r['platform']),[]).append(r)
selected=[];cells=[]
for cohort in ['VAL1280','TRAIN1024']:
 for platform in ['drone','quad']:
  pool=eligible.get((cohort,platform),[]);chosen=sorted(pool,key=lambda r:hashlib.sha256(('R73_mechanism64_v1|'+r['id']).encode()).hexdigest())[:16];selected+=chosen;cells.append({'cohort':cohort,'platform':platform,'eligible':len(pool),'selected':len(chosen)})
assert len(selected)==64
for f in ['input','experiment']:(H/f).mkdir(parents=True,exist_ok=True)
dump(H/'input/MECHANISM64_PRIVATE.json',selected);dump(H/'PARSE_COVERAGE_ROWS.json',coverage);dump(H/'PARSE_COVERAGE_SUMMARY.json',{'n':len(rr),'counts':dict(counts),'simple_entity_old':semantic_simple_old,'simple_entity_new':semantic_simple_new,'target_color_requests_recovered':attribute_recovered,'cells':cells,'no_GT_or_source_error_selection':True});dump(H/'PARSER_TESTS.json',tests())
p={'source':'Waymo_only_epoch100','n':64,'cohort_cell_quota':16,'source_boxes':'original_A256_frozen','matching_parameters':184960,'arms':ARMS,'explicit_image_position_only':True,'relative_entity_position_not_new_supervision':True,'absolute_constraint':'entire_original_box_projection_in_opposite_half;center0.5;margin0.05;all_group_members_valid_and_opposite','unknown_projection_policy':'all_8_corners_front;full_rectangle_inside_image;no_clipping','grouping':'R71_R70_existing_map_no_regrouping','new_position_path_requires_color_support':False,'parser_only_path':'unified_ALLREL_direction_pattern;R72_numeric_gates_unchanged','update':'episodic_source_reset_existing_MLP_single_Adam_proposal_lr0.0005_clip1','backtrack':'same_frozen_N_reject_plus_KL;1_to1over256;minimum_loss_drop_max1e-12_1e-6absL;transactional_state_rollback','objective':'-log(1-P(N))+KL(P0||P),lambda1','direct_control':'exact_same_objective_distribution_optimum_pN=p0N/2','mismatch_control':'swap_left_right_same_scope_geometry_and_budget','head_initial':'crossplatform/round50/episodic336_capture_attempt001/SOURCE_MATCHING_MLP_PRIVATE.npz','evidence_root':'crossplatform/round73/evidence64_attempt001','adapt_root':'crossplatform/round73/adapt64_attempt001','prediction_root':'crossplatform/round73/predicted64_attempt001','GT_used_for_selection_or_steps':False,'source_training':False,'FINAL_access':False,'external_feedback':False,'conditional_expansion':'only_if_mechanism_set_direct_position_and_TTA_show_grounding_correction;no_parameter_change_after_scoring','selected_manifest_sha256':sha(H/'input/MECHANISM64_PRIVATE.json'),'programs_sha256':{},'inputs_sha256':{}}
for f in ['common73.py','parser73.py','parser_tests73.py','register73.py']:p['programs_sha256']['research/round73/'+f]=sha(H/f)
for f in ['research/round72/text72.py','research/round72/constraints72.py','research/round72/geometry72.py','research/round72/adapt72.py','research/round58/numerics58.py','crossplatform/round7/camera_projection.py']:p['programs_sha256'][f]=sha(ROOT/f)
for f in ['research/round69/input/ALL2304_PRIVATE.json','research/round24/input/ALL336_PRIVATE.json','research/round69/input/COHORT_BY_ID_PRIVATE.json','crossplatform/round72/evidence2640_camera_contract_attempt001/SEALED.json','crossplatform/round72/evidence2640_attempt001/SEALED.json',p['head_initial']]:p['inputs_sha256'][f]=sha(ROOT/f)
assert not(H/'experiment/DESIGN_LOCK.json').exists();dump(H/'experiment/DESIGN_LOCK.json',p);print({'n':64,'counts':dict(counts),'cells':cells,'target_color_recovered':attribute_recovered})
