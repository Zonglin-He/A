"""Fixed targettext/path inputs only; no metadata/GT/pixels/pointarray reads."""
from pathlib import Path
import json,hashlib,re,sys,collections
R=Path(__file__).resolve().parent;EX=R.parent;ROOT=R.parents[3];P=ROOT/'crossplatform';R5=P/'round5/baseline_engineering';G=P/'round7/multimodal_pseudo.py';sys.path.insert(0,str(P/'round7'));from multimodal_pseudo import target_color
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main():
 plan=read(R/'INPUT_PLAN.json');salt=plan['selection_salt'];rank=lambda x,role:hashlib.sha256((salt+'|'+role+'|'+x['id']).encode()).hexdigest();allrows=[];sources={};inputcounts={};counts={};smoke=[];mapping=[];indexrows=read(R5/'SCORING_INDEX_PRIVATE.json');idx={x['id']:(i,x) for i,x in enumerate(indexrows)};color_re=re.compile(r'\b(?:'+'|'.join(plan['audit_color_lexicon_only_for_negative_stratum'])+r')\b',re.I)
 for platform in ['drone','quad']:
  fp=R5/'inputs'/(platform+'_all.json');sources[str(fp)]=sha(fp);rr=read(fp);assert len(rr)==plan['expected_full_counts'][platform];assert all(x['platform']==platform and x['split']=='val' for x in rr)
  for source_i,x in enumerate(rr):
   assert x['id'] in idx;i,old=idx[x['id']];assert old['key']==x['key'] and old['platform']==platform and old['scene']==x['scene'] and old['frame']==x['frame'] and old['object_index']==x['object_index'];color,reason=target_color(x['caption']);row={**x,'r7_input_index':len(allrows),'R5_platform_input_index':source_i,'R5_scoring_index_row':i,'target_color':color,'target_color_parse_reason':reason,'audit_text_has_finite_color_lexeme':bool(color_re.search(x['caption'])),'historical_target_research_exposure':'R5_allofficialVAL_already_scored; notfreshconfirmation'}
   # Whitelisted paths only from oldscoring index, no metadata payload read.
   row['camera_metadata_path']=old['meta_path'];row['camera_metadata_sha256_expected']=old['metadata_container_sha256'];row['image_path']=str(Path(x['lidar_path']).with_name('image.jpg'));allrows.append(row);mapping.append({'r7_input_index':row['r7_input_index'],'id':x['id'],'R5_platform_file':str(fp),'R5_platform_input_index':source_i,'R5_scoring_index_row':i})
  inputcounts[platform]=len(rr)
 assert len({x['id'] for x in allrows})==len(allrows)==5912
 for platform,scene in sorted({(x['platform'],x['scene']) for x in allrows}):
  rr=[x for x in allrows if x['platform']==platform and x['scene']==scene];selected=[];frames=set();colors=set()
  for slot in range(3):
   pool=[x for x in rr if x['target_color'] is not None and x['frame'] not in frames];fresh=[x for x in pool if x['target_color'] not in colors]
   if not pool:break
   x=min(fresh or pool,key=lambda x:rank(x,'supported'));selected.append({**x,'smoke_role':'supported_color_'+str(slot+1)});frames.add(x['frame']);colors.add(x['target_color'])
  supported_n=len(selected);pool=[x for x in rr if x['target_color'] is None and x['frame'] not in frames];preferred=[x for x in pool if not x['audit_text_has_finite_color_lexeme']]
  assert pool,'No unsupportednegative frame; do not inventsilent substitute'
  x=min(preferred or pool,key=lambda x:rank(x,'negative'));selected.append({**x,'smoke_role':'unsupported_negative_no_finite_color_lexeme' if not x['audit_text_has_finite_color_lexeme'] else 'unsupported_negative_colorphrase'});frames.add(x['frame'])
  while len(selected)<4:
   pool=[x for x in rr if x['target_color'] is None and x['frame'] not in frames];assert pool,'Not enough distinctnegativeframes tofill predeclareddeficit';x=min(pool,key=lambda x:rank(x,'supported_deficit_fallback'));selected.append({**x,'smoke_role':'unsupported_fallback_for_unavailable_supported_slot'});frames.add(x['frame'])
  assert len(frames)==4;smoke.extend(selected);counts[platform+'/'+scene]={'platform':platform,'scene':scene,'all_inputs':len(rr),'unique_frames':len({x['frame'] for x in rr}),'supported_target_colors':dict(collections.Counter(x['target_color'] for x in rr if x['target_color'])),'parser_reasons':dict(collections.Counter(x['target_color_parse_reason'] for x in rr)),'no_finite_color_lexeme_count':sum(not x['audit_text_has_finite_color_lexeme'] for x in rr),'smoke_count':4,'smoke_supported':supported_n,'smoke_supported_goal':3,'smoke_supported_deficit':3-supported_n,'smoke_unique_frames':len(frames),'smoke_supported_colors':sorted(colors)}
 assert len(smoke)==20 and len(counts)==5;overlays=[]
 for key in sorted(counts):
  rr=[x for x in smoke if x['platform']+'/'+x['scene']==key];supported=[x for x in rr if x['target_color'] is not None];overlays.append(min(supported or rr,key=lambda x:rank(x,'overlay')))
 outputs={'full_inputs.json':allrows,'smoke_inputs.json':smoke,'projection_inputs.json':overlays,'input_audit/R5_INDEX_MAPPING_PRIVATE.json':mapping}
 for name,v in outputs.items():
  fp=EX/name;assert not fp.exists();fp.write_text(json.dumps(v,indent=2,ensure_ascii=False))
 receipt={'state':'GTfree_target_inputs_fixed_before_pixels_or_points','full_n':5912,'full_by_platform':inputcounts,'smoke_n':20,'smoke_unique_frames':len({(x['platform'],x['scene'],x['frame']) for x in smoke}),'projection_n':5,'by_scene':counts,'source_input_sha256':sources,'R5_scoring_index_sha256':sha(R5/'SCORING_INDEX_PRIVATE.json'),'R5_index_usage':'Only id/key/platform/scene/frame/objectindex/pathhash mapping; no GTvalues exist inthat index. No metadata file payload opened.','input_plan_sha256':sha(R/'INPUT_PLAN.json'),'selection_code_sha256':sha(Path(__file__)),'frozen_target_color_parser_module_sha256':sha(G),'outputs':{name:sha(EX/name) for name in outputs},'no_GT_box_class_positive_map_metric_or_pixels_read':True,'no_model_GPU_scoring':True,'scope':'Engineering smoke intentionally color-enriched; allVAL notfiltered; historicallyR5exposed targets; genuineWaymosource crossplatform model'};(R/'INPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
