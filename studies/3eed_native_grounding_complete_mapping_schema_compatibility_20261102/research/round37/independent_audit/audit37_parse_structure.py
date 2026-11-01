"""Inspect inert raw JSON structure after fixed wrappers; no GT or new parser readout."""
import json,re,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;P=ROOT/'crossplatform/round37/inference208_attempt003/calls_PRIVATE'
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 rows=[];counts=collections.Counter();fields=collections.Counter()
 for p in sorted(P.glob('*_point.json')):
  x=json.loads(p.read_text());z=x['result']
  if z['parsed']['status']!='parse_failure':continue
  t=z['raw_answer_PRIVATE'].strip();wr=[]
  for pat,name in [(r'```(?:json)?\s*(.*?)\s*```','markdown_json_fence'),(r'<answer>\s*(.*?)\s*</answer>','answer_tag')]:
   m=re.fullmatch(pat,t,re.S|re.I)
   if m:t=m.group(1).strip();wr.append(name)
  try:
   o=json.loads(t);kind='singleton_list_object' if isinstance(o,list) and len(o)==1 and isinstance(o[0],dict) else 'other_valid_JSON';keys=sorted(o[0]) if kind=='singleton_list_object' else [];fields[','.join(keys)]+=1
  except Exception:kind='invalid_JSON';keys=[]
  counts[kind]+=1;rows.append({'alias':x['binding']['alias'],'structure':kind,'object_keys':keys,'fixed_wrappers_removed':wr,'original_strict_parse_reason':z['parsed']['reason'],'raw_answer_sha256':hashlib.sha256(z['raw_answer_PRIVATE'].encode()).hexdigest()})
 assert len(rows)==63 and counts=={'singleton_list_object':62,'invalid_JSON':1}
 diag={'state':'PASS_STRUCTURE_ONLY','point_strict_parse_failures':63,'after_original_fixed_wrapper_removal':dict(counts),'singleton_object_key_patterns':dict(fields),'rows':rows,'no_R38_parser_or_prediction_or_taskGT_used':True,'no_raw_answer_or_coordinates_exported_here':True,'original_R37_strict_predictions_unchanged':True}
 dump(HERE/'PARSE_FAILURE_STRUCTURE_DIAGNOSTIC.json',diag)
 resultpath=HERE/'AUDIT_RESULT.json';result=json.loads(resultpath.read_text());original=dict(result);dump(HERE/'INITIAL_PARSE_SHAPE_METADATA_ERROR.json',{'state':'auditor_descriptive_metadata_corrected','original_wrong_shape_counts':original['strict_parse_fail_shape_counts'],'reason':'Initial supplemental classifier called JSON decoder before removing already-allowed markdown wrappers; strict parser audit and all readouts/task metrics remained correct.','correct_counts':dict(counts),'correction_uses_no_GT_no_R38_no_new_generation':True})
 result['strict_parse_fail_shape_counts']=dict(counts);result['parse_failure_structure_diagnostic']='PARSE_FAILURE_STRUCTURE_DIAGNOSTIC.json';result['supplemental_shape_classifier_correction']='INITIAL_PARSE_SHAPE_METADATA_ERROR.json';dump(resultpath,result);print(json.dumps({k:v for k,v in diag.items() if k!='rows'}))
if __name__=='__main__':main()
