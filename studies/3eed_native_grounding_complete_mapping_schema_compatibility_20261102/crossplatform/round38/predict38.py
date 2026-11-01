"""Apply one fixed compatible JSON parser to ALL sealed R37 answers, no VLM."""
from common38 import *
import numpy as np

def main():
 check_plan();assert read(EXP/'SYNTHETIC_CHECKS.json')['state']=='PASSED';verify(R37);out=BASE/'predictions208_attempt001';out.mkdir(exist_ok=False)
 for n in ['cases_PRIVATE','predictions_PRIVATE']:(out/n).mkdir()
 recovery=[]
 for r in rows():
  case=prepared(r);z=arrays(r);old=load_npz(R37/'predictions_PRIVATE'/(r['id']+'.npz'));oldrow=read(R37/'predictions_PRIVATE'/(r['id']+'.json'));oldq=dict(zip(old['arm_names'].tolist(),map(int,old['queries'])));q={a:oldq[a] for a in ['source','R29_direct','legacy_G']};reads={};calls={}
  for interface in ['point','som']:
   path=R37/oldrow['call_refs'][interface]['file'];assert sha(path)==oldrow['call_refs'][interface]['sha256'];raw=read(path)['result'];p=parse_compatible(raw['raw_answer_PRIVATE'],interface);sp=raw['parsed'];strictvalid=sp['status']!='parse_failure'
   if strictvalid:assert p['status']==sp['status'] and p['value']==sp['value'],('strict_value_changed',r['alias'],interface)
   recovered=sp['status']=='parse_failure' and p['status']!='parse_failure';detail={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'interface':interface,'old_status':sp['status'],'new_status':p['status'],'recovered':recovered,'singleton_unwrapped':p['singleton_unwrapped'],'label_ignored':p['optional_label_ignored'],'value_equal_raw_unique_position':p['value_equal_raw_unique_position'],'strict_valid_value_unchanged':True if strictvalid else None,'old_accepted_value':sp['value'],'new_value':p['value'],'raw_unique_position':p.get('raw_unique_position'),'raw_file':str(path.relative_to(ROOT)),'raw_sha256':sha(path)};recovery.append(detail);calls[interface]={'strict_parse':sp,'compatible_parse':p,'recovery':detail,'raw_file':str(path.relative_to(ROOT)),'raw_sha256':sha(path)};new=dict(raw,parsed=p)
   for mapping in ['fullmap','oldsubset']:
    name=interface+'_strict_'+mapping;reads[name]=oldrow['readouts'][interface+'_'+mapping];q[name]=oldq[interface+'_'+mapping];name=interface+'_compatible_'+mapping;reads[name]=direct_readout(interface,new,case,z,subset=mapping=='oldsubset');q[name]=reads[name]['query']
    if strictvalid:assert reads[name]==oldrow['readouts'][interface+'_'+mapping]
  if calls['som']['strict_parse']['status']!='parse_failure':assert q['som_compatible_fullmap']==q['som_strict_fullmap'] and q['som_compatible_oldsubset']==q['som_strict_oldsubset']
  queries=np.array([q[a] for a in ARMS]);np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),queries=queries,boxes=z['source_boxes'][queries]);dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'calls':calls,'readouts':reads,'no_new_VLM_or_feedback_filter':True})
 dump(out/'RECOVERY_ROWS_PRIVATE.json',recovery);seal(out,'R38_cached_schema_compatibility_all208_predictions_before_scoring',n=208,interfaces=2,new_VLM_calls=0,new_GPU_or_TTA_steps=0,original_R37_seal_sha256=sha(R37/'SEALED.json'),plan_sha256=sha(EXP/'PLAN_LOCK.json'),old_strict_predictions_preserved=True);print('PREDICT38_SEALED',sum(r['recovered'] for r in recovery))
if __name__=='__main__':main()
