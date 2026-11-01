"""CPU verify parser recovery equals unique raw position and fixed mapping."""
from common38 import *
import numpy as np

def main():
 check_plan();out=BASE/'predictions208_attempt001';verify(out);count=0;recovered=[];strictunchanged=0
 for r in rows():
  z=arrays(r);case=prepared(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));rc=read(out/'cases_PRIVATE'/(r['id']+'.json'));q=dict(zip(p['arm_names'].tolist(),map(int,p['queries'])));assert p['arm_names'].tolist()==ARMS and np.array_equal(p['boxes'],z['source_boxes'][p['queries']])
  for i in ['point','som']:
   c=rc['calls'][i];path=ROOT/c['raw_file'];assert sha(path)==c['raw_sha256'];raw=read(path)['result'];compatible=parse_compatible(raw['raw_answer_PRIVATE'],i);assert compatible==c['compatible_parse'];assert raw['parsed']==c['strict_parse']
   if compatible['status']!='parse_failure':assert compatible['value_equal_raw_unique_position'] and compatible['value']==compatible['raw_unique_position']
   if c['strict_parse']['status']!='parse_failure':assert compatible['value']==c['strict_parse']['value'];strictunchanged+=1
   if c['recovery']['recovered']:recovered.append(c['recovery'])
   for mapping in ['fullmap','oldsubset']:
    new=dict(raw,parsed=compatible);a=direct_readout(i,new,case,z,subset=mapping=='oldsubset');assert a==rc['readouts'][i+'_compatible_'+mapping] and a['query']==q[i+'_compatible_'+mapping]
   count+=1
 assert all(x['interface']=='point' and x['value_equal_raw_unique_position'] for x in recovered);dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','all_cached_calls_reparsed':count,'strict_valid_values_unchanged':strictunchanged,'recovered_unique_positions':len(recovered),'recovered_coordinate_values_unchanged':True,'SoM_strict_and_compatible_identical':True,'no_GT_GPU_VLM_or_TTA':True,'prediction_seal_sha256':sha(out/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json')});print('AUDIT38_PASSED',count,len(recovered))
if __name__=='__main__':main()
