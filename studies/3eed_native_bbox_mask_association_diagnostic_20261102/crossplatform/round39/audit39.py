"""Postseal CPU replay; unchanged cached controls and original boxes verified."""
from common39 import *
import numpy as np

def main():
 check_plan();out=BASE/'inference208_attempt001';verify(out);verify(PREVIOUS);count=0;st={}
 for r in rows():
  case=prepared(r);z=arrays(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));prev=load_npz(PREVIOUS/'predictions_PRIVATE'/(r['id']+'.npz'));rec=read(out/'predictions_PRIVATE'/(r['id']+'.json'));path=out/rec['call_ref']['file'];assert sha(path)==rec['call_ref']['sha256'];raw=read(path)['result'];grid=raw['image_grid_thw'];wh=[grid[0][2]*14,grid[0][1]*14];assert raw['model_resized_wh']==wh and raw['original_wh']==case['image_size_wh'];assert raw['parsed']==parse_bbox(raw['raw_answer_PRIVATE'],wh);assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'][:len(previous.ARMS)],prev['queries']);assert np.array_equal(p['boxes'],z['source_boxes'][p['queries']]);q=dict(zip(ARMS,map(int,p['queries'])))
  for m in ['fullmap','oldsubset']:
   n='bbox_'+m;a=bbox_readout(raw,case,z,subset=m=='oldsubset');assert a==rec['readouts'][n] and a['query']==q[n]
  st[raw['parsed']['status']]=st.get(raw['parsed']['status'],0)+1;count+=1
 assert len((EXP/'GENERATION_BUDGET.jsonl').read_text().splitlines())==208;dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','observations':count,'all11_cached_controls_queries_unchanged':True,'original_source_boxes_exact':True,'parser_grid_center_unique_mask_group_replayed':True,'parser_status':st,'new_VLM_calls':208,'prediction_seal_sha256':sha(out/'SEALED.json'),'no_GT_TTA_or_new_boxes':True,'plan_sha256':sha(EXP/'PLAN_LOCK.json')});print('R39_AUDIT_PASSED',st)
if __name__=='__main__':main()
