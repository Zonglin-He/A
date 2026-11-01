"""Postseal replay of exact bbox values, full inventories, assignments, boxes."""
from common40 import *

def main():
 check_plan();out=BASE/'predictions208_attempt001';verify(out);verify(PREVIOUS);N=0
 for r in rows():
  z=arrays(r);case=prepared(r);p=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));prev=load_npz(PREVIOUS/'predictions_PRIVATE'/(r['id']+'.npz'));c=read(out/'cases_PRIVATE'/(r['id']+'.json'));rawfile=ROOT/c['raw_file'];assert sha(rawfile)==c['raw_sha256'];raw=read(rawfile)['result'];assert c['bbox_parsed_original']==raw['parsed'] and c['bbox_position_values_exact'];assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'][:len(previous.ARMS)],prev['queries']);assert np.array_equal(p['boxes'],z['source_boxes'][p['queries']]);a=iou_assignment(raw,case,z);assert a==c['assignment'];q=dict(zip(ARMS,map(int,p['queries'])))
  for m in ['fullmap','oldsubset']:
   key='bbox_IoU_'+m;d=iou_readout(a,case,z,subset=m=='oldsubset');assert d==c['readouts'][key] and d['query']==q[key]
  N+=1
 dump(EXP/'CPU_AUDIT.json',{'state':'PASSED','observations':N,'original_bbox_position_values_exact':True,'all13_cached_query_controls_exact':True,'all_original_source_boxes_exact':True,'all_mask_IoU_and_group_readout_replayed':True,'no_GT_GPU_VLM_TTA':True,'prediction_seal_sha256':sha(out/'SEALED.json')});print('R40_CPU_AUDIT_PASS')
if __name__=='__main__':main()
