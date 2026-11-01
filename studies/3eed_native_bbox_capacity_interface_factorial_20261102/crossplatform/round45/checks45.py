"""CPU synthetic-only raw/schema/center/rectIoU contracts; no model/media/GT."""
from common45 import *
import ast,tempfile
def main():
 import common45 as c
 with tempfile.TemporaryDirectory(prefix='r45_raw_contract_') as d:
  p=Path(d);raw=p/'raw.json';tok=p/'tokens.json';out=p/'call.json';text='[{"bbox_2d":[1,2,7,8],"label":"inert untrusted text"}]';capture_decoded_raw(raw,text);atomic(tok,{'generated_token_ids':[1,151645]});saved=raw.read_bytes();h=c.sha
  def fail(_):raise RuntimeError('injected postraw hash failure')
  c.sha=fail
  try:annotate_saved(raw,{'eos_token_ids':[151645,151643],'model_resized_wh':[20,10]},tok,out);raise AssertionError('missing injected failure')
  except RuntimeError as e:assert 'injected' in str(e)
  finally:c.sha=h
  assert raw.read_bytes()==saved and not out.exists();r=annotate_saved(raw,{'eos_token_ids':[151645,151643],'model_resized_wh':[20,10]},tok,out);assert r['parsed']['status']=='parsed' and r['parsed']['value']==[1,2,7,8] and r['parsed']['optional_label_ignored']
 bad=['[{"bbox_2d":[1,2,7,8]},{"bbox_2d":[2,2,5,5]}]','{"bbox_2d":[1,2,7,8],"bbox":[1,2,7,8]}','{"bbox_2d":[1,2,7,8],"bbox_2d":[1,2,7,8]}','{"bbox_2d":[true,2,7,8]}','{"bbox_2d":[NaN,2,7,8]}','{"bbox_2d":[1,2,7,8]}`','{"bbox_2d":[1,2,7,8],"run":"tool"}'];assert all(parse_bbox(x,[20,10])['status']=='parse_failure' for x in bad)
 assert parse_bbox('{"bbox_2d":[7,2,1,8]}',[20,10])['status']=='invalid_box';assert parse_bbox('{"bbox_2d":[1,2,21,8]}',[20,10])['status']=='invalid_box'
 masks=np.zeros((2,10,20),bool);masks[0,1:9,1:9]=True;masks[1,1:9,10:19]=True;rects=np.array([[1,1,9,9],[10,1,19,9]],float);z={'source_scores':np.array([.9,.8,.7]),'masks_bool':masks,'mask_boxes':rects};case={'image_size_wh':[20,10],'full_groups':{'0':[],'1':[2]},'old_subset_groups':{'1':[2]},'candidates':[{'mask_index':i,'bbox_xyxy':b.tolist()} for i,b in enumerate(rects)]}
 def result(box):return {'parsed':parse_bbox(json.dumps({'bbox_2d':box}),[560,280]),'image_grid_thw':[[1,20,40]],'model_resized_wh':[560,280]}
 a=result([28,28,252,252]);assert bbox_readout(a,case,z)['status']=='mask_without_3D_group';assoc=iou_assignment(a,case,z);assert assoc['mask_index']==0 and iou_readout(assoc,case,z)['status']=='mask_without_3D_group' # never choose second merely to get a3Dgroup
 overlap=iou_assignment(result([28,28,336,252]),case,z);assert overlap['mask_index']==0 and overlap['mask_IoU'][1]>0 and iou_readout(overlap,case,z)['status']=='mask_without_3D_group'
 b=result([280,28,532,252]);assert bbox_readout(b,case,z)['query']==2 and iou_readout(iou_assignment(b,case,z),case,z)['query']==2
 ties={**z,'mask_boxes':np.array([[1,1,9,9],[1,1,9,9]],float)};tc={**case,'candidates':[{'mask_index':i,'bbox_xyxy':[1,1,9,9]} for i in range(2)]};assert iou_assignment(a,tc,ties)['status']=='exact_max_rectangle_IoU_tie'
 assert iou_assignment(result([0,0,14,14]),case,z)['status']=='all_rectangle_IoU_zero';assert len(ARMS)==21 and len(set(ARMS))==21
 for f in ['common45.py','run45.py','checks45.py']:ast.parse((BASE/f).read_text())
 atomic(EXP/'CPU_TECH_CHECKS.json',{'state':'PASSED','implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'raw_survives_injected_postprocessing_failure':True,'bbox_singleton_label_security_negative_count':len(bad),'invalid_xyxy_not_repaired':True,'center_axis_inverse_fixed':True,'all_inventory_max_before_group_availability':True,'max_empty_group_no_secondbest':True,'exacttie_and_allzero_unknown':True,'21_unique_arms':True,'no_model_processor_realmedia_taskGT_or_GPU':True});print('R45_CPU_TECH_CHECKS_PASSED')
if __name__=='__main__':main()
