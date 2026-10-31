"""CPU array-only scene↔semantic interface fixture; no real files/model/GT."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).parent.parent/'round7'))
from camera_projection import CameraSpec
from scene_observation import build_scene_observation
from semantic_runner import interpret_answers,json_safe
D=Path(__file__).resolve().parents[2]/'research/round15/association'
def main():
 points=np.asarray([[a+offset,b,10.] for offset in (-1.,1.) for a in np.linspace(-.5,.5,5) for b in np.linspace(-.5,.5,5)])
 masks=np.zeros((2,64,64),bool);masks[0,30:35,26:31]=True;masks[1,30:35,34:39]=True
 inst={'boxes_xyxy':np.asarray([[26,30,31,35],[34,30,39,35]]),'class_ids':np.asarray([1,1]),'scores':np.asarray([.8,.9]),'masks_bool':masks,'instance_indices':np.asarray([10,20])}
 caption='the man left of another man in the image'
 raw=json.dumps({'status':'RESOLVED','target':{'quote':'the man','head':'man'},'references':[{'entity':'R1','quote':'another man','head':'man','distinct_from_target':True}],'attributes':[],'relations':[{'id':'L1','subject':'T','object':'R1','quote':'left of another man','predicate':'LEFT_OF','frame':'camera_image','viewpoint_quote':'the image'}]})
 ca=[{'candidate_id':'C001','mask_id':'M00010'},{'candidate_id':'C002','mask_id':'M00020'}];cb=[{'candidate_id':'C001','mask_id':'M00020'},{'candidate_id':'C002','mask_id':'M00010'}]
 v0='{"status":"RESOLVED","roles":[{"entity":"T","candidates":["C001"]},{"entity":"R1","candidates":["C002"]}],"attribute_checks":[]}'
 v1='{"status":"RESOLVED","roles":[{"entity":"T","candidates":["C002"]},{"entity":"R1","candidates":["C001"]}],"attribute_checks":[]}'
 selected=[]
 for domain in ('quad','drone'):
  cam=CameraSpec(domain,np.asarray([[40.,0,32],[0,40,32],[0,0,1]]),np.eye(4),np.zeros(5));scene=build_scene_observation(raw_points=points,camera_spec=cam,image_size_wh=(64,64),instances=inst,image_sha256='a'*64)
  out=interpret_answers(caption=caption,text_raw=raw,visual_raws=[v0,v1],candidates_by_order=[ca,cb],scene=scene);g=out['graph_match'];assert g['status']=='unique_supported' and g['selected_target_mask_id']=='M00010' and g['search_complete'];json.dumps(json_safe(out));selected.append(g['selected_target_mask_id'])
  try:interpret_answers(caption=caption,text_raw=raw,visual_raws=[v0,v1],candidates_by_order=[ca[:1],cb],scene=scene)
  except ValueError:pass
  else:raise AssertionError('Beam truncation cannot become complete inventory')
 assert selected==['M00010','M00010']
 r={'state':'PASS_R15_SHARED_SCENE_SEMANTIC_SYNTHETIC_INTEROP','same_core_bothplatforms':True,'bbox4_centroid2_extent2_flags_compatible':True,'global_exclusive_zbuffer_before_semantics':True,'scene_has_no_sourceq0_optimization_or_GT':True,'complete_mask_inventory_required':True,'json_safe_receipt_roundtrip':True,'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('semantic_graph.py'),Path(__file__).with_name('semantic_runner.py'),Path(__file__).with_name('scene_observation.py'))},'actual_model_GPU_data_GT_score_or_image_read':False};(D/'INTEGRATION_CPU_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(r['state'])
if __name__=='__main__':main()
