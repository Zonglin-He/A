"""R37 A208-only G interfaces. Prompts never receive source scores or task GT."""
import sys,json,hashlib,re,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round13')]
import common30
from common30 import read,dump,sha,seal,verify,load_npz
from association import numbered_candidates,render_numbered_panel,array_sha
BASE=ROOT/'crossplatform/round37';EXP=ROOT/'research/round37/experiment'
ARMS=['source','R29_direct','legacy_G','point_fullmap','point_oldsubset','som_fullmap','som_oldsubset']

def rows():return common30.rows('A208')
def feature(r):return common30.feature(r)
def candidates_for(r,frame_or_masks,boxes=None):
 """Pure ordering, same R13 permutation0 hash; no source/VLM/task fields read."""
 if isinstance(frame_or_masks,dict):masks=frame_or_masks['instances']['masks_bool'];boxes=frame_or_masks['instances']['boxes_xyxy']
 else:masks=frame_or_masks
 cc=numbered_candidates(r['id'],masks,boxes,permutation=0)
 return [dict(c,candidate_id=str(j+1),target_id=j+1) for j,c in enumerate(cc)]
def marker_panel(rgb,candidates):return render_numbered_panel(rgb,candidates)
def prepared(r):return read(BASE/'prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'))
def arrays(r):return load_npz(BASE/'prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'))
def check_plan():
 p=read(EXP/'PLAN_LOCK.json')
 for n,h in p['program_sha256'].items():assert sha(ROOT/n)==h,n
 for n,h in p['input_sha256'].items():assert sha(ROOT/n)==h,n
 return p

def parse_json(raw,interface):
 t=str(raw).strip();wrappers=[]
 m=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',t,flags=re.S|re.I)
 if m:t=m.group(1).strip();wrappers.append('markdown_json_fence')
 m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',t,flags=re.S|re.I)
 if m:t=m.group(1).strip();wrappers.append('answer_tag')
 def pairs(ps):
  d={}
  for k,v in ps:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 try:o=json.loads(t,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite constant')))
 except Exception as e:return {'status':'parse_failure','reason':type(e).__name__,'value':None,'wrappers':wrappers}
 if not isinstance(o,dict) or len(o)!=1:return {'status':'parse_failure','reason':'exact_one_key_object_required','value':None,'wrappers':wrappers}
 key=next(iter(o));allowed={'point':{'point','point2d','point_2d'},'som':{'target_id','id','candidate_id'}}[interface]
 if key not in allowed:return {'status':'parse_failure','reason':'unknown_key','value':None,'wrappers':wrappers}
 v=o[key]
 if v is None:return {'status':'explicit_null','reason':None,'value':None,'wrappers':wrappers,'canonical_key':'point' if interface=='point' else 'target_id','input_key':key}
 if interface=='point':okay=isinstance(v,list) and len(v)==2 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v)
 else:okay=isinstance(v,int) and not isinstance(v,bool)
 if not okay:return {'status':'parse_failure','reason':'invalid_value_type_or_shape','value':None,'wrappers':wrappers}
 return {'status':'parsed','reason':None,'value':v,'wrappers':wrappers,'canonical_key':'point' if interface=='point' else 'target_id','input_key':key}
def inverse_point(point,original_wh,grid):
 assert len(grid)==3 and grid[0]==1;W,H=map(int,original_wh);Hp,Wp=int(grid[1])*14,int(grid[2])*14;x,y=map(float,point)
 if not (0<=x<Wp and 0<=y<Hp):return None
 return [x*W/Wp,y*H/Hp]
def mask_at_point(point,masks):
 W,H=masks.shape[2],masks.shape[1]
 if point is None or not (0<=point[0]<W and 0<=point[1]<H):return None,'coordinate_outside_image'
 hits=np.flatnonzero(masks[:,int(np.floor(point[1])),int(np.floor(point[0]))])
 if len(hits)!=1:return None,'no_mask' if len(hits)==0 else 'overlapping_multiple_masks'
 return int(hits[0]),'single_independent_mask'
def direct_readout(interface,result,case,z,subset=False):
 q0=int(z['source_scores'].argmax());parsed=result['parsed'];m=None;point=None;status=parsed['status']
 if parsed['status']=='parsed':
  if interface=='point':point=inverse_point(parsed['value'],case['image_size_wh'],result['image_grid_thw'][0]);m,status=mask_at_point(point,z['masks_bool'])
  else:
   c=next((c for c in case['candidates'] if c['target_id']==parsed['value']),None)
   if c is None:status='unlisted_target_id'
   else:m=int(c['mask_index']);status='listed_independent_mask'
 table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else []
 if not g:return {'query':q0,'mapped':False,'status':status if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group','mask_index':m,'original_point':point,'group_size':0,'query_changed':False,'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[]))}
 q=max(g,key=lambda i:(float(z['source_scores'][i]),-i));return {'query':int(q),'mapped':True,'status':'mapped_original_query','mask_index':m,'original_point':point,'group_size':len(g),'query_changed':q!=q0,'old_subset_contains_selected_mask':str(m) in case['old_subset_groups'],'full_group_nonempty':bool(case['full_groups'].get(str(m),[]))}
