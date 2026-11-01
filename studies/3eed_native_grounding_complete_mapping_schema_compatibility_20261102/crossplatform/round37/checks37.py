"""CPU parser/safety/math and actual Qwen2.5 processor-grid positive controls."""
import numpy as np
from common37 import *
def main():
 from PIL import Image
 from transformers import AutoProcessor
 p=check_plan();tests=[]
 for bad in ['__import__("os").system("id")','{"point":[NaN,1]}','{"point":[true,2]}','{"point":[1,2],"point":[3,4]}','{"point":[1,2,3]}','{"target_id":"1"}','{"target_id":false}','{"point":{"x":1,"y":2}}']:
  interface='som' if 'target_id' in bad else 'point';assert parse_json(bad,interface)['status']=='parse_failure',bad
 tests.append('unsafe_nonfinite_duplicate_wrong_type_payloads_rejected_without_eval')
 for text,key,v in [('```json\n{"point2d":[1,2]}\n```','point',[1,2]),('<answer>{"id":3}</answer>','som',3),('{"point":null}','point',None)]:assert parse_json(text,key)['value']==v
 tests.append('fixed_wrappers_aliases_and_explicit_null')
 proc=AutoProcessor.from_pretrained(p['model_dir'],local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=p['min_pixels'],max_pixels=p['max_pixels']);assert proc.image_processor.patch_size==14 and proc.image_processor.merge_size==2
 synthetic=[]
 for W,H in [(640,480),(1920,1200),(2400,640)]:
  grid=proc.image_processor(images=[Image.new('RGB',(W,H),'gray')],return_tensors='np')['image_grid_thw'][0].tolist();Hp,Wp=grid[1]*14,grid[2]*14;point=inverse_point([Wp*.25,Hp*.75],[W,H],grid);assert np.allclose(point,[W*.25,H*.75]);assert inverse_point([Wp,0],[W,H],grid) is None;synthetic.append({'original_wh':[W,H],'grid':grid,'resized_wh':[Wp,Hp],'positive_quarter_threequarter':point})
 tests.append('actual_processor_grid_times14_absolute_pixels_not_1000_or_original_size')
 verify(BASE/'prepared208_attempt001');lookup={r['id']:r for r in rows()};positive=[]
 for spec in read(EXP/'PREFLIGHT12_PRIVATE.json'):
  r=lookup[spec['id_PRIVATE']];c=prepared(r);z=arrays(r);im=Image.open(r['imagepath']).convert('RGB');grid=proc.image_processor(images=[im],return_tensors='np')['image_grid_thw'][0].tolist();owned=z['masks_bool'].sum(0)==1;available=[]
  for m,g in c['full_groups'].items():
   yx=np.argwhere(z['masks_bool'][int(m)]&owned)
   if g and len(yx):available.append((int(m),yx))
  if available:
   m,yx=available[0];y,x=yx[len(yx)//2];W,H=c['image_size_wh'];point=[(x+.5)*grid[2]*14/W,(y+.5)*grid[1]*14/H];orig=inverse_point(point,[W,H],grid);got,status=mask_at_point(orig,z['masks_bool']);assert got==m and status=='single_independent_mask';res=direct_readout('point',{'parsed':{'status':'parsed','value':point},'image_grid_thw':[grid]},c,z);assert res['mapped'] and res['mask_index']==m;positive.append({'alias':r['alias'],'available_mapped_exclusive_positive':True,'positive_can_equal_source_q0':res['query']==c['source_q0'],'grid':grid,'mask_index_PRIVATE':m})
  else:positive.append({'alias':r['alias'],'available_mapped_exclusive_positive':False,'grid':grid})
 assert any(x['available_mapped_exclusive_positive'] for x in positive);dump(EXP/'CPU_TECH_CHECKS.json',{'state':'PASSED','tests':tests,'synthetic_processor_coordinates':synthetic,'preflight_geometry_coordinate_positive_controls':positive,'no_VLM_generations_GPU_or_taskGT':True,'coordinate_reference':p['coordinate_reference']});print('CHECKS37_PASSED')
if __name__=='__main__':main()
