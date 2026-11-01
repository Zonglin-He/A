"""Two generated blue-object absolute-pixel native point positive controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,json,numpy as np
from dataclasses import replace
from PIL import Image,ImageDraw
from common37 import *
from association import DEFAULT_CONFIG
from local_engine37 import LocalFrozenVLM,check_amendment
from run37 import generate

def main():
 p=check_plan();check_amendment();out=BASE/'unit_control_attempt002';out.mkdir(exist_ok=False);engine=LocalFrozenVLM(Path(p['model_dir']),replace(DEFAULT_CONFIG,max_new_tokens=96));records=[];ledger=EXP/'GENERATION_BUDGET.jsonl'
 for W,H in [(672,448),(1008,672)]:
  im=Image.new('RGB',(W,H),'white');ImageDraw.Draw(im).rectangle([.2*W,.7*H,.3*W,.8*H],fill=(0,0,255));path=out/(str(W)+'x'+str(H)+'.png');im.save(path);attempts=ledger.read_text().splitlines() if ledger.exists() else [];assert len(attempts)<440
  with ledger.open('a') as f:f.write(json.dumps({'attempt_index':len(attempts)+1,'interface':'point_unit_control','canvas_wh':[W,H],'before_taskGT':True,'started_unix':time.time()})+'\n');f.flush();os.fsync(f.fileno())
  r={'imagepath':str(path),'caption':'the blue square'};case={'image_size_wh':[W,H]};res=generate(engine,'point',r,case);assert res['model_resized_wh']==[W,H],'Canvas must remain actual input size';v=res['parsed']['value'];inside=bool(res['parsed']['status']=='parsed' and v is not None and 0<=v[0]<W and 0<=v[1]<H and im.getpixel((int(np.floor(v[0])),int(np.floor(v[1]))))==(0,0,255));rec={'canvas_wh':[W,H],'blue_normalized_bbox':[.2,.7,.3,.8],'actual_resized_wh':res['model_resized_wh'],'output_inside_actual_blue_region':inside,'result':res};records.append(rec);dump(out/(str(W)+'x'+str(H)+'.json'),rec)
 assert engine.verify_unchanged();receipt={'state':'PASSED' if all(r['output_inside_actual_blue_region'] for r in records) else 'FAILED_MODEL_COORDINATE_UNIT_CONTROL','actual_model_calls':2,'same_point_messages_schema_decode_as_A208':True,'not_task_data_or_GT':True,'records':records,'no_per_image_unit_choice':True};dump(out/'RESULT.json',receipt);seal(out,'R37_two_real_model_blue_object_absolute_pixel_controls',n=2,calls=2,passed=receipt['state']=='PASSED');print('UNIT_CONTROL37',receipt)
if __name__=='__main__':main()
