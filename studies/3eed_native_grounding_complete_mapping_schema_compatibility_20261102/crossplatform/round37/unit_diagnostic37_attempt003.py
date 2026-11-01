"""Fixed 2 sizes x 4 physical blue-square positions: capability, not 8/8 gate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,json,numpy as np
from dataclasses import replace
from PIL import Image,ImageDraw
from common37 import *
from association import DEFAULT_CONFIG
from local_engine37 import LocalFrozenVLM
from attempt003_contract import check_attempt003,check_plan_inputs003 as check_plan
from run37 import generate

def main():
 plan=check_plan();check_attempt003();out=BASE/'unit_diagnostic_attempt003';out.mkdir(exist_ok=False);assert all(sha(Path(plan['model_dir'])/n)==h for n,h in plan['model_files_sha256'].items());engine=LocalFrozenVLM(Path(plan['model_dir']),replace(DEFAULT_CONFIG,max_new_tokens=96));records=[];ledger=EXP/'GENERATION_BUDGET.jsonl'
 for W,H in [(672,448),(1008,672)]:
  for cx,cy in [(.25,.25),(.25,.75),(.75,.25),(.75,.75)]:
   im=Image.new('RGB',(W,H),'white');side=.1*min(W,H);x=cx*W;y=cy*H;rect=[x-side/2,y-side/2,x+side/2,y+side/2];ImageDraw.Draw(im).rectangle(rect,fill=(0,0,255));name=f'{W}x{H}_{int(cx*100)}_{int(cy*100)}';path=out/(name+'.png');im.save(path);attempts=ledger.read_text().splitlines() if ledger.exists() else [];assert len(attempts)<440
   with ledger.open('a') as f:f.write(json.dumps({'attempt_index':len(attempts)+1,'interface':'point_unit_diagnostic','canvas_wh':[W,H],'center_norm':[cx,cy],'phase':'attempt003_synthetic_fixed_factorial','before_taskGT':True,'started_unix':time.time()})+'\n');f.flush();os.fsync(f.fileno())
   res=generate(engine,'point',{'imagepath':str(path),'caption':'the blue square'},{'image_size_wh':[W,H]});assert res['model_resized_wh']==[W,H];v=res['parsed']['value'];nonnull=res['parsed']['status']=='parsed';inside=bool(nonnull and 0<=v[0]<W and 0<=v[1]<H and im.getpixel((int(np.floor(v[0])),int(np.floor(v[1]))))==(0,0,255));absnorm=[float(v[0]/W),float(v[1]/H)] if nonnull else None;norm1000=[float(v[0]/1000),float(v[1]/1000)] if nonnull else None;rec={'canvas_wh':[W,H],'center_norm':[cx,cy],'center_absolute_pixels':[x,y],'square_side_pixels':side,'blue_rect_absolute_pixels':rect,'output_inside_actual_blue_region':inside,'absolute_normalized_prediction':absnorm,'counterfactual1000_normalized_prediction':norm1000,'absolute_normalized_L2_to_center':float(np.linalg.norm(np.asarray(absnorm)-[cx,cy])) if nonnull else None,'counterfactual1000_normalized_L2_to_center':float(np.linalg.norm(np.asarray(norm1000)-[cx,cy])) if nonnull else None,'result':res};records.append(rec);dump(out/(name+'.json'),rec)
 pairs=[]
 for center in [(.25,.25),(.25,.75),(.75,.25),(.75,.75)]:
  a,b=[r for r in records if r['center_norm']==list(center)];va=a['result']['parsed']['value'];vb=b['result']['parsed']['value'];okay=a['result']['parsed']['status']==b['result']['parsed']['status']=='parsed';pairs.append({'center_norm':list(center),'small_output':va,'large_output':vb,'expected_absolute_scale':1.5,'large_minus_1p5small_pixels':[float(vb[i]-1.5*va[i]) for i in [0,1]] if okay else None,'absolute_normalized_response_difference':float(np.linalg.norm(np.asarray(a['absolute_normalized_prediction'])-b['absolute_normalized_prediction'])) if okay else None,'counterfactual1000_response_difference':float(np.linalg.norm(np.asarray(a['counterfactual1000_normalized_prediction'])-b['counterfactual1000_normalized_prediction'])) if okay else None})
 assert engine.verify_unchanged();parsed=[r for r in records if r['result']['parsed']['status']=='parsed'];summary={'state':'COMPLETED_MODEL_CAPABILITY_DIAGNOSTIC','model_calls':8,'physical_square_side_correct':True,'all_actual_grid_times14_equals_canvas':True,'inside_blue_count':sum(r['output_inside_actual_blue_region'] for r in records),'denominator_all_calls':8,'parsed_nonnull_count':len(parsed),'mean_absolute_normalized_L2':float(np.mean([r['absolute_normalized_L2_to_center'] for r in parsed])) if parsed else None,'mean_counterfactual1000_normalized_L2':float(np.mean([r['counterfactual1000_normalized_L2_to_center'] for r in parsed])) if parsed else None,'no_per_image_unit_choice_no_calibration':True,'perfect_localization_not_technical_gate':True,'original_002_failed_fixture_preserved':True,'fixed_pinned_absolute_protocol_for_dataset_unchanged':True,'paired_resize_responses':pairs,'records':records};dump(out/'RESULT.json',summary);seal(out,'R37_fixed_physical_blue_square_8call_model_capability_diagnostic',n=8,calls=8,not_taskGT=True,not_perfect_hit_gate=True);print('UNIT_DIAGNOSTIC37', {k:v for k,v in summary.items() if k not in ['records']})
if __name__=='__main__':main()
