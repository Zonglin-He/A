"""Frozen sameQwen single nativebbox, A208 only, preflight reused, no GT/TTA."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common39 import *
import time,argparse
from PIL import Image
from association import DEFAULT_CONFIG
from local_engine37 import LocalFrozenVLM
from dataclasses import replace

def generate(engine,r,case):
 torch=engine.torch;im=Image.open(r['imagepath']).convert('RGB');assert list(im.size)==case['image_size_wh'];pre=engine.processor.image_processor(images=[im],return_tensors='pt');grid=pre['image_grid_thw'].tolist();assert len(grid)==1 and grid[0][0]==1;W,H=grid[0][2]*14,grid[0][1]*14;assert 256<=grid[0][1]*grid[0][2]//4<=1024
 prompt=engine.processor.apply_chat_template(messages_bbox(r['caption'],[W,H]),tokenize=False,add_generation_prompt=True);inputs=engine.processor(text=[prompt],images=[im],return_tensors='pt',padding=False,truncation=False).to('cuda');assert inputs['image_grid_thw'].tolist()==grid and inputs['input_ids'].shape[1]<=8192
 torch.cuda.synchronize();t=time.perf_counter()
 with torch.inference_mode():o=engine.model.generate(**inputs,do_sample=False,max_new_tokens=96,use_cache=True)
 torch.cuda.synchronize();tail=o[0,inputs['input_ids'].shape[1]:];raw=engine.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False);eos=engine.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 return {'raw_answer_PRIVATE':raw,'parsed':parse_bbox(raw,[W,H]),'image_grid_thw':grid,'model_resized_wh':[W,H],'original_wh':case['image_size_wh'],'image_visual_tokens':[grid[0][1]*grid[0][2]//4],'input_tokens':int(inputs['input_ids'].shape[1]),'generated_tokens':int(len(tail)),'ended_with_eos':any(int(x) in eos for x in tail),'seconds_generate':time.perf_counter()-t,'serialized_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'query_sha256':hashlib.sha256(r['caption'].encode()).hexdigest(),'images_array_sha256':[array_sha(np.asarray(im))],'max_new_tokens':96}

def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['preflight','full'],required=True);args=a.parse_args();plan=check_plan();assert read(EXP/'CPU_TECH_CHECKS.json')['state']=='PASSED';verify(PREVIOUS);out=BASE/'inference208_attempt001';out.mkdir(exist_ok=True)
 for n in ['calls_PRIVATE','predictions_PRIVATE']:(out/n).mkdir(exist_ok=True)
 binding={'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'previous_seal_sha256':sha(PREVIOUS/'SEALED.json'),'prepared_seal_sha256':sha(ROOT/'crossplatform/round37/prepared208_attempt001/SEALED.json'),'interface':'bbox_2d','n':208,'model_files_sha256':plan['model_files_sha256']}
 if (out/'RUN_BINDING.json').exists():assert read(out/'RUN_BINDING.json')==binding
 else:dump(out/'RUN_BINDING.json',binding)
 assert not (out/'SEALED.json').exists(),'Never overwrite completed inference'
 if args.phase=='full':assert read(out/'PREFLIGHT_RESULT.json')['state']=='PASSED'
 rr=rows();preids={x['id_PRIVATE'] for x in read(ROOT/'research/round37/experiment/PREFLIGHT12_PRIVATE.json')};rr=[r for r in rr if r['id'] in preids] if args.phase=='preflight' else rr;assert len(rr)==(12 if args.phase=='preflight' else 208)
 modeldir=Path(plan['model_dir']);assert all(sha(modeldir/n)==h for n,h in plan['model_files_sha256'].items());engine=LocalFrozenVLM(modeldir,replace(DEFAULT_CONFIG,max_new_tokens=96));ledger=EXP/'GENERATION_BUDGET.jsonl';records=[];newcalls=0;start=time.perf_counter()
 try:
  for j,r in enumerate(rr):
   case=prepared(r);z=arrays(r);path=out/'calls_PRIVATE'/(r['id']+'_bbox.json');cb={'id_PRIVATE':r['id'],'alias':r['alias'],'interface':'bbox','plan_sha256':binding['plan_sha256'],'image_sha256':case['image_sha256'],'query_sha256':hashlib.sha256(r['caption'].encode()).hexdigest()}
   if path.exists():record=read(path);assert record['binding']==cb
   else:
    attempts=ledger.read_text().splitlines() if ledger.exists() else [];assert len(attempts)<208;event={'attempt_index':len(attempts)+1,'alias':r['alias'],'interface':'bbox','phase':args.phase,'plan_sha256':binding['plan_sha256'],'started_unix':time.time()}
    with ledger.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
    record={'state':'completed_single_greedy_generation_no_retry','binding':cb,'result':generate(engine,r,case)};dump(path,record);newcalls+=1
   result=record['result'];reads={'bbox_'+m:bbox_readout(result,case,z,subset=m=='oldsubset') for m in ['fullmap','oldsubset']};old=load_npz(PREVIOUS/'predictions_PRIVATE'/(r['id']+'.npz'));assert old['arm_names'].tolist()==previous.ARMS;q=np.r_[old['queries'],[reads[n]['query'] for n in ['bbox_fullmap','bbox_oldsubset']]].astype(np.int64);assert len(q)==len(ARMS)
   np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),queries=q,boxes=z['source_boxes'][q]);row={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'readouts':reads,'call_ref':{'file':str(path.relative_to(out)),'sha256':sha(path)},'previous_case_file':str((PREVIOUS/'cases_PRIVATE'/(r['id']+'.json')).relative_to(ROOT))};dump(out/'predictions_PRIVATE'/(r['id']+'.json'),row);records.append(row)
   if (j+1)%4==0:print('R39',args.phase,j+1,'newcalls',newcalls,'seconds',round(time.perf_counter()-start,1),flush=True)
  assert engine.verify_unchanged()
  if args.phase=='preflight':
   raw=[read(out/x['call_ref']['file'])['result'] for x in records];mapped=sum(x['readouts']['bbox_fullmap']['mapped'] for x in records);parsed=sum(x['parsed']['status'] in ['parsed','explicit_null'] for x in raw);passed=parsed>0 and mapped>0;receipt={'state':'PASSED' if passed else 'STOPPED_ZERO_VALID_INTERFACE_COVERAGE','n':12,'newcalls':newcalls,'parsed_or_null':parsed,'nonzero_legal_3D_mapped':mapped,'query_changes':sum(x['readouts']['bbox_fullmap']['query_changed'] for x in records),'same_prompt_reusable':True,'task_GT_not_read':True,'software_validity_not_perfect_localization':True,'plan_sha256':binding['plan_sha256']};dump(out/'PREFLIGHT_RESULT.json',receipt);print('R39_PREFLIGHT',receipt)
  else:
   assert len(list((out/'calls_PRIVATE').glob('*.json')))==208 and len(list((out/'predictions_PRIVATE').glob('*.npz')))==208;dump(out/'ALL_READOUTS_PRIVATE.json',records);seal(out,'R39_A208_nativebbox_all_predictions_before_task_GT',n=208,successful_VLM_calls=208,total_generation_attempts=len(ledger.read_text().splitlines()),newcalls_this_phase=newcalls,preflight_calls_reused=12,model_frozen_unchanged=True,no_TTA_or_new_3D_boxes=True,plan_sha256=binding['plan_sha256'],seconds_this_phase=time.perf_counter()-start,peak_bytes=int(engine.torch.cuda.max_memory_allocated()));print('R39_FULL_SEALED')
 except BaseException as e:dump(out/('FAILURE_'+str(time.time_ns())+'.json'),{'state':'FAILED_OR_INTERRUPTED_PRESERVED','phase':args.phase,'type':type(e).__name__,'message':str(e),'newcalls':newcalls,'automatic_answer_retry':False});raise
if __name__=='__main__':main()
