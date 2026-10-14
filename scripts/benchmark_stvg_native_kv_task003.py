"""Sixteen fixed calls: task-output equivalence and hot repeat on all eight exposed sources."""
import sys,os,time,json,hashlib,tempfile,copy,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import benchmark_stvg_native_hot_teacher_v1 as h
BASE=ROOT/'artifacts/stvg_native_hot_teacher_v1/task_equivalence003'
sha=h.teacher.common.sha;read=h.teacher.common.read;write=h.teacher.common.write
def main():
 rt=read(BASE/'RUNTIME.json')
 assert rt['status']=='locked'and rt['GT_read']is False and rt['use_cache']is True
 for p,s in rt['pins'].items():assert sha(ROOT/p)==s,p
 sys.addaudithook(h.teacher.common.guard)
 oldlock=read(h.PREPARATION_LOCK);roster=read(h.ROSTER_PATH)
 specs=[dict(alias=('HC' if r['dataset']=='hc2' else 'V')+f"{int(r['case_id'][-3:])+1:02d}",case_id=r['case_id'],dataset=r['dataset'])for r in roster['records']]
 cases={spec['case_id']:h._case_binding(spec,roster)for spec in specs}
 call_plan=[dict(call_index=i+1,alias=spec['alias'],case_id=spec['case_id'],dataset=spec['dataset'],repeat_index=rep,manual_seed=h.MANUAL_SEED)for i,(spec,rep)in enumerate((spec,rep)for spec in specs for rep in (1,2))]
 for expected in oldlock['cases']:
  c=cases[expected['case_id']]
  for k,v in expected.items():assert c[k]==v,k
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 verified=h.teacher.verify_local_model_receipt();start=time.perf_counter()
 h.teacher.MAX_MEMORY={0:'22GiB','cpu':'4GiB'}
 processor,model,torch,device,mapping,load_s,peak=h.teacher._load_qwen(ROOT/h.teacher.CHECKPOINT_REL,BASE/'offload')
 original=h._model_generation_config(model);assert original['use_cache']is False
 model.generation_config.use_cache=True
 assert model.generation_config.to_dict()['use_cache'] is True
 rows=[];prev={}
 for plan in call_plan:
  c=cases[plan['case_id']];torch.manual_seed(h.MANUAL_SEED);torch.cuda.manual_seed_all(h.MANUAL_SEED)
  t=time.perf_counter();g=h.teacher._one_generation(c['request_object'],c['marker_video_path_private'],processor=processor,model=model,torch=torch,device=device);parsed=h._parse_generation(c,g)
  private=BASE/'private_calls'/f'{plan["alias"]}_{plan["repeat_index"]}.json'
  write(private,dict(plan=plan,generation=g,parsed=parsed,use_cache_actual=True,GT_read=False))
  r=h._call_receipt(plan,c,g,parsed,None,time.perf_counter()-t,prev.get(plan['case_id']))
  r.update(use_cache_actual=True,private_output_record_sha256=sha(private))
  oldgeneration=c['old_generation'];r['exact_input_match_old']=all(g[k]==oldgeneration[k]for k in ['input_ids_sha256','rendered_prompt_sha256','input_token_count','visual_token_count','video_grid_thw'])
  p=BASE/'calls'/f'{plan["alias"]}_{plan["repeat_index"]}.json';write(p,r)
  rows.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),status=r['status'],exact_text_old=r['output_exact_match_old_unseeded_text'],exact_id_old=r['selected_id_match_old_unseeded_parse'],exact_input_old=r['exact_input_match_old'],repeat=r['repeat_comparison'],generation_s=g['model_generate_s'],output_tokens=g['output_token_count'],peak_allocated_bytes=g['peak_allocated_bytes'],peak_reserved_bytes=g['peak_reserved_bytes']))
  prev[plan['case_id']]=dict(repeat_index=plan['repeat_index'],raw_output_private=g['raw_output'],generation_private=g,parsed_private=parsed)
  write(BASE/'GPU_STATUS.json',dict(status='running',done=len(rows),total=len(call_plan),GT_read=False,time=time.time()),mutable=True)
  torch.cuda.empty_cache()
 exact=all(x['exact_input_old']and x['exact_id_old']for x in rows)and all(x['repeat']['raw_text_exact_match_previous_repeat']for x in rows if x['repeat']['previous_repeat_index'])
 write(BASE/'GLOBAL_BARRIER.json',dict(status='complete',count=len(rows),GT_read=False,task_input_and_ID_exact_old_and_repeat_text_exact=exact,old_text_exact_required=False,rows=rows,model_load_count=1,model_load_s=load_s,actual_device_map=mapping,weight_readback=verified,original_generation_config=original,only_generate_override={'use_cache':True},max_new_tokens=2048,wall_seconds=time.perf_counter()-start,time=time.time()))
 print('KV task-output qualification complete; exact=',exact)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  write(BASE/'FAILURE_CAPTURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
