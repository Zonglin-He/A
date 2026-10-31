"""Root-executed R14 Drone 7arm role/mapping assay. No GT/taskscore/optimizer.
Import safe. GPU/model/image IO only behind an actual root hashed plan.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1];sys.path[:0]=[str(D),str(B/'crossplatform/round13')]
from drone_association import (DRAFT_CONFIG,asdict,parse_text_roles,text_role_messages,parse_visual_roles,
                              visual_role_messages,role_consensus,map_selected_mask_positive_support)
from association import (DEFAULT_CONFIG as OLD_CONFIG,array_sha,numbered_candidates,render_numbered_panel,
                         map_masks_to_query_groups,consensus_readout,readout_choice,first_noun_text)
from association_runner import FrozenVLM,serial,atomic_json
from input_io import read,sha,verify_seal,load_case
ARMS=('source','old_full_order0','old_full_consensus','old_noun_consensus','semantic_only','mapping_only','combined')

class DroneFrozenVLM(FrozenVLM):
 def predict_messages(self,messages,images,max_new_tokens):
  t=self.torch;text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  kw={'text':[text],'return_tensors':'pt','padding':False,'truncation':False}
  if images:kw['images']=images
  ins=self.processor(**kw).to('cuda')
  if ins.input_ids.shape[1]>8192:raise RuntimeError('locked8192prompt cap; no silent truncation')
  grid=ins['image_grid_thw'].tolist() if 'image_grid_thw' in ins else []
  assert len(grid)==len(images) and all(g[0]==1 and g[1]*g[2]//4<=1024 for g in grid)
  t.cuda.synchronize();start=time.perf_counter()
  with t.inference_mode():out=self.model.generate(**ins,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True)
  t.cuda.synchronize();tail=out[0,ins.input_ids.shape[1]:];raw=self.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False);eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  return {'raw_answer_PRIVATE':raw,'query_message_sha256':hashlib.sha256(text.encode()).hexdigest(),'image_grid_thw':grid,'input_tokens':int(ins.input_ids.shape[1]),'generated_tokens':len(tail),'max_new_tokens':max_new_tokens,'ended_eos':any(int(x) in eos for x in tail),'seconds_generate':time.perf_counter()-start,'new_actual_model_call':True}

def source_readout(q,boxes,status):
 return {'q_selected':int(q),'box_raw6':boxes[q].copy(),'readout_status':status,'original_source_box':True}

def main(a):
 start=time.perf_counter();assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
 assert plan['stage']=='R14_Drone_subject_roles_and_positive_support_mapping' and plan['GT_read_permitted'] is False and plan['model_execution_permitted'] and plan['GPU_execution_permitted']
 assert plan['drone_config']==asdict(DRAFT_CONFIG) and plan['legacy_association_config']==asdict(OLD_CONFIG) and plan['expected_cases']==64 and plan['formal_expected_new_calls']==320 and plan['formal_expected_reused_calls']==128
 for rel,h in plan['program_sha256'].items():assert sha(B/rel)==h
 assert plan['program_sha256']['crossplatform/round14/drone_runner.py']==sha(__file__)
 data={}
 for k in ('manifest','case_index','frame_index'):
  assert sha(plan[k])==plan[k+'_sha256'];data[k]=read(plan[k])
 rows=data['manifest'];cases={c['id']:c for c in data['case_index']};oldids={c['id'] for c in data['case_index'] if c.get('legacy_case_alias') is not None}
 assert len(rows)==len(cases)==64 and len(oldids)==32 and len({r['id'] for r in rows})==64 and oldids<=set(cases) and all(r['platform']=='drone' for r in rows)
 assert {r['id'] for r in rows[:32]}==oldids and all(r['id'] not in oldids for r in rows[32:])
 expected_seals={str((Path(c[k])/'SEALED.json').as_posix()) for c in cases.values() for k in ('source_dir','mask_dir')}
 assert len(expected_seals)==4 and set(plan['capture_seals_sha256'])==expected_seals,'Exact four old/new source/mask seals required'
 for rel,h in plan['capture_seals_sha256'].items():assert sha(B/rel)==h;verify_seal((B/rel).parent)
 oldA=Path(plan['legacy_association_dir']);assert sha(oldA/'SEALED.json')==plan['legacy_association_seal_sha256'];oldseal=verify_seal(oldA);assert oldseal['n_model_calls']==696 and oldseal['model_frozen_parameters_buffers_unchanged']
 assert oldseal['model_files_sha256']==plan['model_files_sha256'];modeldir=Path(plan['model_dir'])
 for name,h in plan['model_files_sha256'].items():assert sha(modeldir/name)==h
 selected=rows
 if a.limit is not None:
  assert a.limit==2 and plan['allow_engineering_first_old_first_new'];assert rows[0]['id'] in oldids and rows[32]['id'] not in oldids;selected=[rows[0],rows[32]]
 scope='engineering_firstold_firstnew14logical10actual4reused' if a.limit else 'formal64_320new128strictlyreused_calls';out=Path(a.output);assert not out.exists();out.mkdir(parents=True);atomic_json(out/'registered_plan.json',plan);callroot=out/'calls_PRIVATE';callroot.mkdir();caseroot=out/'cases_PRIVATE';caseroot.mkdir();journal=out/'calls.jsonl';newcalls=reused=0
 def savecall(name,record):
  p=callroot/(name+'.json');assert not p.exists();atomic_json(p,record)
  with journal.open('a') as f:f.write(json.dumps({'path':str(p.relative_to(out)),'sha256':sha(p)})+'\n');f.flush();os.fsync(f.fileno())
  return record
 engine=None
 try:
  loadstart=time.perf_counter();engine=DroneFrozenVLM(modeldir,OLD_CONFIG);atomic_json(out/'MODEL_LOAD_RECEIPT.json',{'loading_info':engine.loading_info,'seconds':time.perf_counter()-loadstart,'frozen_eval_safetensors_local_only':True,'scope':scope})
  for ri,row in enumerate(selected):
   case=cases[row['id']];alias=case['case_alias'];assert all(x.isalnum() or x=='_' for x in alias)
   frame,c,bindings=load_case(row,case,data['frame_index']);boxes=c['boxes_raw6'][0];scores=c['text_scores'][0];q0=int(c['query_index']);modelp=np.array(frame['raw_points'],np.float32,copy=True);modelp[:,2]+=np.float32(1.8)
   ins=frame['instances'];cand=[numbered_candidates(row['id'],ins['masks_bool'],ins['boxes_xyxy'],permutation=k) for k in range(2)]
   from PIL import Image
   original=Image.fromarray(frame['rgb']);panels=[render_numbered_panel(frame['rgb'],cc) for cc in cand];panelhash=[array_sha(np.asarray(p)) for p in panels];rgbhash=array_sha(frame['rgb']);noun,nounreceipt=first_noun_text(row['caption']);legacy={}
   for condition,query in (('full',row['caption']),('noun',noun)):
    legacy[condition]=[]
    for order in range(2):
     if row['id'] in oldids:
      oldalias=case['legacy_case_alias'];src=oldA/'calls'/(oldalias+'__'+condition+'__order'+str(order)+'.json');assert oldseal['files'][str(src.relative_to(oldA))]==sha(src);record=read(src);b=record['binding']
      assert b['case_id_PRIVATE']==row['id'] and b['condition']==condition and b['numbering']==order and b['bindings']==bindings and b['original_RGB_array_sha256']==rgbhash and b['panel_RGB_array_sha256']==panelhash[order] and b['query_sha256']==hashlib.sha256(query.encode()).hexdigest() and b['candidate_id_to_mask_index']=={x['candidate_id']:x['mask_index'] for x in cand[order]}
      result=record['result'];reused+=1;savecall(alias+'__old_'+condition+'__order'+str(order),{'scope':'strict_bound_old_R13_call_reuse_no_new_model_call','legacy_call_sha256':sha(src),'legacy_case_alias':oldalias,'result':result,'new_actual_model_call':False})
     else:
      result=engine.predict([original,panels[order]],query,cand[order]);newcalls+=1;savecall(alias+'__old_'+condition+'__order'+str(order),{'scope':'fresh_legacy_control','bindings':bindings,'result':result,'new_actual_model_call':True})
     legacy[condition].append(result)
   oldmap=map_masks_to_query_groups(boxes_raw6=boxes,text_scores=scores,q0=q0,points_model=modelp,camera_spec=frame['camera'],masks_bool=ins['masks_bool'],mask_boxes_xyxy=ins['boxes_xyxy'])
   oldfull=consensus_readout([x['parsed'] for x in legacy['full']],cand,oldmap,q0,boxes);oldnoun=consensus_readout([x['parsed'] for x in legacy['noun']],cand,oldmap,q0,boxes);raw0=readout_choice(legacy['full'][0]['parsed'],cand[0],oldmap,q0,boxes)
   textcall=engine.predict_messages(text_role_messages(row['caption']),[],DRAFT_CONFIG.text_role_max_new_tokens);newcalls+=1;roles=parse_text_roles(textcall['raw_answer_PRIVATE'],row['caption']);savecall(alias+'__new_text_roles',{'result':textcall,'parsed':roles,'bindings':bindings})
   vresults=[];vparsed=[]
   for order in range(2):
    v=engine.predict_messages(visual_role_messages(row['caption'],roles,[x['candidate_id'] for x in cand[order]]),[original,panels[order]],DRAFT_CONFIG.visual_role_max_new_tokens);newcalls+=1;pv=parse_visual_roles(v['raw_answer_PRIVATE'],cand[order],roles);vresults.append(v);vparsed.append(pv);savecall(alias+'__new_visual_roles_order'+str(order),{'result':v,'parsed':pv,'bindings':bindings,'panel_RGB_array_sha256':panelhash[order]})
   semantic=role_consensus(vparsed,roles,ins['class_ids']);masknew=semantic['selected_mask_index'];maskold=oldfull['selected_mask_index'] if oldfull['two_numberings_same_physical_mask'] else None
   if masknew is not None:
    selected=next(x for x in cand[0] if x['mask_index']==masknew);semantic_only=readout_choice({'candidate_id':selected['candidate_id'],'parse_status':'valid_candidate_ID'},cand[0],oldmap,q0,boxes)
   else:semantic_only=source_readout(q0,boxes,'fallback_q0_'+semantic['status'])
   cache={}
   def newmap(mask):
    if mask not in cache:cache[mask]=map_selected_mask_positive_support(boxes_raw6=boxes,text_scores=scores,q0=q0,points_model=modelp,camera_spec=frame['camera'],masks_bool=ins['masks_bool'],selected_mask_index=mask)
    return cache[mask]
   mapping_only=newmap(maskold);combined=newmap(masknew);qs=np.asarray([q0,raw0['q_selected'],oldfull['q_selected'],oldnoun['q_selected'],semantic_only['q_selected'],mapping_only['q_selected'],combined['q_selected']],np.int64);armboxes=boxes[qs].copy();assert np.array_equal(armboxes,boxes[qs])
   statuses=['source',raw0['readout_status'],oldfull['readout_status'],oldnoun['readout_status'],semantic_only['readout_status'],mapping_only['mapping_status'],combined['mapping_status']]
   with (caseroot/(alias+'.npz')).open('wb') as f:np.savez_compressed(f,sample_id=np.asarray(row['id']),case_alias=np.asarray(alias),case_role=np.asarray('old_R13_Drone32_development' if row['id'] in oldids else 'recent_content_disjoint32_historically_exposed'),arm_names=np.asarray(ARMS),arm_query_index=qs,arm_boxes_raw6=armboxes,arm_status=np.asarray(statuses),source_boxes_raw6=boxes,source_text_scores256=scores,q_source=np.asarray(q0),actual_changed_axes=armboxes!=boxes[q0])
   atomic_json(caseroot/(alias+'.json'),{'id_PRIVATE':row['id'],'case_alias':alias,'bindings':bindings,'old_control_reused':row['id'] in oldids,'noun_control':nounreceipt,'text_roles':roles,'semantic_consensus':semantic,'old_full_consensus':oldfull,'old_noun_consensus':oldnoun,'semantic_only':semantic_only,'mapping_only':mapping_only,'combined':combined,'oldmapping':oldmap,'selected_new_mask_prediction_class':int(ins['class_ids'][masknew]) if masknew is not None else None,'no_GT_parameter_update_or_box_regression':True})
   atomic_json(out/'PROGRESS.json',{'scope':scope,'completed_cases':ri+1,'expected':len(selected),'new_model_calls':newcalls,'strictly_reused_old_calls':reused})
   if (ri+1)%5==0 or a.limit:print(json.dumps({'event':'Drone_progress','n':ri+1,'expected':len(selected),'new_calls':newcalls,'reuse':reused}),flush=True)
  assert engine.verify_unchanged()
  assert (newcalls,reused)==((10,4) if a.limit else (320,128))
  files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SEALED.json'};atomic_json(out/'SEALED.json',{'state':'sealed_R14_Drone_first2_engineering_no_GT' if a.limit else 'sealed_R14_Drone64_seven_direct_arms_before_GT','n_cases':len(selected),'arm_names':list(ARMS),'new_model_calls':newcalls,'reused_old_R13_calls':reused,'logical_calls':newcalls+reused,'plan_sha256':sha(a.plan),'program_sha256':plan['program_sha256'],'capture_seals_sha256':plan['capture_seals_sha256'],'model_files_sha256':plan['model_files_sha256'],'coarse_guard_only_reject_no_reselect':True,'no_parameter_update_GT_or_newbox':True,'model_state_unchanged':True,'seconds_process':time.perf_counter()-start,'peak_allocated_bytes':int(engine.torch.cuda.max_memory_allocated()),'files':files});print(json.dumps({'state':'Drone_sealed','n':len(selected),'sha256':sha(out/'SEALED.json')}))
 except Exception as e:
  atomic_json(out/'FAILURE.json',{'state':'failed_Drone_attempt_preserved','error':str(e),'traceback_PRIVATE':traceback.format_exc(),'new_model_calls_completed':newcalls,'reused_calls':reused,'no_automatic_retry_or_semantic_repair':True});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);main(p.parse_args())
