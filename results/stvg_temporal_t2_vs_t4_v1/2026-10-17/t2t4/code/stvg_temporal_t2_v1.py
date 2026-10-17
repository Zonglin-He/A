"""Isolated two-candidate Temporal control; original T4 modules stay immutable."""
from pathlib import Path
import copy,gc,hashlib,heapq,inspect,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_temporal_t2_vs_t4_v1'
CORE=ROOT/'artifacts/stvg_spatial_controls256_v1'
REPAIR=ROOT/'artifacts/stvg_spatial_caption_repair_v1'
def put(p,d,replace=False):return write(p,d,mutable=replace)
def verify():
 d=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<d['deadline_epoch']
 for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
 return d

def select(case,available):
 """Joint MAP over the two untouched frozen legal grids, excluding Native.

 The score of a pair is the sum of the original offset-grid log scores.
 Sorted Cartesian products are traversed best first; ties retain original
 offset-grid row indices. No expand/contract formula or label is consulted.
 """
 import numpy as np
 from methods.decota_final_simplified_v1.objectives import project
 from scripts.stvg_opd_p6_existing_common001 import config
 from scripts import stvg_table1_temporal_adapter_v1 as a
 ev=project(case['zero']['logits'],case['zero']['actions'],case['records'],config(case['dataset']))
 native=list(case['Native_interval']);rows=[]
 try:a._sample_candidate_ids({'interval_frames':native},available)
 except ValueError:return [],dict(reason='Native_fewer_than_eight_legal_frames')
 for e,rec in zip(ev['offsets'],case['records']):
  score=e['score'].detach().double().cpu().numpy();ij=e['ij'].cpu().numpy();order=np.argsort(-score,kind='stable')
  rows.append([(float(score[k]),int(k),[int(rec['frame_ids'][ij[0,k]]),int(rec['frame_ids'][ij[1,k]])+1]) for k in order])
 assert len(rows)==2 and all(rows)
 def item(i,j):return (-math.fsum((rows[0][i][0],rows[1][j][0])),rows[0][i][1],rows[1][j][1],i,j)
 heap=[item(0,0)];seen={(0,0)};visited=0
 while heap:
  neg,k0,k1,i,j=heapq.heappop(heap);visited+=1;x,y=rows[0][i][2],rows[1][j][2];iv=[min(x[0],y[0]),max(x[1],y[1])]
  if iv!=native and available[0]<=iv[0]<iv[1]<=available[-1]+1:
   try:a._sample_candidate_ids({'interval_frames':iv},available)
   except ValueError:pass
   else:
    def candidate(key,interval):return dict(candidate_id=key,canonical_id=key,role=key.removeprefix('time_'),interval_frames=interval,interval=list(interval))
    return [candidate('time_Native',native),candidate('time_Alternative',iv)],dict(status='selected',joint_frozen_grid_score=-neg,offset_grid_indices=[k0,k1],ranked_joint_pairs_visited=visited,GT_read=False)
  for ni,nj in ((i+1,j),(i,j+1)):
   if ni<len(rows[0]) and nj<len(rows[1]) and (ni,nj) not in seen:heapq.heappush(heap,item(ni,nj));seen.add((ni,nj))
 return [],dict(reason='no_distinct_legal_frozen_grid_alternative',ranked_joint_pairs_visited=visited)

def contracts():
 import numpy as np
 from scripts import stvg_evidence_temporal_fit_v1 as b,stvg_temporal_lite_fit_v1 as lite
 case,hidden,candidates,rewards=b._synthetic_case(seed=42016);candidates=candidates[:2];rewards=rewards[:2]
 pre=b._prepare(case,hidden,candidates,rewards);cfg=lite._lite_config();theta=np.random.default_rng(42017).normal(0,.001,512)
 loss,grad,terms=lite._shared_candidate_loss_gradient(pre,theta,'offset_trust',cfg);error=0.
 for k in range(512):
  d=np.zeros(512);d[k]=1e-6
  fd=(lite._shared_candidate_loss_gradient(pre,theta+d,'offset_trust',cfg)[0]-lite._shared_candidate_loss_gradient(pre,theta-d,'offset_trust',cfg)[0])/2e-6;error=max(error,abs(float(fd-grad[k])))
 assert error<4e-7 and terms['candidate_kl_multiplier']==2 and cfg['lambda_grid']==1 and cfg['l2']==0
 out=lite.fit(case,hidden,candidates,rewards,'temporal_lite');assert out['active_parameters']==512 and out['config']==cfg
 for values in ([.2,.2],[None,None]):
  z=lite.fit(case,hidden,candidates,values,'temporal_lite');assert z['optimizer_steps']==0 and z['actual_nonzero_parameter_count']==0 and z['final_interval']==case['Projection_interval']
 return dict(status='pass',count=512,finite_difference_max_abs_error=error,original_LBFGS_Armijo_full_grid_and_scaling_unchanged=True,two_candidate_softmax_only=True,GT_read=False)

def setup():
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import stvg_table1_temporal_adapter_v1 as a
 sys.addaudithook(guard);rows=[];old={r['case_id']:r for r in read(CORE/'temporal/INPUT_BARRIER.json')['records']}
 for r in read(BASE/'ROSTER_LOCK.json')['records']:
  cid=r['probe_alias'];case,cp=load_case(r);req=read(CORE/'temporal/requests'/(cid+'.json'));available=sorted(a._frame_map_rows(req['frame_map']));cs,proof=select(case,available)
  rows.append(dict(case_id=cid,dataset=r['dataset'],candidates=cs,selection=proof,eligible=len(cs)==2,source_path=str(cp.relative_to(ROOT)),source_sha256=sha(cp),request_path=str((CORE/'temporal/requests'/(cid+'.json')).relative_to(ROOT)),request_sha256=sha(CORE/'temporal/requests'/(cid+'.json')),cpu_blas_profile=old[cid]['cpu_blas_profile'],T4_eligible=old[cid]['eligible'],GT_read=False));del case;gc.collect()
 put(BASE/'CANDIDATE_LOCK.json',dict(status='locked',count=256,records=rows,GT_read=False))
 put(BASE/'CPU_CONTRACT_COMPLETION.json',contracts())
 protocol=dict(status='locked',count=256,per_domain=128,cohort='original core256 exposed development',arms=['Frozen','T4','T2'],primary_comparison='T2-T4 separately by domain on all128',secondary_comparison='common pre-adaptation eligibility; never replaces full-cohort table',Alternative_rule='highest summed original frozen offset-grid score for a union interval different from Native, legal bounds and >=8 physical frames; original row-index tie break',no_fallback_alternative='Native exact, zero adaptation',candidate_count=2,original_T4_unchanged=True,original_temporal_lite_optimizer='512 float64; shared-mean candidate KL times offset count; Full-grid Trust; original L-BFGS/Armijo and decode',effect_margin_pp=.2,noninferiority_criteria='paired source t95 lower bound for T2-T4 >= -0.2pp for BOTH tIoU and vIoU in EACH domain; otherwise inconclusive/not supported',cost_criteria='logical Caption/alignment pairs strictly lower AND matched fresh-work estimate lower; cache wall reported separately, not a fresh speedup claim',GT_before_prediction_seal=False,one_fixed_recipe_no_search=True,CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),GT_read=False)
 put(BASE/'METHOD_LOCK.json',protocol)
 protected=[Path(__file__),ROOT/'scripts/continue_stvg_temporal_t2_v1.py',ROOT/'scripts/score_stvg_temporal_t2_v1.py',ROOT/'scripts/stvg_temporal_lite_fit_v1.py',ROOT/'scripts/stvg_evidence_temporal_fit_v1.py',ROOT/'scripts/stvg_table1_temporal_adapter_v1.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',ROOT/'scripts/stvg_factorial_native256_v1.py',ROOT/'methods/CURRENT_METHOD.json',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'CANDIDATE_LOCK.json',BASE/'METHOD_LOCK.json',CORE/'temporal/INPUT_BARRIER.json',CORE/'temporal/TEACHER_GLOBAL_BARRIER.json',CORE/'temporal/STUDENT_GLOBAL_BARRIER.json',CORE/'temporal/ROOT_READBACK.json']
 put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=256,pins={str(p.relative_to(ROOT)):sha(p) for p in protected},deadline_epoch=time.time()+21600,GT_read=False))
 put(BASE/'STATUS.json',dict(status='running',stage='prepare_inputs',count=256,model_started=False,GT_read=False),replace=True)
 print('T2_FIXED_SELECTION_AND_MATH_LOCKED',sum(x['eligible'] for x in rows),flush=True)

def prepare():
 import numpy as np
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import stvg_table1_temporal_adapter_v1 as a,stvg_factorial_native256_v1 as raw
 verify();sys.addaudithook(guard);roster={r['probe_alias']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};jobs=[];cases=[]
 repair={r['case_id']:r for r in read(REPAIR/'INPUT_BARRIER.json')['files'] if r['cohort']=='core256'}
 repaired={r['case_id']:r for r in read(REPAIR/'CAPTION_COMPLETION.json')['files'] if r['cohort']=='core256'}
 oldbar=read(CORE/'temporal/TEACHER_GLOBAL_BARRIER.json');sealed={f['path']:f['sha256'] for f in oldbar['files']}
 for row in read(BASE/'CANDIDATE_LOCK.json')['records']:
  cid=row['case_id'];case,cp=load_case(roster[cid]);assert sha(cp)==row['source_sha256'];req=read(ROOT/row['request_path']);assert sha(ROOT/row['request_path'])==row['request_sha256'];missing=[];local=[]
  for c in row['candidates']:
   key=c['canonical_id'];sample,stride=a._sample_candidate_ids(c,sorted(a._frame_map_rows(req['frame_map'])));found=None
   for mp in sorted((CORE/'temporal/inputs'/cid).glob('*.json')):
    m=read(mp)
    if list(m['candidate']['interval_frames'])==list(c['interval_frames']) and m['metadata']['sampled_physical_frame_ids']==sample and m['metadata']['clip_fps']==float(case['input']['fps']/stride):
     tp=CORE/'temporal/teacher'/cid/(m['candidate']['canonical_id']+'.json');assert sha(tp)==sealed[str(tp.relative_to(ROOT))] and sha(ROOT/m['input_path'])==m['input_sha256'];found=dict(input_path=m['input_path'],input_sha256=m['input_sha256'],metadata=copy.deepcopy(m['metadata']),reuse_response_path=str(tp.relative_to(ROOT)),reuse_response_sha256=sha(tp),reuse_caption_path=None);break
   if found is None and key=='time_Native' and cid in repair:
    rf=repair[cid];m=read(ROOT/rf['path']);assert sha(ROOT/rf['path'])==rf['sha256'] and sha(ROOT/m['input_path'])==m['input_sha256'];cap=repaired[cid];assert sha(ROOT/cap['path'])==cap['sha256'];found=dict(input_path=m['input_path'],input_sha256=m['input_sha256'],metadata=copy.deepcopy(m['metadata']),reuse_response_path=None,reuse_caption_path=cap['path'],reuse_caption_sha256=cap['sha256'])
   j=dict(case_id=cid,dataset=row['dataset'],candidate=c,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),request_path=row['request_path'],request_sha256=row['request_sha256'],GT_read=False)
   if found is None:missing.append((j,sample))
   else:j.update(found);local.append(j)
  if missing:
   ns=dict(raw.__dict__);ns.update(BASE=BASE,CURRENT_CASE=case);exec(compile(inspect.getsource(raw.native_rgb),__file__,'exec'),ns);rgb=ns['native_rgb'](req,sorted({f for _,s in missing for f in s}))
   for j,sample in missing:
    video,meta=a._render_marked_frames(case,j['candidate'],sample,rgb);p=BASE/'private_inputs'/cid/(j['candidate']['canonical_id']+'.npy');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();np.save(p,video,allow_pickle=False);j.update(input_path=str(p.relative_to(ROOT)),input_sha256=sha(p),metadata=meta,reuse_response_path=None,reuse_caption_path=None);local.append(j)
   del rgb
  for j in sorted(local,key=lambda j:j['candidate']['canonical_id']):
   j['metadata'].update(candidate_id=j['candidate']['canonical_id'],interval_frames=j['candidate']['interval_frames']);p=BASE/'jobs'/cid/(j['candidate']['canonical_id']+'.json');put(p,j);jobs.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
  cases.append(dict(case_id=cid,dataset=row['dataset'],eligible=row['eligible'],candidate_count=len(local),GT_read=False));put(BASE/'INPUT_STATUS.json',dict(status='running',done=len(cases),total=256,count=256,GT_read=False),replace=True);del case;gc.collect()
 put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=256,records=cases,jobs=jobs,GT_read=False));print('T2_INPUTS_COMPLETE',len(jobs),flush=True)

def align_caption(model,processor,device,query,caption,candidate,meta,cap):
 import torch
 from scripts import stvg_table1_temporal_adapter_v1 as a
 batch,ps=a._processor_batch(processor,a.alignment_prompt(query,caption),device=device)
 with torch.inference_mode():out=model(**batch,use_cache=False);logits=out.logits[0,-1,a._abc_token_ids(processor)].float();probs=logits.softmax(0).cpu().tolist()
 return dict(candidate_id=candidate['canonical_id'],raw_caption=caption,caption_status='actual_plain_description' if caption else 'empty_description',caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),query_sha256=hashlib.sha256(query.encode()).hexdigest(),probabilities=probs,label_logits=logits.cpu().tolist(),reward=probs[0]-probs[1] if caption else None,alignment_prompt_sha256=ps,caption_prompt_sha256=cap['caption_prompt_sha256'],caption_output_tokens=cap['caption_output_tokens'],metadata=meta,query_is_own_case=True,teacher_weight_update=False,GT_read=False)

def teacher():
 import torch,numpy as np
 from scripts import run_stvg_native_instance_teacher_storage002 as loader,stvg_table1_temporal_adapter_v1 as a
 from scripts.stvg_qwen_uniform_clock002 import apply
 from scripts.stvg_reference_clock_control_v1 import qualify
 verify();sys.addaudithook(guard);torch.set_num_threads(2);loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
 _,model,_,device,_,load_seconds,_=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,BASE/'offload');model.generation_config.use_cache=True;proof=apply(model);processor=__import__('transformers').AutoProcessor.from_pretrained(str(ROOT/loader.CHECKPOINT_REL),local_files_only=True,trust_remote_code=False);a.configure_original_processor(processor)
 files=[];qualified=set();fresh=aligned=reused=0;tick=time.perf_counter();costs=[]
 for job in read(BASE/'INPUT_BARRIER.json')['jobs']:
  assert sha(ROOT/job['path'])==job['sha256'];m=read(ROOT/job['path']);assert sha(ROOT/m['input_path'])==m['input_sha256'];query=read(ROOT/m['request_path'])['query_text'];video=np.load(ROOT/m['input_path'],allow_pickle=False);meta=m['metadata'];clip=dict(video=video,fps=meta['clip_fps'],metadata=meta);started=time.perf_counter();qualified_new=False
  _,ch=a._processor_batch(processor,a.caption_prompt(),video=video,fps=clip['fps'])
  if m['reuse_response_path']:
   assert sha(ROOT/m['reuse_response_path'])==m['reuse_response_sha256'];old=read(ROOT/m['reuse_response_path']);_,ah=a._processor_batch(processor,a.alignment_prompt(query,old['raw_caption']))
   for k in ('sampled_physical_frame_ids','marked_RGB_sha256','clip_fps','visual_ID'):assert old['metadata'][k]==meta[k]
   assert old['source_case_sha256']==m['source_case_sha256'] and old['query_sha256']==hashlib.sha256(query.encode()).hexdigest() and old['caption_prompt_sha256']==ch and old['alignment_prompt_sha256']==ah
   response=copy.deepcopy(old);response.update(candidate_id=m['candidate']['canonical_id'],metadata=meta);reused+=1;mode='exact_existing_pair'
  elif m['reuse_caption_path']:
   assert sha(ROOT/m['reuse_caption_path'])==m['reuse_caption_sha256'];cap=read(ROOT/m['reuse_caption_path']);assert cap['caption_prompt_sha256']==ch and cap['metadata']['marked_RGB_sha256']==meta['marked_RGB_sha256'];response=align_caption(model,processor,device,query,cap['raw_caption'],m['candidate'],meta,cap);aligned+=1;mode='existing_caption_new_alignment'
  else:response=a.run_hot_candidate_teacher(model,processor,query,m['candidate'],clip,device=device);fresh+=1;mode='fresh_caption_alignment_pair'
  if m['dataset'] not in qualified:
   batch,_=a._processor_batch(processor,a.caption_prompt(),video=video,fps=clip['fps'],device=device);q=qualify(model,batch,dict(job_id=m['case_id']+'_'+m['candidate']['canonical_id']));del batch
   check=a.run_hot_candidate_teacher(model,processor,query,m['candidate'],clip,device=device)
   assert all(check[k]==response[k] for k in ('raw_caption','probabilities','label_logits','reward','caption_prompt_sha256','alignment_prompt_sha256','caption_output_tokens'));qualified.add(m['dataset']);qualified_new=True
   put(BASE/'qualification'/(m['dataset']+'.json'),dict(status='pass',count=1,original_generator_and_cached_response_exact=True,actual_time_encoding_readback=q,GT_read=False))
  response.update(case_id=m['case_id'],source_case_sha256=m['source_case_sha256'],source_input_sha256=job['sha256'],reuse_mode=mode,corrected_clock_proof=proof,GT_read=False);p=BASE/'teacher'/m['case_id']/(m['candidate']['canonical_id']+'.json');put(p,response);files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),case_id=m['case_id']));costs.append(dict(case_id=m['case_id'],candidate_id=m['candidate']['canonical_id'],mode=mode,wall_seconds=time.perf_counter()-started,extra_qualification_pair=qualified_new,GT_read=False));put(BASE/'TEACHER_STATUS.json',dict(status='running',done=len(files),total=len(read(BASE/'INPUT_BARRIER.json')['jobs']),count=256,GT_read=False),replace=True);del video;gc.collect()
 assert qualified=={'hc2','vidstg'}
 put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,exact_original_teacher_both_domains=True,GT_read=False))
 put(BASE/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=256,files=files,new_caption_alignment_pairs=fresh,new_alignment_only_calls=aligned,exact_reused_pairs=reused,extra_qualification_pairs=2,load_seconds=load_seconds,wall_seconds=time.perf_counter()-tick,records=costs,GT_read=False));print('T2_TEACHER_COMPLETE',fresh,aligned,reused,flush=True)

def fits(readback=False):
 import torch
 from threadpoolctl import threadpool_limits
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case,save_pt
 from scripts import stvg_temporal_lite_fit_v1 as lite,stvg_table1_temporal_adapter_v1 as a
 verify();sys.addaudithook(guard);torch.set_num_threads(2);rows={r['case_id']:r for r in read(BASE/'CANDIDATE_LOCK.json')['records']};bar=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert bar['count']==256 and bar['status']=='complete';responses={f['path']:f['sha256'] for f in bar['files']};files=[];times=[];cost=[]
 for r in read(BASE/'ROSTER_LOCK.json')['records']:
  cid=r['probe_alias'];row=rows[cid];case,cp=load_case(r);reward=[];fit=None;tick=time.perf_counter()
  if row['eligible']:
   for c in row['candidates']:
    p=BASE/'teacher'/cid/(c['canonical_id']+'.json');assert sha(p)==responses[str(p.relative_to(ROOT))];reward.append(a._alignment_reward(c['canonical_id'],read(p)))
   with threadpool_limits(limits=row['cpu_blas_profile'],user_api='blas'):fit=lite.fit(case,a._hidden_from_source_case(case),row['candidates'],reward,'temporal_lite')
   interval=list(fit['final_interval'])
  else:interval=list(case['Native_interval'])
  d=dict(case_id=cid,dataset=r['dataset'],source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),fit=fit,rewards=reward,candidates=row['candidates'],selected_interval=interval,no_fit_reason=row['selection'].get('reason') if fit is None else None,GT_read=False);p=BASE/'student'/(cid+'.pt')
  if readback:assert lite._exact_equal(torch.load(p,map_location='cpu',weights_only=False),d);item=dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
  else:item=save_pt(p,d)
  item['case_id']=cid;files.append(item);times.append(dict(case_id=cid,dataset=r['dataset'],source_case_sha256=case['source_case_sha256'],selected_interval=interval,GT_read=False));cost.append(dict(case_id=cid,wall_seconds=time.perf_counter()-tick,fit=fit is not None,optimizer_steps=fit['optimizer_steps'] if fit else 0,GT_read=False));del case,fit;gc.collect();put(BASE/'FIT_STATUS.json',dict(status='running',stage='readback' if readback else 'fit',done=len(files),total=256,count=256,GT_read=False),replace=True)
 if readback:
  assert read(BASE/'STUDENT_GLOBAL_BARRIER.json')['files']==files;put(BASE/'ROOT_READBACK.json',dict(status='pass',count=256,full_fit_payload_exact=True,original_full_grid_decode=True,GT_read=False));put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=256,files=files,GT_read=False))
 else:put(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=256,files=files,records=cost,GT_read=False));put(BASE/'TEMPORAL_LOCK.json',dict(status='locked',count=256,records=times,GT_read=False))

if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='fit':fits()
 elif mode=='readback':fits(True)
 else:globals()[mode]()
