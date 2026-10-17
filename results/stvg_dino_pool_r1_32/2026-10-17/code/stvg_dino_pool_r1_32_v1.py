"""Finite same cached-DINO pool Top1 versus fixed R1-A selection, no GT decisions."""
from pathlib import Path
import copy,collections,gc,hashlib,inspect,json,math,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_dino_pool_r1_32_v1';FACTOR=ROOT/'artifacts/stvg_geometry_config32_v1';OLD=ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap';INPUT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2'
def put(p,d,replace=False):write(p,d,mutable=replace)
def verify():
 r=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<r['deadline_epoch']
 for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
 return r
def setup():
 assert read(FACTOR/'FINAL_COMPLETION.json')['status']=='complete';BASE.mkdir(exist_ok=False)
 for n in ['ROSTER_LOCK.json','REFERENCE_LOCK.json']:(BASE/n).write_bytes((FACTOR/n).read_bytes())
 put(BASE/'METHOD_LOCK.json',dict(status='locked',count=32,domains={'hc2':16,'vidstg':16},exposed_historical_development=True,common_positions=38,supported_sources=19,unsupported_noop_retained=13,pool='exact cached processed_probe three geometric candidates and target_scores; historical Top1 first, remaining stable descending target score and candidate ID; no new threshold, no raw phrase score substitution',selector='one frozen original query-only A global64/local96; original crop Prompt/local5/context4/physical uniform-clock adapter, max96 greedy output; no Caption requirement',ID_order='Top1 first then descending score; fixed, not randomized; possible position bias must be disclosed',center_only_OPD_anchors=True,adjacent_KLT='existing track_adjacent used for visualization only; lost track shown absent, never invent a box',profiles=['current5','original'],all_episodic=True,response_filter=False,acceptance='original reference IoU/finite/parameter+Adam rollback/reject-stop',maximum_new_WHO=38,extra_short_qualification_calls=6,maximum_new_scientific_fits=38,new_DINO_forward=0,new_Temporal=0,new_Caption=0,Top1_cells='exact sealed geometry/config factorial reuse, old costs not zero',selection_not_conditioned_on_factorial_scores=True,GT_read=False))
 put(BASE/'STATUS.json',dict(status='preparing_inputs',count=32,GT_read=False));print('DINO_POOL32_PLAN_LOCKED')
def decode(case,wanted):
 import numpy as np
 from scripts.run_stvg_table1_native_instance_adapter_v1 import _ffmpeg_identity
 ff,_=_ffmpeg_identity();p=Path(case['input']['video_path']);assert sha(p)==case['input']['video_sha256'];w,h=int(case['width']),int(case['height']);cmd=[str(ff),'-v','error','-threads','1','-i',str(p)]
 if case['dataset']=='vidstg':cmd+=['-vsync','0']
 cmd+=['-f','rawvideo','-pix_fmt','rgb24','pipe:1'];proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);out={};i=0;n=w*h*3
 try:
  while wanted-set(out):
   raw=bytearray()
   while len(raw)<n:
    part=proc.stdout.read(n-len(raw))
    if not part:raise RuntimeError('raw source ended before locked frame')
    raw.extend(part)
   if i in wanted:out[i]=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3).copy()
   i+=1
 finally:
  proc.stdout.close()
  if proc.poll() is None:proc.terminate()
  proc.wait(timeout=20)
 return out
def xyxy(box):
 x,y,w,h=map(float,box);return [x-w/2,y-h/2,x+w/2,y+h/2]
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));v=x*y;return v/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v)
def inputs():
 import cv2,numpy as np
 from scripts.stvg_reference_recovery_common_v1 import load_case
 from scripts.stvg_adjacent_affine_tracklet_v1 import track_adjacent
 from scripts import stvg_evidence_spatial_inputs005 as render
 from scripts.stvg_reference_recovery_fit_v1 import source_hashes
 assert read(BASE/'METHOD_LOCK.json')['status']=='locked';sys.addaudithook(guard);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);jobs=[];pools=[];files=[];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};oldjobs=read(INPUT.parent/'round2_prompt009/TEACHER_JOB_LOCK.json')['records'];dinos=[read(OLD/r['path']) for r in read(OLD/'TEACHER_GLOBAL_BARRIER.json')['files'] if r.get('decision_status')=='accepted'];rindex={(r['case_id'],r['position']):r for r in dinos}
 for rec in read(BASE/'ROSTER_LOCK.json')['records']:
  cid=rec['case_id'];case,cp=load_case(rec);ref=refs[cid];proposal=read(INPUT/'new_crop/proposals'/(cid+'.json')) if ref['common_dino_anchors'] else None;wanted={f for a in ref['common_dino_anchors'] for f in range(a['physical_frame_id']-2,a['physical_frame_id']+3)};rgb=decode(case,wanted) if wanted else {};request=read(ROOT/'artifacts/stvg_weightfree_tuning_v1/confirmation003/requests'/(cid+'.json'));assert request['source_video_sha256']==case['input']['video_sha256'];fm={int(r['physical_frame_id']):r for r in request['frame_map']};inp=BASE/'teacher_inputs'/cid;inp.mkdir(parents=True,exist_ok=True)
  for a in ref['common_dino_anchors']:
   j=copy.deepcopy(next(j for j in oldjobs if j['case_id']==cid and j['variant']=='new_crop' and j['center_physical_frame_id']==a['physical_frame_id']));window=next(w for w in proposal['windows'] if w['center_physical_frame_id']==a['physical_frame_id']);ids=window['physical_frame_ids'];assert len(ids)==5 and ids==list(range(ids[0],ids[0]+5));arr=[rgb[f] for f in ids]
   for fid,im in zip(ids,arr):assert hashlib.sha256(im.tobytes()).hexdigest()==window['rgb_sha256_by_physical_frame'][str(fid)],(cid,fid)
   raw=rindex[cid,a['position']];mid=ids.index(a['physical_frame_id']);assert hashlib.sha256(arr[mid].tobytes()).hexdigest()==raw['input_audit']['source_image_sha256'];boxes=raw['raw_boxes_cxcywh'];scores=raw['raw_phrase_scores'];assert len(boxes)==len(scores)==900;top=raw['selected_box_cxcywh'];assert np.array_equal(np.asarray(top,dtype=np.float32),np.asarray(a['box_cxcywh'],dtype=np.float32))
   probe=raw['processed_probe'];pb=probe['boxes_cxcywh'];ps=probe['target_scores'];pi=probe['candidate_ids'];assert len(pb)==len(ps)==len(pi)==3;order=sorted(range(3),key=lambda i:(-float(ps[i]),int(pi[i])));assert pb[order[0]]==top and ps[order[0]]==raw['selected_score'];pool=[dict(expert_candidate_id=pi[i],box_cxcywh=pb[i],score=ps[i]) for i in order]
   h,w=arr[0].shape[:2];scale=min(1,640/max(h,w));gray=[cv2.cvtColor(cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY) for im in arr];tracks=[]
   for k,entry in enumerate(pool,1):
    raw_display=xyxy(entry['box_cxcywh']);assert -1e-7<=min(raw_display) and max(raw_display)<=1+1e-7;display=[max(0.,min(1.,v)) for v in raw_display];track=track_adjacent(gray,ids,mid,display);bb=[r['box_xyxy_norm'] if r is not None else None for r in track['trajectory']];assert np.allclose(bb[mid],display,rtol=0,atol=1e-12);tracks.append(dict(id=k,candidate_id=k,role='DINO_geometry',boxes=bb,center_box=bb[mid],presence=sum(x is not None for x in bb)/5,tracklet=track))
   masks=np.zeros((max(0,len(tracks)-1),5,1,1),np.uint8);video=render._render_crop(arr,tracks,masks);path=inp/(str(a['position'])+'.npy');
   if path.exists():assert np.array_equal(np.load(path,allow_pickle=False),video)
   else:np.save(path,video,allow_pickle=False)
   times=[float(fm[f]['source_timestamp_s']) for f in ids];v=render._video_record(str(path.relative_to(ROOT)),path.read_bytes(),video,times,'candidate_crop_contact');context=j['videos'][0];assert sha(ROOT/context['path'])==context['sha256'];jobid=f'{cid}_dino_{a["position"]}';j.update(job_id=jobid,variant='DINO_pool_fixed_A',candidate_bindings=tracks,candidate_count=len(tracks),allowed_output_ids=[0,*range(1,len(tracks)+1)],visible_id_to_candidate={str(k):k for k in range(1,len(tracks)+1)},videos=[context,v]);j['prompt']=render._confirmation003_prompt(request['query_text'],context['actual_timestamps_seconds'],times,list(range(1,len(tracks)+1)));j['prompt_sha256']=hashlib.sha256(j['prompt'].encode()).hexdigest();j['source_grid_position']=a['position'];j['reference_weight']=a['weight'];j['query_text']=request['query_text'];j['width']=w;j['height']=h;row=dict(job_id=jobid,case_id=cid,dataset=rec['dataset'],position=a['position'],physical_frame_id=a['physical_frame_id'],weight=a['weight'],candidates=pool,top1_box_cxcywh=top,raw_DINO_input_sha256=raw['input_audit']['source_image_sha256'],actual_all5_RGB_match_historical=True,GT_read=False);pools.append(row);jobs.append(j);files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
  put(BASE/'STATUS.json',dict(status='preparing_inputs',done=len([r for r in read(BASE/'ROSTER_LOCK.json')['records'] if r['case_id']<=cid]),count=32,actual_jobs=len(jobs),GT_read=False),replace=True)
 assert len(jobs)==38;put(BASE/'TEACHER_JOB_LOCK.json',dict(status='locked',count=32,records=jobs,GT_read=False));put(BASE/'POOL_LOCK.json',dict(status='locked',count=32,records=pools,GT_read=False));put(BASE/'INPUT_BARRIER.json',dict(status='complete',count=32,jobs=38,files=files,all5_RGB_and_DINO_center_RGB_exact=True,GT_read=False))
 pins=source_hashes();paths=[Path(__file__),ROOT/'scripts/run_stvg_dino_pool_r1_32_v1.py',BASE/'METHOD_LOCK.json',BASE/'TEACHER_JOB_LOCK.json',BASE/'POOL_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'REFERENCE_LOCK.json',FACTOR/'STUDENT_GLOBAL_BARRIER.json',ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json',ROOT/'scripts/stvg_referent_abc_native002.py',ROOT/'scripts/stvg_referent_abc_tokens002.py',ROOT/'scripts/stvg_qwen_uniform_clock002.py',ROOT/'scripts/stvg_adjacent_affine_tracklet_v1.py'];pins.update({str(p.relative_to(ROOT)):sha(p) for p in paths});put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=32,deadline_epoch=time.time()+14400,pins=pins,GT_read=False));print('DINO_POOL_INPUT38_EXACT_RGB_COMPLETE')
def prepare(m,p,device,j):
 import torch
 from scripts.stvg_referent_abc_native002 import processor_batch,pack,geometry
 from scripts.stvg_referent_abc_tokens002 import select
 b,ps=processor_batch(p,j,ROOT,device);v,ma,gr=geometry(j,j['width'],j['height'])
 with torch.inference_mode():e,pos,delta,f,runs=pack(m,b);q=m.get_input_embeddings()(p.tokenizer(j['query_text'],add_special_tokens=False,return_tensors='pt')['input_ids'].to(device))[0].float().cpu().numpy()
 z=f.float().cpu().numpy();pol=[select(z[a:d].reshape(len(v[i]),256,-1),q,q,v[i],ma[i],gr[i]) for i,(a,d) in enumerate(((0,512),(512,1280)))];return b,ps,e,pos,f,runs,pol
def teacher():
 import torch
 from scripts import run_stvg_referent_abc002 as loader
 from scripts.stvg_referent_abc_native002 import sequence_keep,greedy
 from scripts.run_stvg_spatial_who_order_v1 import parse
 verify();sys.addaudithook(guard);loader.BASE=BASE;m,p,device,proof,secs=loader.load_teacher();qualified=set();files=[];jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records']
 for j in jobs:
  verify();tick=time.perf_counter();b,ps,e,pos,f,runs,pol=prepare(m,p,device,j);ix=sequence_keep(b,m,runs,[x['indices']['A'] for x in pol])
  with torch.inference_mode():
   if j['dataset'] not in qualified:
    m.model.rope_deltas=None;original=m.generate(**b,max_new_tokens=8,do_sample=False,use_cache=True)[0,b['input_ids'].shape[1]:].cpu().tolist();ours,_=greedy(m,e,pos,sequence_keep(b,m,runs,[range(512),range(768)]),b['input_ids'],8);assert original==ours;_,audit=greedy(m,e,pos,ix,b['input_ids'],8,check_recompute=True);assert audit['cached_uncached_next_token_checks']>=2;put(BASE/'qualification'/(j['dataset']+'.json'),dict(status='pass',count=1,unpruned_native8_exact=True,compressed_cached_uncached_checks=audit['cached_uncached_next_token_checks'],extra_short_generations=3,GT_read=False));qualified.add(j['dataset'])
   tokens,audit=greedy(m,e,pos,ix,b['input_ids'],96)
  raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip();path=BASE/'teacher'/(j['job_id']+'.json');put(path,dict(status='complete',job_id=j['job_id'],case_id=j['case_id'],raw_output=raw,parsed=parse(raw,j),job_sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),sequence_readback=audit,visual_token_count=sum(len(x['indices']['A']) for x in pol),visual_indices=[list(map(int,x['indices']['A'])) for x in pol],prompt_sha256=ps,clock_proof=proof,output_count=len(tokens),truncated=len(tokens)>=96,wall_seconds=time.perf_counter()-tick,GT_read=False));files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size));put(BASE/'STATUS.json',dict(status='running',stage='Teacher',done=len(files),total=38,count=32,GT_read=False),replace=True);del b,e,pos,f,runs,pol;gc.collect();torch.cuda.empty_cache()
 assert qualified=={'hc2','vidstg'};put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass',count=2,new_short_generation_calls=6,GT_read=False));put(BASE/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,response_count=38,files=files,new_WHO=38,new_visual_encoder_calls=38,model_load_seconds=secs,GT_read=False))
def teacher_root():
 from scripts.run_stvg_spatial_who_order_v1 import parse
 verify();jobs={j['job_id']:j for j in read(BASE/'TEACHER_JOB_LOCK.json')['records']};pools={p['job_id']:p for p in read(BASE/'POOL_LOCK.json')['records']};out={r['case_id']:[] for r in read(BASE/'ROSTER_LOCK.json')['records']};decisions=[]
 for f in read(BASE/'TEACHER_GLOBAL_BARRIER.json')['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=read(ROOT/f['path']);j=jobs[d['job_id']];assert d['parsed']==parse(d['raw_output'],j) and d['job_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest() and d['visual_token_count']==160 and d['sequence_readback']['all_nonvisual_tokens_preserved'];chosen=d['parsed'].get('selected_candidate_id');pool=pools[d['job_id']];box=None
  if chosen is not None:
   box=pool['candidates'][int(chosen)-1]['box_cxcywh'];out[j['case_id']].append(dict(position=pool['position'],physical_frame_id=pool['physical_frame_id'],weight=pool['weight'],box_cxcywh=box))
  decisions.append(dict(job_id=j['job_id'],case_id=j['case_id'],status=d['parsed']['status'],selected_id=chosen,same_Top1=chosen==1,pool_size=len(pool['candidates']),GT_read=False))
 put(BASE/'R1_REFERENCE_LOCK.json',dict(status='locked',count=32,records=[dict(case_id=k,anchors=v) for k,v in out.items()],decisions=decisions,GT_read=False));put(BASE/'TEACHER_ROOT_READBACK.json',dict(status='pass',count=32,response_count=38,all_selected_boxes_exact_pool_member=True,GT_read=False))
def worker():
 import numpy as np,torch
 from scripts import stvg_reference_recovery_common_v1 as common,run_stvg_reference_swap_student005 as old,run_stvg_adjacent_motion_student_v1 as h,stvg_target_teacher_student_v1 as core,stvg_pipeline_speed_v1 as speed,run_stvg_table1_source_v1 as src
 from scripts.stvg_reference_recovery_fit_v1 import fit_reference_recovery,locked_config
 from scripts.run_stvg_table1_student004 import stable_fit_key
 from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
 from scripts.run_decota_paper_main_v1 import model_for,frames_for
 from scripts.stvg_mllm_feedback_common_v1 import raw_row
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward,transient_native_masks
 from vg_tta.stvg_opd_frozen_visual_host011 import frozen_visual_host
 from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_attention_storage
 from vg_tta.stvg_opd_frozen_module_cpu008 import inactive_modules_on_host
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from methods.decota_final_simplified_v1.backbone import query_subject
 verify();assert read(BASE/'TEACHER_ROOT_READBACK.json')['status']=='pass';sys.addaudithook(guard);install_clean_loader();torch.set_num_threads(2);torch.manual_seed(20261014);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 r1={r['case_id']:r['anchors'] for r in read(BASE/'R1_REFERENCE_LOCK.json')['records']};refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};orig={r['probe_alias']:r for r in read(ROOT/'artifacts/stvg_reference_recovery_v1/Optimizer_replay/ROSTER_LOCK.json')['records']};prior={r['case_id']:r for r in read(FACTOR/'STUDENT_GLOBAL_BARRIER.json')['files']};oldprior={Path(r['path']).stem:r for r in read(FACTOR/'BASELINE_BARRIER.json')['files']};files=[];model=current=None;qualified=set();capture_calls=scientific_fits=reused_identical_queries=0
 for rec in read(BASE/'ROSTER_LOCK.json')['records']:
  verify();cid=rec['case_id'];ds=rec['dataset'];raw=r1[cid];case,cp=common.load_case(rec);ids=list(map(int,case['frame_ids']));p=ROOT/prior[cid]['path'];assert sha(p)==prior[cid]['sha256'];before=torch.load(p,map_location='cpu',weights_only=False);oldp=ROOT/oldprior[cid]['path'];assert sha(oldp)==oldprior[cid]['sha256'];oldbefore=torch.load(oldp,map_location='cpu',weights_only=False);preds={a:copy.deepcopy(before['arms'][a]) for a in ['F','T']};fits={};outs={};same=[{k:a[k] for k in ['position','physical_frame_id','weight','box_cxcywh']} for a in raw]==[{k:a[k] for k in ['position','physical_frame_id','weight','box_cxcywh']} for a in refs[cid]['common_dino_anchors']];cap=None
  for profile,suffix in [('current5','SHORT'),('original','ORIGINAL')]:
   for pre in ['S_','TS_']:preds[pre+'Top1_'+suffix]=copy.deepcopy(before['arms'][pre+'DINO_common_'+suffix])
  if same:
   reused_identical_queries+=1
   for profile,suffix in [('current5','SHORT'),('original','ORIGINAL')]:
    fits[suffix]=copy.deepcopy((oldbefore if suffix=='SHORT' else before)['fits']['DINO_common']);outs[suffix]=None
    for pre in ['S_','TS_']:preds[pre+'R1_'+suffix]=copy.deepcopy(preds[pre+'Top1_'+suffix])
  elif raw:
   transient=dict(case,source=src.SOURCE_MODEL_NAME[ds],_source_case_sha256=case['source_case_sha256']);_,row=src._validate_roster_record(orig[cid]);transient['parent']=row['source'];saved=h.catalog;h.catalog=lambda _: (transient,{})
   try:cap=h._capture_source(dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256']),model,current,torch,collections,model_for,frames_for,raw_row,frozen_forward,prediction,state_hash)
   finally:h.catalog=saved
   capture_calls+=1;model,current=cap[:2];records,batch,base,native=cap[7],cap[8],cap[9],cap[10];assert cap[6]==ids and cap[12]==case['pixel_sha256'];assert np.array_equal(h._xyxy(native['boxes']),before['arms']['F']['boxes']);initial={k:v.detach().cpu().clone() for k,v in base.initial.items()};key=stable_fit_key(ds,rec['query_key_sha256']);anchors=old._matched_anchors(raw,ids,torch)
   for profile,suffix in [('current5','SHORT'),('original','ORIGINAL')]:
    def checkpoint(payload):return common.save_pt(BASE/'checkpoints'/cid/suffix/(str(payload['next_step'])+'.pt'),payload)
    f=fit_reference_recovery(base,initial,anchors,ids,key,locked_config(ds,profile),profile=profile,dataset=ds,validate=True,reference_sha256=sha(BASE/'R1_REFERENCE_LOCK.json'),checkpoint_callback=checkpoint);scientific_fits+=1;out=speed.full_prefix_readout(base,ids,records,f['final_state']);assert torch.equal(out['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu()) and h._same_logits(out['logits'],native['logits'])
    if ds not in qualified:
     actual=old._full_readout(h,model,batch,ids,records,f,cap[4]['parses']['subject'],core,torch,transient_native_masks,frozen_visual_host,frozen_attention_storage,inactive_modules_on_host,query_subject);speed.compare_tree(actual,out,'actual_RGB_vs_prefix_readout');qualified.add(ds);put(BASE/'qualification'/(ds+'_STUDENT_READOUT.json'),dict(status='pass',count=1,actual_full_RGB_vs_prefix_exact=True,GT_read=False))
    fits[suffix]=f;outs[suffix]=out
    for pre in ['S_','TS_']:preds[pre+'R1_'+suffix]=dict(boxes=h._xyxy(out['boxes']),interval=list(preds['F' if pre=='S_' else 'T']['interval']))
   assert state_hash(model.state_dict())==case['source_model_state_sha256'];del base,records,batch,initial
  else:
   for suffix in ['SHORT','ORIGINAL']:
    fits[suffix]=outs[suffix]=None
    for pre in ['S_','TS_']:preds[pre+'R1_'+suffix]=copy.deepcopy(preds['F' if pre=='S_' else 'T'])
  dest=BASE/'student'/(cid+'.pt');payload=dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256'],reference_records=refs[cid],r1_anchors=raw,frame_ids=ids,arms=preds,fits=fits,readouts=outs,whole_Query_Top1_reused=same,baseline_sha256=prior[cid]['sha256'],GT_read=False);r=common.save_pt(dest,payload);r['case_id']=cid;files.append(r);put(BASE/'STATUS.json',dict(status='running',stage='Student',done=len(files),total=32,count=32,GT_read=False),replace=True);del case,before,oldbefore,payload,fits,outs,cap;gc.collect();torch.cuda.empty_cache()
 assert scientific_fits<=38;put(BASE/'STUDENT_QUALIFICATION_COMPLETION.json',dict(status='pass',count=len(qualified),actual_full_RGB_qualification_calls=len(qualified),if_no_changed_references_qualification='inherited exact baseline whole-Query output',GT_read=False));put(BASE/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,new_scientific_fits=scientific_fits,exact_Top1_whole_Query_reuses=reused_identical_queries,source_capture_calls=capture_calls,GT_read=False))
def root():
 import torch,numpy as np
 from scripts import review_stvg_reference_carry_optimizer_v1 as audit
 verify();bar=read(BASE/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete' and bar['count']==32;code=inspect.getsource(audit._audit_fit).replace('fit["profile"] == "current5"','fit["profile"] in ("current5", "original")').replace('== 5','== config["steps"]');ns=dict(audit.__dict__);exec(compile(code,'variable-profile-audit','exec'),ns);rr={r['case_id']:r for r in read(BASE/'ROSTER_LOCK.json')['records']};checks=0
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
  for suffix,fit in d['fits'].items():
   if fit and not d['whole_Query_Top1_reused']:
    assert fit['reference_sha256']==sha(BASE/'R1_REFERENCE_LOCK.json');ns['_audit_fit'](fit,d['r1_anchors'],rr[d['case_id']],fit['config'],sha(BASE/'R1_REFERENCE_LOCK.json'),torch,np);checks+=len(fit['path'])
   elif fit:
    assert d['whole_Query_Top1_reused'] and fit['reference_sha256'] in [sha(OLD/'REFERENCE_LOCK.json'),sha(FACTOR/'REFERENCE_LOCK.json')]
   else:assert not d['r1_anchors']
   for pre in ['S_','TS_']:assert d['arms'][pre+'R1_'+suffix]['interval']==d['arms']['F' if pre=='S_' else 'T']['interval']
   if d['whole_Query_Top1_reused']:
    for pre in ['S_','TS_']:assert np.array_equal(d['arms'][pre+'R1_'+suffix]['boxes'],d['arms'][pre+'Top1_'+suffix]['boxes'])
 put(BASE/'ROOT_READBACK.json',dict(status='pass',count=32,declared_step_math_checks=checks,GT_read=False));put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=bar['files'],GT_read=False))
def score():
 # Same metric/GT reader; every new prediction sealed before this call.
 import torch
 from scripts.stvg_reference_recovery_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass';bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();rows=[];usage=[];quality=[]
 for f in bar['files']:
  assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);cid=d['case_id'];case=cases[cid];support={a['position'] for a in d['reference_records']['common_dino_anchors']}
  for arm,p in d['arms'].items():
   v,_=metric._score_student_arm(p,case,targets[cid],support,off,ind);v.update(R_vIoU_at_0p3=float(v['m_vIoU']>=.3),R_vIoU_at_0p5=float(v['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset=case['dataset'],arm=arm,metrics=v))
  for suffix,fit in d['fits'].items():usage.append(dict(case_id=cid,dataset=case['dataset'],profile=suffix,reused_whole_Query=d['whole_Query_Top1_reused'],has_reference=bool(d['r1_anchors']),accepted_steps=fit['optimizer_steps'] if fit else 0,proposed_steps=fit['proposed_optimizer_steps'] if fit else 0))
 summary,paired=subset_aggregate(rows);put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=32,per_domain=16,rows=rows,summary=summary,paired=paired,usage=usage,decisions=read(BASE/'R1_REFERENCE_LOCK.json')['decisions'],GT_bindings=bindings,GT_read=True,development_not_confirmation=True));put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=32,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True))
if __name__=='__main__':globals()[sys.argv[1]]()
