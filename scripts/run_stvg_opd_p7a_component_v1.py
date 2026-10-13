"""Finite actual OPD scope x lifetime experiment; no labels before seal."""
import collections,gc,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p7a_component_common_v1 import *

def bare(f):
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare as original
 return original(f)
def compared(a,b):
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 c=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(a,b,c);return c
def fit_one(base,row,expert,initial,cfg,arm,qualification):
 import torch
 from vg_tta.stvg_opd_p3_scope_precision001 import Installation
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import Geometry
 from vg_tta.stvg_opd_paper_ablations_v1 import fit_variant
 from vg_tta.stvg_opd_p7a_component_state_audit_v1 import adam_readback
 rec=Installation(qualification=qualification);hook=Geometry(rec,'OPD_internal',None)
 try:
  f=fit_variant(base,initial,expert,row['frame_ids'],row['key'],'LN_only' if arm.startswith('LN') else 'on_policy',cfg)
  f['geometry']=dict(rounds=hook.rows,scoped_Adam_recording=True,GT_read=False)
  assert digest(f['initial'])==digest(initial) and not torch.count_nonzero(f['initial'][Q])
  obj=objective(f,expert);adam=adam_readback(f);n,coords=noise(f,row['key'])
  return f,dict(status='pass',objective=obj,Adam=adam,noise=n,sample_coordinates=coords,
   native_head_VJP=rec.last_vjp,GT_read=False)
 finally:hook.close();rec.uninstall();base.restore(base.initial)

def capture(model,r,cache):
 import torch
 from scripts.run_decota_paper_main_v1 import read_row,frames_for,unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.stvg_opd_paper_inputs_v1 import observation
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached
 ip=ROOT/r['input']['input_path'];assert sha(ip)==r['input']['input_sha256'];inp=normalized(load(ip))
 row=read_row(r['dataset'],r['query_ordinal']);tick=time.perf_counter();frames,ids,reuse=frames_for(r['dataset'],row,cache)
 shifted,pixel,spec=observation(r['dataset'],row,'clean',frames)
 batch,records,base=frozen_forward(model,shifted,row)
 native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
 assert pixel==inp['pixel_sha256']==r['pixel_sha256'] and spec==inp['corruption_spec'] is None
 assert ids==inp['frame_ids'] and digest(ids)==r['frame_ids_sha256']
 assert model._fixed_full_state_hash==inp['source_model_state_sha256']==r['source_model_sha256']
 assert native['physical_interval']==inp['interval'] and native['indices']==inp['indices']
 assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and digest(inp['expert'])==r['evidence_sha256']
 iv,check=projection(r['dataset'],base,records)
 temporal=dict(dataset=r['dataset'],parent_id=r['parent_id'],capture=dict(records=records,frame_ids=ids,
  zero=detached({k:base.zero[k] for k in ['boxes','logits','actions']},'cpu'),source_model_state_sha256=model._fixed_full_state_hash),
  frame_ids=ids,pixel_sha256=pixel,Native_interval=native['physical_interval'],Projection_interval=iv,
  projection_check=check,GT_read=False,actual_new_DINO_calls=0)
 cost=dict(frozen_capture_seconds=time.perf_counter()-tick,decoded_frame_reuse=reuse,actual_new_DINO_calls=0,
  actual_frozen_capture_forwards=1)
 ex=unpack_expert(inp['expert']);del frames,shifted,batch
 return base,row,inp,ex,temporal,cost

def run_gpu(qualification):
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 from scripts.run_decota_paper_main_v1 import model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_p7a_component_state_audit_v1 import lifecycle
 from scripts.decota_paper_common_v1 import budget
 activate_old();verify();standby();lease=gpu();d=read(BASE/'DESIGN_LOCK.json')
 if not qualification:assert read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
 phase='qualification' if qualification else 'formal';expected=d['qualification_fits' if qualification else 'formal_fits']
 barrier=BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json')
 if barrier.exists():lease.close();return
 files={};capturefiles={};count=rounds=captures=0;model=None;cache=collections.OrderedDict();started=time.time()
 lookup={(r['dataset'],r['query_ordinal']):r for r in d['records']}
 try:
  for ds,stage in d['stages'].items():
   if model is not None:
    assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
   model=model_for(stage['source']);mh=model._fixed_full_state_hash;source=load(ROOT/stage['source_initial_path'])
   for order,seq in stage['orders'].items():
    for repeat in ([0,1] if qualification else [0]):
     previous={a:None for a in ARMS};prevhash={a:None for a in ARMS}
     for at,q in enumerate(seq[:2] if qualification else seq):
      budget();standby();r=lookup[ds,q];base,row,inp,ex,temp,shared=capture(model,r,cache);captures+=1
      assert digest(base.initial)==digest(source)
      cp=BASE/phase/'captures'/ds/order/f'{at:05}_repeat{repeat}.pt'
      if cp.exists():
       oldtemp=load(cp);compared(temp,oldtemp)
      else:save(cp,temp)
      capturefiles[str(cp.relative_to(ROOT))]=sha(cp)
      if qualification and repeat:compared(temp,load(cp.with_name(f'{at:05}_repeat0.pt')))
      if not qualification and at<2:
       compared(temp,load(BASE/'qualification/captures'/ds/order/f'{at:05}_repeat0.pt'))
      for arm in ARMS:
       cfg=stage['configs'][arm];alpha=config(ds,'joint_carry')['writeback'];initial=incoming(source,previous[arm],arm)
       p=BASE/phase/ds/order/arm/(f'{at:05}_repeat{repeat}.pt' if qualification else f'{at:05}.pt')
       if p.exists():
        rc=read(p.with_suffix('.json'));assert sha(p)==rc['sha256'] and not rc['GT_read'];z=load(p)
        assert z['previous_payload_sha256']==prevhash[arm] and z['config']==cfg and digest(z['fit']['initial'])==digest(initial)
       else:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        f,checked=fit_one(base,row,ex,initial,cfg,arm,qualification);torch.cuda.synchronize();fit_seconds=time.perf_counter()-tick
        c=commit(source,initial,f['state'],arm,alpha)
        life=lifecycle(source,initial,f['state'],c,previous[arm],arm.endswith('carry'),cfg['writeback'])
        old_control=None
        if arm=='joint_carry':old_control=compared(bare(f),bare(load(ROOT/r['originals'][order]['path'])['fit']))
        z=dict(dataset=ds,source=stage['source'],parent_id=r['parent_id'],query_ordinal=q,order=order,arrival=at,arm=arm,
         config=cfg,fit=f,committed=c,previous_payload_sha256=prevhash[arm],audit=checked,lifecycle=life,
         source_model_sha256=mh,original_input_binding=r['input'],old_joint_carry_complete_bitwise=old_control,
         frame_ids=inp['frame_ids'],Native_interval=temp['Native_interval'],Projection_interval=temp['Projection_interval'],
         temporal_capture_path=str(cp.relative_to(ROOT)),temporal_capture_sha256=sha(cp),pixel_sha256=r['pixel_sha256'],
         runtime_sha256=sha(BASE/'RUNTIME.json'),Query_reset=True,Adam_reset=True,GT_read=False,
         compute=dict(instrumented_fit_seconds=fit_seconds,**shared,capture_owner=arm==ARMS[0],
          CUDA_peak_allocated=torch.cuda.max_memory_allocated(),actual_active_parameters=f['active_parameters']))
        staging=p.parent/('UNACCEPTED_'+p.name);save(staging,z)
        if qualification and repeat:
         first=load(p.with_name(f'{at:05}_repeat0.pt'));z['complete_repeat_bitwise']=compared(bare(f),bare(first['fit']))
         z['Adam_repeat_bitwise']=compared(f['geometry'],first['fit']['geometry'])
        if not qualification and at<2:
         first=load(BASE/'qualification'/ds/order/arm/f'{at:05}_repeat0.pt')
         z['qualified_formal_bitwise']=compared(bare(f),bare(first['fit']))
         z['Adam_qualified_formal_bitwise']=compared(f['geometry'],first['fit']['geometry'])
        save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(status='accepted_complete_OPD_component_fit',sha256=h,bytes=p.stat().st_size,
         dataset=ds,arm=arm,order=order,arrival=at,active_parameters=f['active_parameters'],GT_read=False,time=time.time()))
        staging.unlink();del f,checked
       lifecycle(source,z['fit']['initial'],z['fit']['state'],z['committed'],previous[arm],arm.endswith('carry'),cfg['writeback'])
       previous[arm]=z['committed'];prevhash[arm]=sha(p);files[str(p.relative_to(ROOT))]=sha(p)
       count+=1;rounds+=len(z['fit']['rounds']);del z
       status(BASE/'GPU_STATUS.json',dict(status='running_actual_'+phase,dataset=ds,arm=arm,order=order,done=count,total=expected,
        captures=captures,rounds=rounds,pid=os.getpid(),GT_read=False,time=time.time()))
       print('OPD_COMPONENT_PROGRESS',phase,ds,order,arm,count,expected,flush=True)
      del base,row,inp,ex,temp;gc.collect();torch.cuda.empty_cache()
   assert state_hash(model.state_dict())==mh
  assert count==expected and captures==d['qualification_captures' if qualification else 'formal_captures']
  assert rounds==d['qualification_rounds' if qualification else 'formal_rounds'];verify();standby()
  write(barrier,dict(status='pass' if qualification else 'sealed',files=files,capture_files=capturefiles,
   actual_complete_GPU_fits=count,actual_complete_fits=count,actual_complete_rounds=rounds,actual_frozen_captures=captures,
   original_joint_carry_full_controls=count//4,actual_new_DINO_calls=0,GT_read=False,
   runtime_sha256=sha(BASE/'RUNTIME.json'),elapsed_seconds=time.time()-started,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();lease.close()

def root_readback(qualification):
 import torch
 from vg_tta.stvg_opd_p7a_component_state_audit_v1 import adam_readback,lifecycle
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from scripts.run_decota_paper_main_v1 import unpack_expert,read_row
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();activate_old();verify();standby()
 d=read(BASE/'DESIGN_LOCK.json');phase='qualification' if qualification else 'formal'
 bp=BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json');b=read(bp)
 assert b['actual_complete_GPU_fits']==(64 if qualification else 512)
 lookup={(r['dataset'],r['query_ordinal']):r for r in d['records']};counts=collections.Counter();prev={};prevhash={};compares=collections.Counter()
 for rel,h in b['files'].items():
  p=ROOT/rel;assert sha(p)==h==read(p.with_suffix('.json'))['sha256'];z=load(p);f=z['fit'];r=lookup[z['dataset'],z['query_ordinal']]
  stage=d['stages'][z['dataset']];source=load(ROOT/stage['source_initial_path'])
  repeat=int(p.stem.split('repeat')[1]) if qualification else 0;key=(z['dataset'],z['order'],z['arm'],repeat)
  assert z['previous_payload_sha256']==prevhash.get(key);alpha=stage['configs']['joint_carry']['writeback']
  assert f['config']==z['config']==stage['configs'][z['arm']]
  inp=normalized(load(ROOT/r['input']['input_path']));ex=unpack_expert(inp['expert'])
  assert objective(f,ex)==z['audit']['objective'] and adam_readback(f)==z['audit']['Adam']
  assert lifecycle(source,f['initial'],f['state'],z['committed'],prev.get(key),z['arm'].endswith('carry'),z['config']['writeback'])==z['lifecycle']
  assert noise(f,read_row(z['dataset'],z['query_ordinal'])['key'])==(z['audit']['noise'],z['audit']['sample_coordinates'])
  t=load(ROOT/z['temporal_capture_path']);assert sha(ROOT/z['temporal_capture_path'])==z['temporal_capture_sha256']
  from methods.decota_final_simplified_v1.objectives import project
  from scripts.stvg_opd_p6_existing_common001 import config as tc,check_projection
  c=t['capture'];e=project(c['zero']['logits'],c['zero']['actions'],c['records'],tc(z['dataset']))
  check_projection(c['zero']['logits'],c['zero']['actions'],c['records'],e,tc(z['dataset']))
  iv=[x['interval'] for x in e['offsets']];assert z['Projection_interval']==[min(x[0] for x in iv),max(x[1] for x in iv)]
  assert torch.equal(c['zero']['boxes'],inp['native_boxes']) and z['Native_interval']==inp['interval']
  if z['arm']=='joint_carry':
   old=load(ROOT/r['originals'][z['order']]['path']);cc=compared(bare(f),bare(old['fit']));assert cc==z['old_joint_carry_complete_bitwise'];counts['old_complete_controls']+=1
  if qualification and repeat:
   first=load(p.with_name(p.stem.replace('repeat1','repeat0')+'.pt'))
   compared(bare(f),bare(first['fit']));compared(f['geometry'],first['fit']['geometry']);counts['whole_repeat_pairs']+=1
  if not qualification and z['arrival']<2:
   first=load(BASE/'qualification'/z['dataset']/z['order']/z['arm']/f"{z['arrival']:05}_repeat0.pt")
   compared(bare(f),bare(first['fit']));compared(f['geometry'],first['fit']['geometry']);counts['qualified_formal_pairs']+=1
  counts['fits']+=1;counts['rounds']+=len(f['rounds']);counts['active_state_coordinates']+=f['active_parameters']
  counts['scoped_Adam_scalar_checks']+=z['audit']['Adam']['scalar_checks']
  counts['empty_fits']+=f['empty'];counts['actual_backward_steps']+=f['gradient_calls']
  prev[key]=z['committed'];prevhash[key]=h;assert sha(p)==h
 assert counts['fits']==b['actual_complete_GPU_fits'] and counts['rounds']==b['actual_complete_rounds']
 name='ROOT_QUALIFICATION.json' if qualification else 'ROOT_DEPLOYMENT_READBACK.json'
 receipt=dict(status='pass',scope='all actual OPD fits/complete repeats/scoped optimizer/absolute source/reset/own carry/noise/native projection/old full controls',
  counts=dict(counts),barrier_sha256=sha(bp),CPU_only=True,GT_read=False,independent_full_decoder_Jacobian=False,time=time.time())
 write(BASE/name,receipt);print(name,dict(counts),flush=True)

def controller():
 verify();standby();env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
 for mode in ['qualification','qualification_root','formal','deployment_root','score']:
  script=Path(__file__).name if mode!='score' else 'score_stvg_opd_p7a_component_v1.py'
  args=[str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/'+script,mode]
  with (BASE/(mode+'.log')).open('xb') as log:
   w=subprocess.Popen(args,cwd=ROOT,env=env if mode in ['qualification','formal'] else {**env,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
   status(BASE/'STATUS.json',dict(status='running_finite_'+mode,stage=mode,controller_pid=os.getpid(),worker_pid=w.pid,
    GT_read=mode=='score',time=time.time()));code=w.wait();assert code==0,(mode,code)
  write(BASE/'worker_exit'/(mode+'.json'),dict(status='pass',stage=mode,pid=w.pid,exit_code=code,time=time.time()))
 status(BASE/'STATUS.json',dict(status='pending_actual_independent_scalar_root_PDF_public_archive',done=512,total=512,GT_read=True,time=time.time()))

if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='prepare':prepare()
  elif mode=='controller':controller()
  elif mode in ['qualification_root','deployment_root']:root_readback(mode=='qualification_root')
  else:assert mode in ['qualification','formal'];run_gpu(mode=='qualification')
 except BaseException:
  p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',stage=sys.argv[1] if len(sys.argv)>1 else 'unknown',evidence=str(p.relative_to(ROOT)),time=time.time()));raise
