"""Root-authorized R51 real episodic fit / installed-full-model closure.

UNEXECUTED source draft. These are separate --stage fit|closure commands with
separate root plans/grants. No task GT, external model, reference refresh or
box fitting is imported. All G/identity/groups predate any update.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
from model51 import ROOT,SOURCE_SHA,PREFIX_KEYS,load_source,fingerprint,array_sha,native_matching,SENTINEL
from structure51 import validate_structure,readout51

MANIFEST='research/round24/input/ALL336_PRIVATE.json'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'
INTERFACES=('mlp','lora_qv')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save(path,arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(path)
def npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def exact(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def bound(folder,seal,rel):
 assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 p=folder/rel;assert sha(p)==seal['files'][rel],str(p)
 return p
def scalar_readout(value):
 return {k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
def seal(out,state,**extra):
 dump(out/'SEALED.json',{'state':state,**extra,'files':{p.relative_to(out).as_posix():sha(p)
  for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SEALED.json'}})


def context(args):
 pp=Path(args.plan).resolve();plan=read(pp);grant=read(args.authorization)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_DEV336_ACTUAL_'+args.stage.upper()+'_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['stage']==args.stage and plan['n']==336 and plan['manifest']==MANIFEST and plan['source_plan']==SOURCE_PLAN
 assert plan['source_checkpoint_sha256']==SOURCE_SHA and plan['interfaces']==list(INTERFACES)
 assert plan['steps']==[0,1,3] and plan['ranknet_tau']==.1 and plan['keep_beta']==1. and plan['lr']==.0005
 assert plan['new_G_or_reference_calls']==0 and plan['GT_read'] is False and plan['original_A_boxes_only'] is True
 assert plan['direct_G_gate_PASS'] is True
 required={MANIFEST,SOURCE_PLAN,plan['direct_G_gate_receipt']}
 for key in ('capture_A_root','capture_B_root','structure_root'):required.add(plan[key]+'/SEALED.json')
 if args.stage=='closure':required.add(plan['fit_root']+'/SEALED.json')
 assert required<=set(plan['inputs_sha256'])
 for group in ('inputs_sha256','programs_sha256'):
  for rel,h in plan[group].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and 'FINAL' not in rel
   assert 'round46/final' not in rel and sha(ROOT/rel)==h,rel
 needed={'actualrun51.py','structure51.py','model51.py','lora51.py','tta51.py','instances51.py','captureB51.py'}
 assert {'crossplatform/round51/'+x for x in needed}<=set(plan['programs_sha256'])
 assert read(ROOT/plan['direct_G_gate_receipt'])['gate_PASS'] is True
 assert 0<=plan['probability_max_abs_bound']<=2e-5 and 0<=plan['logodds_max_abs_bound']<=2e-4
 assert plan['LoRA_initialization_seed']==0
 roots={};seals={}
 for key in ('capture_A_root','capture_B_root','structure_root'):
  folder=(ROOT/plan[key]).resolve();assert folder.is_relative_to(ROOT/'crossplatform/round51')
  roots[key]=folder;seals[key]=read(folder/'SEALED.json');assert seals[key]['n']==336
 assert seals['capture_A_root']['state']=='R51_DEV336_SINGLE_WAYMO_MODEL_SUBJECT_REFERENCE_HYPOTHESES_NO_GT'
 assert seals['capture_B_root']['state']=='R51_CONDITIONAL_DEV336_B_ONLY_SOURCE_PREFIX_CAPTURE_NO_GT'
 assert seals['structure_root']['state']=='R51_DEV336_FIXED_POINT_CARRIER_STRUCTURES_BEFORE_TTA_NO_GT'
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 if args.stage=='fit':assert plan['max_optimizer_steps']==2016 and plan['max_full_grounder_forwards']==0
 else:
  assert plan['max_optimizer_steps']==0 and plan['max_full_grounder_forwards']==16
  assert plan['selection_rule']=='HASH_MIN_ACCEPTED_NONZERO_S1_S3_PER_INTERFACE_PLATFORM'
 out=(ROOT/plan['output_root']).resolve();assert out.is_relative_to(ROOT/'crossplatform/round51')
 assert out not in roots.values()
 binding={'stage':args.stage,'plan_sha256':sha(pp),'source_checkpoint_sha256':SOURCE_SHA,
  'input_seals':{k:sha(v/'SEALED.json') for k,v in roots.items()}}
 if out.exists():assert args.resume and not (out/'SEALED.json').exists() and read(out/'RUN_BINDINGS.json')==binding
 else:out.mkdir(parents=True);dump(out/'RUN_BINDINGS.json',binding)
 return plan,rows,out,roots,seals,binding


def load_episode(row,roots,seals):
 sid=row['id'];aroot=roots['capture_A_root'];broot=roots['capture_B_root'];sroot=roots['structure_root']
 acase=read(bound(aroot,seals['capture_A_root'],'cases_PRIVATE/'+sid+'.json'))
 bcase=read(bound(broot,seals['capture_B_root'],'cases_PRIVATE/'+sid+'.json'))
 apath=bound(aroot,seals['capture_A_root'],acase['subject']['file'])
 bpath=bound(broot,seals['capture_B_root'],bcase['B_file'])
 assert sha(apath)==acase['subject']['sha256']==bcase['A_source_file_sha256']
 assert sha(bpath)==bcase['B_file_sha256'] and acase['A_seed']==row['seed']
 caches={'A':npz(apath),'B':npz(bpath)}
 info=read(bound(sroot,seals['structure_root'],'cases_PRIVATE/'+sid+'.json'))
 arrays=npz(bound(sroot,seals['structure_root'],'arrays_PRIVATE/'+sid+'.npz'))
 assert info['input_bindings']['A_file_sha256']==sha(apath) and info['input_bindings']['B_file_sha256']==sha(bpath)
 assert info['input_bindings']['B_capture_seal_sha256']==sha(broot/'SEALED.json')
 for view in ('A','B'):
  c=caches[view]
  assert c['boxes'].shape==(256,6) and c['h'].shape==(256,288) and c['scores'].shape==c['logodds'].shape==(256,)
  assert c['prefix_query_mask_is_none'].item() and all(k in c for k in PREFIX_KEYS if k!='prefix_query_mask')
 exact(caches['A']['token_mask'],caches['B']['token_mask'],'same_full_text_mask_A_B')
 validate_structure(info,arrays,caches)
 metadata={'A':{'seed':row['seed'],'point_sample_sha256':acase['point_sample_sha256'],
  'point_indices_sha256':acase['point_indices_sha256']},
  'B':{'seed':bcase['B_seed'],'point_sample_sha256':bcase['point_sample_sha256'],
  'point_indices_sha256':bcase['point_indices_sha256']}}
 return caches,info,arrays,metadata


def fit(plan,rows,out,roots,seals,binding):
 import torch
 from tta51 import episodic
 model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model);records=[];new=0
 for row in rows:
  caches,info,arrays,_=load_episode(row,roots,seals);sid=row['id']
  for interface in INTERFACES:
   tag=sid+'__'+interface;cp=out/'calls_PRIVATE'/(tag+'.json');target=out/'episodes_PRIVATE'/(tag+'.npz')
   casepath=out/'cases_PRIVATE'/(tag+'.json')
   cb={'plan_sha256':binding['plan_sha256'],'id_PRIVATE':sid,'interface':interface,
    'structure_case_sha256':sha(roots['structure_root']/'cases_PRIVATE'/(sid+'.json'))}
   if cp.exists():
    record=read(cp);assert record['state']=='COMPLETED' and record['binding']==cb,'Incomplete episode is preserved, never automatically refit'
    assert sha(target)==record['output_sha256'] and sha(casepath)==record['case_sha256'];receipt=read(casepath)
   else:
    record={'state':'STARTED','binding':cb,'reserved_optimizer_steps':3 if info['accepted'] else 0,
     'actual_optimizer_steps':None,'failure_cost_policy':'RESERVE_ALL_THREE_NO_AUTOMATIC_RETRY'};dump(cp,record);tick=time.perf_counter()
    try:
     # A rejected pair can contain coincident diagnostic groups. It supplies NO
     # training labels: empty pairs also leave all candidates in keep scope.
     pairs=info['pairs'] if info['accepted'] else {v:None for v in ('A','B')}
     fitted=episodic(model,caches,pairs,interface=interface,accepted=info['accepted'],lr=plan['lr'],
      initial_probability_bound=plan['probability_max_abs_bound'],initial_logodds_bound=plan['logodds_max_abs_bound'],seed=0)
     payload={};endpoints={};nonzero={}
     base=readout51({v:{'probability':caches[v]['scores'],'logodds':caches[v]['logodds']} for v in ('A','B')},info,arrays)
     for step in (0,1,3):
      snap=fitted['snapshots'][step];endpoint=readout51(snap['views'],info,arrays);endpoints[str(step)]=scalar_readout(endpoint)
      if step==0:assert endpoint['canonical_A_query']==base['canonical_A_query'],'STOP_source_replay_consensus_endpoint_changed'
      for view in ('A','B'):
       for key in ('probability','logodds'):payload[f's{step}_{view}_{key}']=snap['views'][view][key]
      payload[f's{step}_scores']=endpoint['scores'];payload[f's{step}_consensus_probability']=endpoint['consensus_probability']
      payload[f's{step}_canonical_A_query']=np.array(endpoint['canonical_A_query'],np.int64)
      payload[f's{step}_selected_original_A_box']=caches['A']['boxes'][endpoint['canonical_A_query']].copy()
      nonzero[str(step)]=any(x['different_source_elements']>0 for x in snap['update_report']['parameters'])
      if info['accepted'] and step in (1,3):
       for name,value in snap['state'].items():payload[f'state_s{step}__{name}']=value
     payload['source_A_query']=np.array(info['source_A_query'],np.int64)
     payload['source_selected_original_A_box']=caches['A']['boxes'][info['source_A_query']].copy()
     payload['literal_G_A_query']=np.array(info['literal_G_A_query'],np.int64)
     payload['literal_G_scores']=arrays['literal_G_scores_A'].copy()
     payload['literal_G_selected_original_A_box']=caches['A']['boxes'][info['literal_G_A_query']].copy()
     save(target,payload);record.update(state='RAW_FIT_SAVED',output_sha256=sha(target),actual_optimizer_steps=fitted['optimizer_steps']);dump(cp,record)
     receipt={'id_PRIVATE':sid,'platform':row['platform'],'interface':interface,'accepted':info['accepted'],
      'nonzero_parameter_updates':nonzero,'optimizer_steps':fitted['optimizer_steps'],'trace':fitted['trace'],
      'initial_bridges':fitted['initial_bridges'],'endpoints':endpoints,'source_A_query':info['source_A_query'],
      'literal_G_A_query':info['literal_G_A_query'],'rejection_reasons':info['rejection_reasons'],
      'point_identity_is_hypothesis_not_GT':True,'new_G_or_reference_calls':0,'new_full_grounder_forwards':0,
      'seconds':time.perf_counter()-tick,'GT_read':False,
      'update_reports':{str(s):fitted['snapshots'][s]['update_report'] for s in (0,1,3)}}
     dump(casepath,receipt);record.update(state='COMPLETED',case_sha256=sha(casepath));dump(cp,record);new+=1
    except BaseException as error:
     record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));dump(cp,record);raise
   records.append(receipt)
  if len(records)%32==0:print('R51_FIT_EPISODES',len(records),flush=True)
 assert len(records)==672 and fingerprint(model)==sourcefp
 counts={'n':336,'interface_episodes':672,'optimizer_steps':sum(r['optimizer_steps'] for r in records),
  'new_full_grounder_forwards':0,'new_G_or_reference_calls':0,'GT_read':False,'source_restored':True,
  'source_fingerprint':sourcefp,'new_episodes_this_process':new,'peak_bytes':int(torch.cuda.max_memory_allocated())}
 assert counts['optimizer_steps']<=plan['max_optimizer_steps']
 dump(out/'RECEIPTS_PRIVATE.json',records);dump(out/'COUNTS.json',counts)
 seal(out,'R51_DEV336_REAL_EPISODIC_FITS_SAME_INFORMATION_NO_GT',**counts,plan_sha256=binding['plan_sha256'])


def installed_forward(model,cache,points,indices,caption,interface,on_enter):
 """A genuine complete model forward with the adapted controller installed.

 No source_capture call: that API deliberately forbids adapted parameters.
 Immutable pre-decoder5 tensors, original query anchors and point IDs must be
 byte-exact. New LoRA last-head boxes are returned only as discarded diagnostics.
 """
 import torch
 device=next(model.parameters()).device;captured={}
 assert all(not m.training for m in model.modules())
 def before(_module,args,kwargs):
  assert len(args)==6 and kwargs.get('detected_feats') is None and kwargs.get('detected_mask') is None
  for key,value in zip(PREFIX_KEYS,args):captured[key]=None if value is None else value.detach().cpu().numpy().copy()
 hook=model.decoder[5].register_forward_pre_hook(before,with_kwargs=True)
 try:
  tensor=torch.as_tensor(np.ascontiguousarray(points,np.float32),device=device)[None];on_enter()
  with torch.no_grad():o=model({'point_clouds':tensor,'text':[caption+SENTINEL]})
 finally:hook.remove()
 mask=torch.as_tensor(cache['token_mask'],device=device,dtype=torch.bool)
 p,z=native_matching(o['last_proj_queries'],o['proj_tokens'],mask)
 actual_box=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].detach().cpu().numpy()
 result={'probability':p.detach().cpu().numpy(),'logodds':z.detach().cpu().numpy(),
  'discarded_new_last_head_geometry':actual_box,
  'full_query_anchor_xyz':o['query_points_xyz'][0].detach().cpu().numpy(),
  'full_query_points_sample_inds':o['query_points_sample_inds'][0].detach().cpu().numpy(),
  'full_raw_point_indices':np.asarray(indices,np.int64).copy(),
  'full_proj_tokens':o['proj_tokens'][0].detach().cpu().numpy(),
  'full_text_attention_mask':o['text_attention_mask'][0].detach().cpu().numpy(),
  'full_prefix_query_mask_is_none':np.array(captured['prefix_query_mask'] is None)}
 for key,value in captured.items():
  if value is not None:result['full_'+key]=value
 return result


def verify_installed(cache,actual,interface):
 """Check after raw output persistence, so even a failed bridge has evidence."""
 assert actual['full_prefix_query_mask_is_none'].item()
 for key in PREFIX_KEYS:
  if key!='prefix_query_mask':exact(actual['full_'+key],cache[key],'frozen_final_decoder_prefix_'+key)
 exact(actual['full_query_anchor_xyz'],cache['query_anchor_xyz'],'query_anchors_fixed')
 exact(actual['full_query_points_sample_inds'],cache['query_points_sample_inds'],'query_order_fixed')
 exact(actual['full_raw_point_indices'],cache['point_indices'],'raw_sample_indices_fixed')
 exact(actual['full_proj_tokens'],cache['tokens'],'language_tokens_frozen')
 assert not actual['full_text_attention_mask'].any()
 if interface=='mlp':exact(actual['discarded_new_last_head_geometry'],cache['boxes'],'MLP_full_model_original_all256_boxes_exact')


def closure(plan,rows,out,roots,seals,binding):
 import torch
 from lora51 import MatchingMLP,QVLoRA
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
 fitroot=(ROOT/plan['fit_root']).resolve();assert fitroot.is_relative_to(ROOT/'crossplatform/round51')
 fs=read(fitroot/'SEALED.json');assert fs['state']=='R51_DEV336_REAL_EPISODIC_FITS_SAME_INFORMATION_NO_GT'
 records=read(bound(fitroot,fs,'RECEIPTS_PRIVATE.json'));rowindex={r['id']:r for r in rows}
 chosen=[];missing=[]
 for interface in INTERFACES:
  for platform in ('drone','quad'):
   eligible=[r for r in records if r['interface']==interface and r['platform']==platform and r['accepted']
    and all(r['nonzero_parameter_updates'][str(s)] for s in (1,3))]
   if eligible:chosen.append(min(eligible,key=lambda r:hashlib.sha256((plan['selection_salt']+'|'+interface+'|'+r['id_PRIVATE']).encode()).hexdigest()))
   else:missing.append({'interface':interface,'platform':platform,'state':'NO_NONZERO_ACCEPTED_STATES_NOT_TTA_CLOSURE'})
 selected={'selected':[{'id_PRIVATE':r['id_PRIVATE'],'interface':r['interface'],'platform':r['platform']} for r in chosen],
  'missing':missing,'selection_uses_GT':False,'rule':plan['selection_rule']}
 selectionfile=out/'SELECTED_BEFORE_FORWARD_PRIVATE.json'
 if selectionfile.exists():assert read(selectionfile)==selected
 else:dump(selectionfile,selected)
 if not chosen:
  seal(out,'R51_INSTALLED_CLOSURE_NOT_APPLICABLE_VERIFIED_NO_NONZERO_STATES',n=0,new_full_grounder_forwards=0,missing=missing,GT_read=False);return
 model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model)
 entered=sum(read(p).get('model_forward_entered',False) for p in (out/'calls_PRIVATE').glob('*.json')) if (out/'calls_PRIVATE').exists() else 0
 checks=[]
 for item in chosen:
  sid=item['id_PRIVATE'];row=rowindex[sid];interface=item['interface'];tag=sid+'__'+interface
  caches,info,arrays,metadata=load_episode(row,roots,seals)
  fitted=npz(bound(fitroot,fs,'episodes_PRIVATE/'+tag+'.npz'))
  controller=MatchingMLP(model) if interface=='mlp' else QVLoRA(model,rank=4,alpha=4,seed=0)
  try:
   for step in (1,3):
    controller.reset();module=controller.module if interface=='mlp' else controller.adapter
    parameters=dict(module.named_parameters());prefix=f'state_s{step}__'
    state={k[len(prefix):]:v for k,v in fitted.items() if k.startswith(prefix)}
    assert set(state)==set(parameters)
    with torch.no_grad():
     for name,p in parameters.items():p.copy_(torch.as_tensor(state[name],device=p.device,dtype=p.dtype))
    assert any(x['different_source_elements']>0 for x in controller.report()['parameters'])
    viewoutputs={}
    for view in ('A','B'):
     cache=caches[view];meta=metadata[view];viewrow=dict(row,seed=meta['seed'])
     points,indices,_=sample_points(viewrow)
     assert array_sha(points)==meta['point_sample_sha256'] and array_sha(indices)==meta['point_indices_sha256']
     callid=tag+f'__s{step}__'+view;cp=out/'calls_PRIVATE'/(callid+'.json');target=out/'outputs_PRIVATE'/(callid+'.npz')
     cb={'plan_sha256':binding['plan_sha256'],'id_PRIVATE':sid,'interface':interface,'step':step,'view':view,
      'fit_episode_sha256':sha(fitroot/'episodes_PRIVATE'/(tag+'.npz')),'point_sample_sha256':array_sha(points)}
     if cp.exists():
      record=read(cp);assert record['state']=='COMPLETED' and record['binding']==cb,'No incomplete full-model retry'
      assert sha(target)==record['raw_output_sha256'];actual=npz(target)
     else:
      record={'state':'STARTED','binding':cb,'model_forward_entered':False};dump(cp,record);tick=time.perf_counter()
      def on_enter():
       nonlocal entered
       assert entered<plan['max_full_grounder_forwards'];entered+=1;record['model_forward_entered']=True;dump(cp,record)
      try:
       actual=installed_forward(model,cache,points,indices,row['caption'],interface,on_enter)
       save(target,{k:v for k,v in actual.items() if isinstance(v,np.ndarray)})
       record.update(state='RAW_FULL_MODEL_OUTPUT_SAVED',raw_output_sha256=sha(target));dump(cp,record)
       verify_installed(cache,actual,interface)
       pe=float(np.max(np.abs(actual['probability']-fitted[f's{step}_{view}_probability'])))
       ze=float(np.max(np.abs(actual['logodds']-fitted[f's{step}_{view}_logodds'])))
       assert pe<=plan['probability_max_abs_bound'] and ze<=plan['logodds_max_abs_bound']
       assert int(actual['probability'].argmax())==int(fitted[f's{step}_{view}_probability'].argmax())
       record.update(state='COMPLETED',prefix_anchor_order_byteexact=True,probability_max_abs_error=pe,
        logodds_max_abs_error=ze,native_argmax_exact=True,new_geometry_is_discarded=True,seconds=time.perf_counter()-tick)
       dump(cp,record)
      except BaseException as error:
       record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));dump(cp,record);raise
     viewoutputs[view]=actual;checks.append(record)
    endpoint=readout51(viewoutputs,info,arrays)
    assert endpoint['canonical_A_query']==int(fitted[f's{step}_canonical_A_query']),'STOP_installed_final_readout_differs'
    exact(caches['A']['boxes'][endpoint['canonical_A_query']],fitted[f's{step}_selected_original_A_box'],'canonical_A_selected_original_box_exact')
    dump(out/'endpoints_PRIVATE'/(tag+f'__s{step}.json'),{**scalar_readout(endpoint),'final_canonical_A_query_and_box_exact':True,
     'final_scores_max_abs_error':float(np.max(abs(endpoint['scores']-fitted[f's{step}_scores']))),'step':step,'interface':interface})
  finally:
   controller.reset()
   if interface=='lora_qv':controller.remove()
   else:model.requires_grad_(False)
  assert fingerprint(model)==sourcefp
 assert entered==4*len(chosen) and entered<=16
 result={'state':'PASS_R51_ACTUAL_INSTALLED_FULL_MODEL_CLOSURE','n':len(chosen),'missing':missing,
  'complete_required_interface_platform_coverage':not missing,
  'new_full_grounder_forwards':entered,'checks':checks,'source_restored':True,'source_fingerprint':sourcefp,
  'source_prefix_query_anchors_order_byteexact':True,'final_only_original_A_boxes':True,
  'optimizer_steps':0,'new_G_or_reference_calls':0,'GT_read':False,'peak_bytes':int(torch.cuda.max_memory_allocated())}
 dump(out/'RESULT_PRIVATE.json',result);seal(out,'R51_ACTUAL_INSTALLED_FULL_MODEL_CLOSURE_COMPLETE_NO_GT',**{k:v for k,v in result.items() if k not in ('state','checks')},plan_sha256=binding['plan_sha256'])


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--stage',choices=('fit','closure'),required=True)
 parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True);parser.add_argument('--resume',action='store_true')
 args=parser.parse_args();plan,rows,out,roots,seals,binding=context(args)
 import torch
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 try:(fit if args.stage=='fit' else closure)(plan,rows,out,roots,seals,binding)
 except BaseException as error:
  dump(out/f'FAILED_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),'automatic_retry':False,'GT_read':False});raise


if __name__=='__main__':main()
