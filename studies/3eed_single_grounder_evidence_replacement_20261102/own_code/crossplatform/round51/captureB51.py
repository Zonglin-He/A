"""UNEXECUTED conditional DEV336 B-only source capture, no references/TTA/GT.

Reuse existing R51 subject-A prefix, resample same physical coordinates with
the locked original R50 B seed rule, and perform at most 336 new source calls.
The root must separately authorize this only after the direct-G gate.
"""
from pathlib import Path
import argparse,hashlib,json,random,sys,time
import numpy as np
from model51 import ROOT,load_source,source_capture,fingerprint,array_sha,SOURCE_SHA

MANIFEST='research/round24/input/ALL336_PRIVATE.json'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'
B_SALT='R50episodic_B_v1'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(path,value):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as stream:np.savez_compressed(stream,**value)
 tmp.replace(path)


def bseed(row):
 return int(hashlib.sha256((B_SALT+'|'+row['id']+'|'+str(row['seed'])).encode()).hexdigest()[:16],16)


def context(args):
 pp=Path(args.plan).resolve();grant=read(args.authorization);plan=read(pp)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_CONDITIONAL_DEV336_B_ONLY_SOURCE_CAPTURE_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['n']==336 and plan['manifest']==MANIFEST and plan['source_plan']==SOURCE_PLAN
 assert plan['source_checkpoint_sha256']==SOURCE_SHA and plan['max_grounder_forwards']==336
 assert plan['new_A_forwards']==0 and plan['new_reference_forwards']==0
 assert plan['B_sampling_salt']==B_SALT and plan['no_GT_newG_external_models_or_TTA'] is True
 assert plan['direct_G_gate_verified_before_B_capture'] is True
 required={MANIFEST,SOURCE_PLAN,plan['capture_A_root']+'/SEALED.json',plan['direct_G_gate_receipt']}
 assert required<=set(plan['inputs_sha256'])
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   assert 'FINAL' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h,rel
 assert all(r in plan['programs_sha256'] for r in ('crossplatform/round51/captureB51.py','crossplatform/round51/model51.py'))
 gate=read(ROOT/plan['direct_G_gate_receipt']);assert gate['gate_PASS'] is True
 aroot=(ROOT/plan['capture_A_root']).resolve();assert aroot.is_relative_to(ROOT/'crossplatform/round51')
 aseal=read(aroot/'SEALED.json')
 assert aseal['state']=='R51_DEV336_SINGLE_WAYMO_MODEL_SUBJECT_REFERENCE_HYPOTHESES_NO_GT' and aseal['n']==336
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 out=(ROOT/plan['output_root']).resolve();assert out.is_relative_to(ROOT/'crossplatform/round51') and not out.is_relative_to(aroot)
 binding={'plan_sha256':sha(pp),'A_capture_seal_sha256':sha(aroot/'SEALED.json'),'manifest_sha256':sha(ROOT/MANIFEST),
  'B_sampling_salt':B_SALT,'registered_B_calls':336,'source_checkpoint_sha256':SOURCE_SHA}
 if out.exists():assert args.resume and not (out/'SEALED.json').exists() and read(out/'RUN_BINDINGS.json')==binding
 else:out.mkdir(parents=True);dump(out/'RUN_BINDINGS.json',binding)
 for name in ('calls_PRIVATE','samples_PRIVATE','cases_PRIVATE'):(out/name).mkdir(exist_ok=True)
 return plan,rows,out,aroot,aseal,binding


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True);p.add_argument('--resume',action='store_true')
 args=p.parse_args();plan,rows,out,aroot,aseal,binding=context(args);start=time.perf_counter()
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
 import torch
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 try:model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model)
 except BaseException as error:
  dump(out/'LOAD_FAILURE.json',{'error_type':type(error).__name__,'error':str(error),'grounder_forwards':0});raise
 versions=[(n,p._version) for n,p in model.named_parameters()];load_seconds=time.perf_counter()-start
 entered=sum(read(path).get('model_forward_entered',False) for path in (out/'calls_PRIVATE').glob('*.json'))
 completed=new=reused=0
 try:
  for row in rows:
   sid=row['id'];acasepath=aroot/'cases_PRIVATE'/(sid+'.json')
   assert sha(acasepath)==aseal['files'][acasepath.relative_to(aroot).as_posix()]
   acase=read(acasepath);apath=aroot/acase['subject']['file']
   assert sha(apath)==acase['subject']['sha256']==aseal['files'][apath.relative_to(aroot).as_posix()]
   assert acase['A_seed']==row['seed']
   b_row=dict(row,seed=bseed(row));points,indices,nraw=sample_points(b_row)
   assert nraw==acase['raw_point_count']
   assert array_sha(points)!=acase['point_sample_sha256'] and array_sha(indices)!=acase['point_indices_sha256']
   callbinding={'id_PRIVATE':sid,'plan_sha256':binding['plan_sha256'],'A_file_sha256':sha(apath),
    'A_seed':row['seed'],'B_seed':b_row['seed'],'B_sampling_salt':B_SALT,
    'point_sample_sha256':array_sha(points),'point_indices_sha256':array_sha(indices),
    'actual_model_text_sha256':hashlib.sha256((row['caption']+' . not mentioned').encode()).hexdigest()}
   cp=out/'calls_PRIVATE'/(sid+'.json');target=out/'samples_PRIVATE'/(sid+'.npz')
   if cp.exists():
    record=read(cp);assert record['binding']==callbinding and record['state']=='COMPLETED', 'No incomplete-call retry'
    assert sha(target)==record['raw_output_sha256'];reused+=1
   else:
    record={'state':'STARTED','model_forward_entered':False,'binding':callbinding};dump(cp,record)
    tick=time.perf_counter()
    def entered_forward():
     nonlocal entered,new
     assert entered<336;entered+=1;new+=1;record['model_forward_entered']=True;dump(cp,record)
    try:
     arrays,receipt=source_capture(model,points,indices,row['caption'],retain_prefix=True,on_forward_enter=entered_forward)
     save(target,arrays);record.update(state='RAW_MODEL_OUTPUT_SAVED',raw_output_sha256=sha(target),capture_receipt=receipt);dump(cp,record)
     with np.load(apath,allow_pickle=False) as aa:
      assert arrays['token_mask'].dtype==aa['token_mask'].dtype and arrays['token_mask'].tobytes()==aa['token_mask'].tobytes()
     record.update(state='COMPLETED',same_full_text_and_token_mask_as_A=True,seconds=time.perf_counter()-tick);dump(cp,record)
    except BaseException as error:
     record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));dump(cp,record);raise
   payload={'id_PRIVATE':sid,'platform':row['platform'],'A_source_file_PRIVATE':str(apath.relative_to(ROOT)),
    'A_source_file_sha256':sha(apath),'B_file':target.relative_to(out).as_posix(),'B_file_sha256':sha(target),
    'A_seed':row['seed'],'B_seed':b_row['seed'],'raw_point_count':int(nraw),
    'point_sample_sha256':array_sha(points),'point_indices_sha256':array_sha(indices),
    'same_current_input_not_unseen_view':True,'same_coordinates_no_rotation_translation_or_flip':True,
    'A_B_query_slots_require_geometric_correspondence':True,'new_A_forwards':0,'new_reference_forwards':0,'GT_read':False}
   path=out/'cases_PRIVATE'/(sid+'.json')
   if path.exists():assert read(path)==payload
   else:dump(path,payload)
   completed+=1
   if completed%16==0:print('CAPTURE_B51',completed,new,reused,flush=True)
  assert entered==completed==336
  assert versions==[(n,p._version) for n,p in model.named_parameters()] and fingerprint(model)==sourcefp
  counts={'n':336,'B_forwards_all_processes':entered,'B_forwards_this_process':new,'resumed_completed':reused,
   'new_A_forwards':0,'new_reference_forwards':0,'new_external_visual_calls':0,'optimizer_steps':0,'GT_read':False,
   'B_sampling_salt':B_SALT,'all_B_last_decoder_prefixes_saved':True,'source_unchanged':True,
   'source_fingerprint':sourcefp,'load_seconds':load_seconds,'seconds_this_process':time.perf_counter()-start,
   'peak_bytes':int(torch.cuda.max_memory_allocated())}
  dump(out/'COUNTS.json',counts)
  dump(out/'SEALED.json',{'state':'R51_CONDITIONAL_DEV336_B_ONLY_SOURCE_PREFIX_CAPTURE_NO_GT',**counts,
   'plan_sha256':binding['plan_sha256'],'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SEALED.json'}})
  print('CAPTURE_B51_SEALED',counts,flush=True)
 except BaseException as error:
  dump(out/f'FAILED_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),
   'completed':completed,'entered':entered,'new_forwards':new,'GT_read':False,'automatic_retry':False});raise


if __name__=='__main__':main()
