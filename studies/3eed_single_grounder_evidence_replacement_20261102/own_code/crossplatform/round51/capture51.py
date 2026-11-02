"""R51 DEV336 same-model source/reference capture; root GPU execution only.

No Mask/Qwen/DINO/old ownership/media/GT loader. One source model is loaded
once. A uses original sampled XYZ and full caption; each reference own_query
uses those same points, but its own 256 hypotheses. No B capture or TTA here.
Persistent STARTED/raw/COMPLETED calls block automatic retry of failures.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys,time,random
import numpy as np
from model51 import load_source,source_capture,fingerprint,array_sha,bridge_capture,SOURCE_SHA

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round24/input/ALL336_PRIVATE.json'
CANONICAL='crossplatform/round24/capture336_attempt001'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_arrays(path,arrays):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
 tmp.replace(path)


def context(args):
 pp=Path(args.plan).resolve();grant=read(args.authorization)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_DEV336_SAME_MODEL_SUBJECT_REFERENCE_CAPTURE'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 plan=read(pp)
 assert plan['n']==336 and plan['manifest']==MANIFEST and plan['canonical_source_root']==CANONICAL
 assert plan['source_plan']==SOURCE_PLAN and plan['source_checkpoint_sha256']==SOURCE_SHA
 assert plan['no_GT_external_visual_models_or_external_evidence'] is True
 assert plan['capture_B'] is False and plan['subject_uses_original_full_caption'] is True
 required={MANIFEST,SOURCE_PLAN,CANONICAL+'/SEALED.json',plan['graph_index'],plan['pure_text_root']+'/SEALED.json'}
 assert required<=set(plan['inputs_sha256'])
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   assert 'FINAL' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h,rel
 for rel in ('crossplatform/round51/capture51.py','crossplatform/round51/model51.py'):
  assert rel in plan['programs_sha256']
 textroot=Path(plan['pure_text_root'])
 assert textroot.as_posix()=='crossplatform/round50/full5912_attempt001/text', 'Only already-sealed pure Stanza graphs'
 textseal=read(ROOT/textroot/'SEALED.json')
 assert textseal['state']=='R50_FULL5912_LITERAL_TEXT_COMPLETE' and textseal['n']==5912
 rows=read(ROOT/MANIFEST);index=read(ROOT/plan['graph_index'])
 assert len(rows)==336 and len({r['id'] for r in rows})==336 and set(index)=={r['id'] for r in rows}
 graphs={};calls=0
 for row in rows:
  sid=row['id'];assert re.fullmatch('[A-Za-z0-9_-]+',sid)
  route=index[sid];relative=Path(route['path'])
  assert not relative.is_absolute() and '..' not in relative.parts
  gp=(ROOT/relative).resolve();assert gp.is_relative_to((ROOT/textroot).resolve())
  assert route['caption_sha256']==hashlib.sha256(row['caption'].encode()).hexdigest()
  assert sha(gp)==route['sha256']==textseal['files'][gp.relative_to(ROOT/textroot).as_posix()]
  graph=read(gp);refs=graph['references'];assert len(refs)<=4
  assert len({r['entity'] for r in refs})==len(refs)
  assert all(re.fullmatch('R[0-9]+',r['entity']) and isinstance(r['own_query'],str) and r['own_query'] for r in refs)
  graphs[sid]=graph;calls+=1+len(refs)
 assert calls==plan['registered_logical_grounder_calls']<=plan['max_grounder_forwards']<=1680
 out=(ROOT/plan['output_root']).resolve()
 assert out.is_relative_to(ROOT/'crossplatform/round51') and out!=ROOT/'crossplatform/round51'
 binding={'plan_sha256':sha(pp),'manifest_sha256':sha(ROOT/MANIFEST),
  'source_plan_sha256':sha(ROOT/SOURCE_PLAN),'graph_index_sha256':sha(ROOT/plan['graph_index']),
  'registered_logical_grounder_calls':calls,'source_checkpoint_sha256':SOURCE_SHA}
 if out.exists():assert args.resume and not (out/'SEALED.json').exists() and read(out/'RUN_BINDINGS.json')==binding
 else:out.mkdir(parents=True);dump(out/'RUN_BINDINGS.json',binding)
 for directory in ('calls_PRIVATE','samples_PRIVATE','references_PRIVATE','cases_PRIVATE'):(out/directory).mkdir(exist_ok=True)
 return plan,rows,index,graphs,out,binding


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True)
 parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 plan,rows,index,graphs,out,runbinding=context(args);start=time.perf_counter()
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'))
 from runner_helpers import sample_points
 import torch
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 try:model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model)
 except BaseException as error:
  dump(out/f'LOAD_FAILURE_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),
   'grounder_forwards':0,'GT_read':False,'automatic_retry':False});raise
 versions=[(n,p._version) for n,p in model.named_parameters()]
 load_seconds=time.perf_counter()-start
 projection=out/'SOURCE_MATCHING_MLP_PRIVATE.npz'
 state={n:p.detach().cpu().numpy() for n,p in model.contrastive_align_projection_image.state_dict().items()}
 if projection.exists():
  with np.load(projection,allow_pickle=False) as old:
   assert set(old.files)==set(state) and all(old[k].dtype==state[k].dtype and old[k].tobytes()==state[k].tobytes() for k in state)
 else:save_arrays(projection,state)
 canonicalseal=read(ROOT/CANONICAL/'SEALED.json')
 logical=0;new_calls=0;reused=0;completed=[]
 entered_total=sum(read(p).get('model_forward_entered',False) for p in (out/'calls_PRIVATE').glob('*.json'))
 def one(row,points,indices,nraw,phrase,role):
  nonlocal logical,new_calls,reused,entered_total
  logical+=1;sid=row['id'];path=out/'calls_PRIVATE'/(sid+'__'+role+'.json')
  target=out/('samples_PRIVATE' if role=='subject' else 'references_PRIVATE')/(sid+('.npz' if role=='subject' else '__'+role+'.npz'))
  actual=phrase+' . not mentioned'
  binding={'id_PRIVATE':sid,'role':role,'point_sample_sha256':array_sha(points),
   'point_indices_sha256':array_sha(indices),'actual_text_sha256':hashlib.sha256(actual.encode()).hexdigest(),
   'graph_sha256':index[sid]['sha256'],'source_checkpoint_sha256':SOURCE_SHA,
   'raw_point_count':int(nraw),'prefix_retained':role=='subject','plan_sha256':runbinding['plan_sha256']}
  if path.exists():
   record=read(path);assert record['binding']==binding and record['state']=='COMPLETED', 'Incomplete call cannot automatically resume/retry'
   assert sha(target)==record['raw_output_sha256'];reused+=1
   return {'file':target.relative_to(out).as_posix(),'sha256':sha(target),
           'call':path.relative_to(out).as_posix(),'capture_receipt':record['capture_receipt']}
  record={'state':'STARTED','binding':binding,'model_forward_entered':False};dump(path,record)
  def entered():
   nonlocal entered_total,new_calls
   assert entered_total<plan['max_grounder_forwards']
   entered_total+=1;new_calls+=1;record['model_forward_entered']=True;dump(path,record)
  tick=time.perf_counter()
  try:
   arrays,receipt=source_capture(model,points,indices,phrase,retain_prefix=role=='subject',on_forward_enter=entered)
   save_arrays(target,arrays)
   record.update(state='RAW_MODEL_OUTPUT_SAVED',raw_output_sha256=sha(target),capture_receipt=receipt);dump(path,record)
   if role=='subject':
    canonical=ROOT/CANONICAL/'samples_PRIVATE'/(sid+'.npz')
    assert sha(canonical)==canonicalseal['files'][canonical.relative_to(ROOT/CANONICAL).as_posix()]
    with np.load(canonical,allow_pickle=False) as old:record['source_bridge']=bridge_capture(arrays,old['boxes'],old['scores'])
   record.update(state='COMPLETED',seconds=time.perf_counter()-tick);dump(path,record)
  except BaseException as error:
   record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-tick)
   dump(path,record);raise
  return {'file':target.relative_to(out).as_posix(),'sha256':sha(target),
          'call':path.relative_to(out).as_posix(),'capture_receipt':receipt}
 try:
  for row in rows:
   points,indices,nraw=sample_points(row)
   subject=one(row,points,indices,nraw,row['caption'],'subject')
   refs={r['entity']:one(row,points,indices,nraw,r['own_query'],r['entity']) for r in graphs[row['id']]['references']}
   result={'id_PRIVATE':row['id'],'platform':row['platform'],'subject':subject,'references':refs,
    'raw_point_count':int(nraw),'A_seed':row['seed'],'point_sample_sha256':array_sha(points),
    'point_indices_sha256':array_sha(indices),'subject_A_canonical_bitexact':True,
    'reference_queries_have_independent_256_slots_not_subject_slot_ids':True,
    'new_B_captures':0,'external_visual_calls':0,'GT_read':False}
   path=out/'cases_PRIVATE'/(row['id']+'.json')
   if path.exists():assert read(path)==result
   else:dump(path,result)
   completed.append(row['id']);dump(out/'PROGRESS.json',{'completed':len(completed),'n':336,
    'new_forwards_this_process':new_calls,'completed_calls_reused_this_process':reused,
    'durable_entered_forwards':entered_total,'seconds_this_process':time.perf_counter()-start})
   if len(completed)%16==0:print('CAPTURE51',len(completed),logical,new_calls,reused,flush=True)
  assert logical==plan['registered_logical_grounder_calls'] and entered_total==logical
  assert all(read(p)['state']=='COMPLETED' for p in (out/'calls_PRIVATE').glob('*.json'))
  assert versions==[(n,p._version) for n,p in model.named_parameters()] and fingerprint(model)==sourcefp
  counts={'n':336,'logical_grounder_calls':logical,'physical_grounder_forwards_all_processes':entered_total,
   'physical_grounder_forwards_this_process':new_calls,'completed_calls_reused_this_process':reused,
   'subject_A_bitexact_bridges':336,'new_B_captures':0,'external_visual_calls':0,'optimizer_steps':0,
   'model_unchanged':True,'source_fingerprint':sourcefp,'load_seconds':load_seconds,
   'seconds_this_process':time.perf_counter()-start,'peak_bytes':int(torch.cuda.max_memory_allocated())}
  dump(out/'COUNTS.json',counts)
  dump(out/'SEALED.json',{'state':'R51_DEV336_SINGLE_WAYMO_MODEL_SUBJECT_REFERENCE_HYPOTHESES_NO_GT',**counts,
   'plan_sha256':runbinding['plan_sha256'],'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
    if p.is_file() and p.name!='SEALED.json' and not p.name.endswith('.tmp')}})
  print('CAPTURE51_SEALED',counts,flush=True)
 except BaseException as error:
  dump(out/f'FAILED_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),
   'completed_cases':len(completed),'new_forwards_this_process':new_calls,'durable_entered_forwards':entered_total,
   'GT_read':False,'automatic_retry_or_case_skip':False});raise


if __name__=='__main__':main()
