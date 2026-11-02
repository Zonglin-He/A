"""UNEXECUTED full5912 reference-only capture; subject source is R8 raw cache.

No source subject forward, GT, original metadata, image or external-model
evidence is read. Only the general R8 manifest is used; no special final-pool
file. Root must lock the selected method and actual graph-derived reference
budget first. Every executed reference uses reference51's textual-head rule.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,random,re
import numpy as np
from model51 import ROOT,load_source,fingerprint,native_matching,array_sha,SOURCE_SHA
from reference51 import text_head_mask

MANIFEST='research/round8/experiment/full5912_gpu_inputs.json'
SOURCE='crossplatform/round8/full5912_attempt001'
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


def save(path,value):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**value)
 tmp.replace(path)


def context(args):
 pp=Path(args.plan).resolve();grant=read(args.authorization);plan=read(pp)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_SELECTED_FULL5912_REFERENCE_ONLY_CAPTURE_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['n']==5912 and plan['manifest']==MANIFEST and plan['subject_source_cache']==SOURCE
 assert plan['source_plan']==SOURCE_PLAN and plan['source_checkpoint_sha256']==SOURCE_SHA
 assert plan['method_selected_before_full_run'] is True and isinstance(plan['selected_method'],str) and plan['selected_method']
 assert plan['new_subject_forwards']==0 and plan['no_GT_or_external_model_evidence'] is True
 required={MANIFEST,SOURCE_PLAN,SOURCE+'/SEALED.json',plan['pure_text_root']+'/SEALED.json'}
 assert required<=set(plan['inputs_sha256'])
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   assert 'FINAL' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h,rel
 assert all(p in plan['programs_sha256'] for p in ('crossplatform/round51/reference_full51.py',
  'crossplatform/round51/reference51.py','crossplatform/round51/model51.py'))
 source=read(ROOT/SOURCE/'SEALED.json')
 assert source['n_samples']==5912 and source['manifest_sha256']==sha(ROOT/MANIFEST)
 assert source['source_checkpoint_sha256']==SOURCE_SHA
 textroot=(ROOT/plan['pure_text_root']).resolve()
 assert textroot==ROOT/'crossplatform/round50/full5912_attempt001/text'
 textseal=read(textroot/'SEALED.json');assert textseal['n']==5912 and textseal['state']=='R50_FULL5912_LITERAL_TEXT_COMPLETE'
 rows=read(ROOT/MANIFEST)
 assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 assert {p:sum(r['platform']==p for r in rows) for p in ('drone','quad')}=={'drone':2984,'quad':2928}
 graphs={};budget=0
 for row in rows:
  sid=row['id'];assert re.fullmatch('[A-Za-z0-9_-]+',sid)
  path=textroot/'cases_PRIVATE'/(sid+'.json');assert sha(path)==textseal['files'][path.relative_to(textroot).as_posix()]
  g=read(path);refs=g['references'];assert len(refs)<=4 and len({r['entity'] for r in refs})==len(refs)
  assert all(re.fullmatch('R[0-9]+',r['entity']) and isinstance(r['own_query'],str) and r['own_query'] for r in refs)
  graphs[sid]={'graph':g,'sha256':sha(path)};budget+=len(refs)
 assert budget==plan['registered_reference_calls']==plan['max_grounder_forwards']
 out=(ROOT/plan['output_root']).resolve();assert out.is_relative_to(ROOT/'crossplatform/round51')
 binding={'plan_sha256':sha(pp),'manifest_sha256':sha(ROOT/MANIFEST),
  'source_cache_seal_sha256':sha(ROOT/SOURCE/'SEALED.json'),'pure_text_seal_sha256':sha(textroot/'SEALED.json'),
  'registered_reference_calls':budget,'new_subject_forwards':0,'selected_method':plan['selected_method']}
 if out.exists():assert args.resume and not (out/'SEALED.json').exists() and read(out/'RUN_BINDINGS.json')==binding
 else:out.mkdir(parents=True);dump(out/'RUN_BINDINGS.json',binding)
 for name in ('calls_PRIVATE','references_PRIVATE','subjects_PRIVATE','cases_PRIVATE'):(out/name).mkdir(exist_ok=True)
 return plan,rows,graphs,out,source,binding


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True);p.add_argument('--resume',action='store_true')
 args=p.parse_args();plan,rows,graphs,out,source_seal,binding=context(args);start=time.perf_counter()
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
 import torch
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 try:model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model)
 except BaseException as error:
  dump(out/'LOAD_FAILURE.json',{'error_type':type(error).__name__,'error':str(error),'grounder_forwards':0});raise
 versions=[(n,p._version) for n,p in model.named_parameters()];load_seconds=time.perf_counter()-start
 entered=sum(read(path).get('model_forward_entered',False) for path in (out/'calls_PRIVATE').glob('*.json'))
 new=logical=reused=usable=cases=0
 try:
  for row in rows:
   sid=row['id'];sourcepath=ROOT/SOURCE/(sid+'.npz')
   assert sha(sourcepath)==source_seal['files'][sourcepath.name]
   # Select only frozen SOURCE arrays/hashes; never load old R8 G/CLIP/TTA.
   with np.load(sourcepath,allow_pickle=False) as z:
    assert str(z['sample_id'])==sid
    boxes=z['source_boxes_raw6'].copy();scores=z['source_text_scores256'].copy()
    pointsha=z['point_sample_sha256'].tobytes().hex();indexsha=z['raw_indices_sha256'].tobytes().hex()
   assert boxes.shape==(256,6) and scores.shape==(256,) and boxes.dtype==scores.dtype==np.float32
   assert np.isfinite(boxes).all() and np.isfinite(scores).all()
   subjectpath=out/'subjects_PRIVATE'/(sid+'.npz')
   if subjectpath.exists():
    with np.load(subjectpath,allow_pickle=False) as z:
     for key,value in (('boxes',boxes),('scores',scores)):
      assert z[key].shape==value.shape and z[key].dtype==value.dtype and z[key].tobytes()==value.tobytes()
   else:save(subjectpath,{'boxes':boxes,'scores':scores})
   refs=graphs[sid]['graph']['references'];result={}
   if refs:
    points,indices,nraw=sample_points(row)
    assert array_sha(points)==pointsha and array_sha(indices)==indexsha, 'Original R8 sampling identity required'
   for ref in refs:
    logical+=1;entity=ref['entity'];callpath=out/'calls_PRIVATE'/(sid+'__'+entity+'.json')
    target=out/'references_PRIVATE'/(sid+'__'+entity+'.npz')
    text,mask,textarrays,maskinfo=text_head_mask(model.tokenizer,ref['own_query'],ref.get('head'))
    callbinding={'id_PRIVATE':sid,'entity':entity,'plan_sha256':binding['plan_sha256'],
     'point_sample_sha256':pointsha,'point_indices_sha256':indexsha,'graph_sha256':graphs[sid]['sha256'],
     'actual_model_text_sha256':maskinfo['actual_model_text_sha256'],'source_checkpoint_sha256':SOURCE_SHA}
    if callpath.exists():
     record=read(callpath);assert record['binding']==callbinding and record['state']=='COMPLETED', 'No automatic retry of incomplete call'
     assert sha(target)==record['raw_output_sha256'];reused+=1
    else:
     record={'state':'STARTED','binding':callbinding,'model_forward_entered':False,'text_head_mask':maskinfo};dump(callpath,record)
     tick=time.perf_counter()
     try:
      pc=torch.as_tensor(points,device='cuda')[None];tm=mask.to('cuda')
      assert entered<plan['max_grounder_forwards'];entered+=1;new+=1
      record['model_forward_entered']=True;dump(callpath,record)
      with torch.no_grad():o=model({'point_clouds':pc,'text':[text]})
      q=o['last_proj_queries'];tokens=o['proj_tokens'];probability,logodds=native_matching(q,tokens,tm)
      arrays={'boxes':torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy(),
       'scores':probability.cpu().numpy(),'logodds':logodds.cpu().numpy(),
       'query':q[0].cpu().numpy(),'tokens':tokens[0].cpu().numpy(),
       'token_logits':(q@tokens.transpose(-1,-2)/.07)[0].cpu().numpy(),**textarrays,
       'point_indices':indices,'query_anchor_xyz':o['query_points_xyz'][0].cpu().numpy(),
       'query_points_sample_inds':o['query_points_sample_inds'][0].cpu().numpy()}
      save(target,arrays);record.update(state='RAW_HEADSPAN_OUTPUT_SAVED',raw_output_sha256=sha(target));dump(callpath,record)
      record.update(state='COMPLETED',seconds=time.perf_counter()-tick);dump(callpath,record)
     except BaseException as error:
      record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));dump(callpath,record);raise
    usable+=int(maskinfo['reference_usable'])
    result[entity]={'file':target.relative_to(out).as_posix(),'sha256':sha(target),'mask':maskinfo,
     'reference_query_slots_not_subject_query_slots':True,'native_logodds_not_calibrated_entity_or_UNKNOWN_probability':True}
   case={'id_PRIVATE':sid,'platform':row['platform'],'subject':{'file':subjectpath.relative_to(out).as_posix(),
    'sha256':sha(subjectpath),'original_R8_source_file_sha256':sha(sourcepath),
    'native_subject_logodds_and_prefix_unavailable_not_reconstructed_from_clipped_probability':True},
    'references':result,'graph_sha256':graphs[sid]['sha256'],'point_sample_sha256':pointsha,
    'point_indices_sha256':indexsha,'new_subject_forwards':0,'GT_read':False}
   cp=out/'cases_PRIVATE'/(sid+'.json')
   if cp.exists():assert read(cp)==case
   else:dump(cp,case)
   cases+=1
   if cases%64==0:
    dump(out/'PROGRESS.json',{'cases':cases,'n':5912,'logical_references':logical,'new_forwards':new,
     'durable_entered_forwards':entered,'reused_completed':reused,'seconds':time.perf_counter()-start})
    print('REFERENCE_FULL51',cases,logical,new,reused,flush=True)
  assert cases==5912 and entered==logical==plan['registered_reference_calls']
  assert all(read(path)['state']=='COMPLETED' for path in (out/'calls_PRIVATE').glob('*.json'))
  assert versions==[(n,p._version) for n,p in model.named_parameters()] and fingerprint(model)==sourcefp
  counts={'n':5912,'source_subjects_reused_bitexact':5912,'new_subject_forwards':0,'logical_reference_calls':logical,
   'reference_forwards_all_processes':entered,'reference_forwards_this_process':new,'resumed_completed_references':reused,
   'text_headspan_usable_references':usable,'UNKNOWN_references_not_negative':logical-usable,
   'model_unchanged':True,'source_fingerprint':sourcefp,'optimizer_steps':0,'external_visual_calls':0,'GT_read':False,
   'load_seconds':load_seconds,'seconds_this_process':time.perf_counter()-start,'peak_bytes':int(torch.cuda.max_memory_allocated())}
  dump(out/'COUNTS.json',counts)
  dump(out/'SEALED.json',{'state':'R51_FULL5912_SELECTED_METHOD_REFERENCE_ONLY_HEADSPAN_CAPTURE_NO_GT',**counts,
   'plan_sha256':binding['plan_sha256'],'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SEALED.json'}})
  print('REFERENCE_FULL51_SEALED',counts,flush=True)
 except BaseException as error:
  dump(out/f'FAILED_{time.time_ns()}.json',{'error_type':type(error).__name__,'error':str(error),
   'cases':cases,'new_forwards':new,'durable_entered':entered,'GT_read':False,'automatic_retry':False});raise


if __name__=='__main__':main()
