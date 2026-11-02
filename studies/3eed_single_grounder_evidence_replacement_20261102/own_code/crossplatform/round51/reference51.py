"""R51 reference-only textual-head readout repair, before task GT scoring.

No locked source/capture code or old cache is modified. Same source model and
573 identical reference inputs are rerun once; only the readout mask changes.
Unknown/generic heads retain diagnostic own-query scores, never negative
evidence. Root owns the GPU lease and execution grant; import is inert.
"""
from pathlib import Path
import argparse,hashlib,json,re,sys,time,random
import numpy as np
from model51 import ROOT,load_source,fingerprint,native_matching,array_sha,SOURCE_SHA,SENTINEL

MANIFEST='research/round24/input/ALL336_PRIVATE.json'
SOURCE_PLAN='research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'
GENERIC_HEADS=frozenset({'','obj','object','objects','thing','things','item','items','target','entity','unknown','none'})


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save(path,arrays):
 tmp=path.with_name(path.name+'.tmp')
 with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
 tmp.replace(path)


def text_head_mask(tokenizer,own_query,head):
 """Pure text offsets, first exact word-boundary head; no GT/class lexicon."""
 import torch
 assert isinstance(own_query,str) and own_query
 text=own_query+SENTINEL
 tokenized=tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True)
 offsets=tokenized['offset_mapping'][0];assert len(offsets)<=256
 value=head.strip() if isinstance(head,str) else ''
 generic=value.casefold() in GENERIC_HEADS
 match=None if generic else re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',own_query,re.I)
 usable=match is not None
 lo,hi=match.span() if usable else (0,len(own_query))
 mask=(offsets[:,1]>lo)&(offsets[:,0]<hi)&(offsets[:,1]>offsets[:,0])
 assert mask.any() and (~mask).any()
 # No positive token may reach into the appended sentinel, even fallback.
 assert not ((offsets[:,0]>=len(own_query))&mask).any()
 receipt={'availability':'TEXT_HEAD_SPAN_AVAILABLE' if usable else 'UNKNOWN_GENERIC_HEAD' if generic else 'UNKNOWN_HEAD_NOT_FOUND_IN_OWN_QUERY',
  'reference_usable':usable,'UNKNOWN_is_not_negative_evidence':not usable,
  'positive_character_span':[lo,hi],'positive_token_count':int(mask.sum()),
  'negative_token_count':int((~mask).sum()),'fallback_positive_only_own_query_excludes_sentinel':not usable,
  'pure_graph_head_not_GT_class_or_3D_identity_truth':True,'actual_model_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
 arrays={'token_offsets':offsets.numpy(),'input_ids':tokenized['input_ids'][0].numpy(),
         'text_attention_mask':tokenized['attention_mask'][0].numpy(),'headspan_token_mask':mask.numpy()}
 return text,mask,arrays,receipt


def context(args):
 pp=Path(args.plan).resolve();grant=read(args.authorization);plan=read(pp)
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_REFERENCE_HEADSPAN_READOUT_REPAIR_NO_GT'
 assert grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['manifest']==MANIFEST and plan['n_subject_rows']==336 and plan['n_references']==573
 assert plan['source_plan']==SOURCE_PLAN and plan['source_checkpoint_sha256']==SOURCE_SHA
 assert plan['max_grounder_forwards']==573 and plan['new_subject_forwards']==0
 assert plan['no_GT_or_external_models_or_old_cache_overwrite'] is True
 required={MANIFEST,SOURCE_PLAN,plan['graph_index'],plan['pure_text_root']+'/SEALED.json',
           plan['source_capture_root']+'/SEALED.json'}
 assert required<=set(plan['inputs_sha256'])
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   assert 'FINAL' not in rel and 'round46/final' not in rel and sha(ROOT/rel)==h,rel
 assert all(p in plan['programs_sha256'] for p in ('crossplatform/round51/reference51.py','crossplatform/round51/model51.py'))
 oldroot=(ROOT/plan['source_capture_root']).resolve();assert oldroot.is_relative_to(ROOT/'crossplatform/round51')
 oldseal=read(oldroot/'SEALED.json')
 assert oldseal['state']=='R51_DEV336_SINGLE_WAYMO_MODEL_SUBJECT_REFERENCE_HYPOTHESES_NO_GT'
 assert oldseal['n']==336 and oldseal['physical_grounder_forwards_all_processes']==909
 textroot=(ROOT/plan['pure_text_root']).resolve()
 assert textroot==ROOT/'crossplatform/round50/full5912_attempt001/text'
 ts=read(textroot/'SEALED.json');assert ts['state']=='R50_FULL5912_LITERAL_TEXT_COMPLETE'
 rows=read(ROOT/MANIFEST);index=read(ROOT/plan['graph_index']);assert len(rows)==336 and set(index)=={r['id'] for r in rows}
 graphs={};count=0
 for row in rows:
  route=index[row['id']];relative=Path(route['path']);assert not relative.is_absolute() and '..' not in relative.parts
  path=(ROOT/relative).resolve();assert path.is_relative_to(textroot)
  assert route['caption_sha256']==hashlib.sha256(row['caption'].encode()).hexdigest()
  assert sha(path)==route['sha256']==ts['files'][path.relative_to(textroot).as_posix()]
  graph=read(path);refs=graph['references'];assert len(refs)<=4 and len({r['entity'] for r in refs})==len(refs)
  assert all(re.fullmatch('R[0-9]+',r['entity']) and isinstance(r['own_query'],str) and r['own_query'] for r in refs)
  graphs[row['id']]=graph;count+=len(refs)
 assert count==573
 out=(ROOT/plan['output_root']).resolve();assert out.is_relative_to(ROOT/'crossplatform/round51') and out!=oldroot
 assert not out.is_relative_to(oldroot), 'Never add files inside the sealed old capture'
 assert not out.exists();out.mkdir(parents=True)
 for name in ('calls_PRIVATE','references_PRIVATE','cases_PRIVATE'):(out/name).mkdir()
 dump(out/'RUN_BINDINGS.json',{'plan_sha256':sha(pp),'old_capture_seal_sha256':sha(oldroot/'SEALED.json'),
  'graph_index_sha256':sha(ROOT/plan['graph_index']),'registered_reference_calls':573,
  'old909_forwards_preserved_not_subtracted_from_cost':909})
 return plan,rows,index,graphs,out,oldroot,oldseal


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True);args=parser.parse_args()
 plan,rows,index,graphs,out,oldroot,oldseal=context(args);start=time.perf_counter()
 sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
 import torch
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 try:model=load_source(read(ROOT/SOURCE_PLAN));sourcefp=fingerprint(model)
 except BaseException as error:
  dump(out/'LOAD_FAILURE.json',{'error_type':type(error).__name__,'error':str(error),'grounder_forwards':0});raise
 versions=[(n,p._version) for n,p in model.named_parameters()]
 load_seconds=time.perf_counter()-start;entered=completed=usable=0;casecount=0
 try:
  for row in rows:
   sid=row['id'];oldcasepath=oldroot/'cases_PRIVATE'/(sid+'.json')
   assert sha(oldcasepath)==oldseal['files'][oldcasepath.relative_to(oldroot).as_posix()]
   prior=read(oldcasepath);refs=graphs[sid]['references'];result={}
   if refs:
    points,indices,nraw=sample_points(row)
    assert array_sha(points)==prior['point_sample_sha256'] and array_sha(indices)==prior['point_indices_sha256']
    assert nraw==prior['raw_point_count']
   for ref in refs:
    entity=ref['entity'];path=out/'calls_PRIVATE'/(sid+'__'+entity+'.json')
    target=out/'references_PRIVATE'/(sid+'__'+entity+'.npz')
    oldref=prior['references'][entity];oldnpz=oldroot/oldref['file']
    assert sha(oldnpz)==oldref['sha256']==oldseal['files'][oldnpz.relative_to(oldroot).as_posix()]
    text,mask,textarrays,maskinfo=text_head_mask(model.tokenizer,ref['own_query'],ref.get('head'))
    assert maskinfo['actual_model_text_sha256']==oldref['capture_receipt']['actual_model_text_sha256']
    record={'state':'STARTED','model_forward_entered':False,'id_PRIVATE':sid,'entity':entity,
     'plan_sha256':sha(args.plan),'old_reference_file_sha256':sha(oldnpz),
     'point_sample_sha256':prior['point_sample_sha256'],'point_indices_sha256':prior['point_indices_sha256'],
     'graph_sha256':index[sid]['sha256'],'text_head_mask':maskinfo};dump(path,record)
    tick=time.perf_counter()
    try:
     pc=torch.as_tensor(points,device='cuda')[None];tm=mask.to('cuda')
     assert entered<573;entered+=1;record['model_forward_entered']=True;dump(path,record)
     with torch.no_grad():outputs=model({'point_clouds':pc,'text':[text]})
     q=outputs['last_proj_queries'];tokens=outputs['proj_tokens']
     probability,logodds=native_matching(q,tokens,tm)
     logits=(q@tokens.transpose(-1,-2)/.07)[0]
     arrays={'boxes':torch.cat([outputs['last_center'],outputs['last_pred_size']],-1)[0].cpu().numpy(),
      'scores':probability.cpu().numpy(),'logodds':logodds.cpu().numpy(),'token_logits':logits.cpu().numpy(),
      'query':q[0].cpu().numpy(),'tokens':tokens[0].cpu().numpy(),**textarrays,
      'point_indices':indices,'query_anchor_xyz':outputs['query_points_xyz'][0].cpu().numpy(),
      'query_points_sample_inds':outputs['query_points_sample_inds'][0].cpu().numpy()}
     save(target,arrays);record.update(state='RAW_HEADSPAN_OUTPUT_SAVED',raw_output_sha256=sha(target));dump(path,record)
     with np.load(oldnpz,allow_pickle=False) as old:
      for key in ('boxes','query_anchor_xyz','query_points_sample_inds','point_indices'):
       assert arrays[key].shape==old[key].shape and arrays[key].dtype==old[key].dtype and arrays[key].tobytes()==old[key].tobytes(),key
     record.update(state='COMPLETED',old_reference_geometry_and_slots_bitexact=True,seconds=time.perf_counter()-tick)
     dump(path,record);completed+=1;usable+=int(maskinfo['reference_usable'])
    except BaseException as error:
     record.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-tick)
     dump(path,record);raise
    result[entity]={'file':target.relative_to(out).as_posix(),'sha256':sha(target),'mask':maskinfo,
      'different_phrase_query_indices_are_independent':True,'old_geometry_unchanged':True}
   dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'subject_forward_count':0,'references':result,
       'old_subject_capture_unchanged':prior['subject'],'graph_sha256':index[sid]['sha256']})
   casecount+=1
   if casecount%16==0:print('REFERENCE51',casecount,completed,usable,flush=True)
  assert casecount==336 and entered==completed==573
  assert versions==[(n,p._version) for n,p in model.named_parameters()] and fingerprint(model)==sourcefp
  counts={'n_subject_rows':336,'n_references':573,'new_reference_forwards':entered,'new_subject_forwards':0,
   'text_head_span_usable_references':usable,'UNKNOWN_references_not_negative':573-usable,
   'old909_forwards_preserved':909,'cumulative_source_reference_capture_forwards':909+entered,
   'all573_old_reference_boxes_and_slots_bitexact':True,'main_subject_mask_boxes_scores_unchanged':True,
   'source_fingerprint':sourcefp,'model_unchanged':True,'optimizer_steps':0,'GT_read':False,
   'load_seconds':load_seconds,'seconds':time.perf_counter()-start,'peak_bytes':int(torch.cuda.max_memory_allocated())}
  dump(out/'COUNTS.json',counts)
  dump(out/'SEALED.json',{'state':'R51_REFERENCE_GRAPH_HEADSPAN_READOUT_REPAIR_NO_GT',**counts,
   'plan_sha256':sha(args.plan),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SEALED.json'}})
  print('REFERENCE51_SEALED',counts,flush=True)
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'entered_forwards':entered,
    'completed_references':completed,'completed_subject_rows':casecount,'GT_read':False,'automatic_retry':False});raise


if __name__=='__main__':main()
