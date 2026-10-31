"""Independent post-seal R15 direct8 audit. No sensor decode, G solve or labels.

Run only after root explicitly releases the actual formal semantic/direct seals.
This audits process/increment validity, not identity, pseudo quality or task gain.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'crossplatform/round15')]
from semantic_graph_v4 import (parse_caption_graph,parse_visual_graph,combine_visual_graphs,match_scene_graph,proposal_consensus,match_relation_flip_diagnostic)
from semantic_runner_v3 import json_safe
ARMS=('source','proposal_assoc','proposal_geo','graph_assoc','graph_geo','graph_geo_noRGB','graph_geo_relation_disabled','graph_geo_relation_flipped_diagnostic')
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
FLAGS={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
CHECKS=0

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(x):return hashlib.sha256(json.dumps(json_safe(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def resolve(p):
 p=Path(p);return p if p.is_absolute() else R/p
def require(x,reason):
 global CHECKS
 if not x:raise AssertionError(reason)
 CHECKS+=1
def verify(root,digest=None):
 root=resolve(root);s=read(root/'SEALED.json')
 if digest is not None:require(sha(root/'SEALED.json')==digest,'Explicit root seal hash')
 expected=set()
 for rel,h in s['files'].items():
  p=Path(rel);require(not p.is_absolute() and '..' not in p.parts,'Relative member only')
  q=(root/p).resolve();require(q.is_relative_to(root.resolve()) and q.is_file() and sha(q)==h,'Sealed member hash');expected.add(q)
 actual={p.resolve() for p in root.rglob('*') if p.is_file() and p.name!='SEALED.json'}
 require(actual==expected,'Complete member allowlist, no unsealed extras')
 return root,s
def evalcopy(raw):
 out=raw.copy();out[:,3:]=np.maximum(out[:,3:],np.float32(1e-6));return out
def finite_array(x,shape,dtype=None):
 x=np.asarray(x);require(x.shape==shape and np.isfinite(x).all(),'Finite array shape')
 if dtype is not None:require(x.dtype==dtype,'Exact native dtype')
 return x
def geometry_arrays(geo,anchor,q):
 boxes=finite_array(geo['arm_boxes_raw6'],(4,6)).astype(np.float32)
 require(np.array_equal(boxes[0],anchor),'Geometry source is associated anchor')
 require(np.array_equal(np.asarray(geo['arm_query_index']),np.repeat(q,4)),'Geometry never reselects query')
 require(np.array_equal(boxes[3],anchor+np.float32(.5)*(boxes[1]-anchor)),'Geometry internal fixed half')
 require(np.array_equal(np.asarray(geo['actual_changed_axes']),boxes!=anchor),'Geometry actual changes')
 active=np.asarray(geo['active_mask6'],bool);accepted=np.asarray(geo['acceptance_mask6'],bool)
 require(active.shape==accepted.shape==(4,6) and np.array_equal(active,accepted),'Geometry search-domain/acceptance arrays')
 require(np.array_equal(active[1:],np.tile(active[1],(3,1))),'Same full/off/half active domain')
 for f in ('point_support_sha256','mask_sha256_by_arm'):require(len(set(geo[f]))==1,'Same geometry support/mask')
 diag=geo['diagnostics'];require(diag['fixed_q0']==q and diag['same_support_gate_active_bounds_full_noRGB'] is True,'Geometry anchor/gate provenance')
 cfg=diag['config'];require(cfg['center_bound_fraction']==.25 and cfg['center_bound_cap_m']==.5 and cfg['max_size_expansion']==1.2,'Frozen geometric bounds')
 if (anchor[3:]>0).all():
  # Outward one float32 ULP permits final float64→float32 delivery rounding,
  # not an empirical tolerance chosen after seeing these predictions.
  cb=np.minimum(.25*anchor[3:].astype(float),.5)
  lo=np.nextafter((anchor[:3].astype(float)-cb).astype(np.float32),np.float32(-np.inf));hi=np.nextafter((anchor[:3].astype(float)+cb).astype(np.float32),np.float32(np.inf))
  upper=np.nextafter((1.2*anchor[3:].astype(float)).astype(np.float32),np.float32(np.inf))
  require((boxes[1:,:3]>=lo).all() and (boxes[1:,:3]<=hi).all(),'Delivered centers inside fixed bounds plus quantization')
  require((boxes[1:,3:]>=anchor[3:]).all() and (boxes[1:,3:]<=upper).all(),'No shrink and bounded expansion')
 for j,status in enumerate(geo['arm_status']):
  if status.startswith('source_fallback:') or status in ('numerical_invalid_source_fallback','source_already_satisfies_positive_constraints'):
   require(np.array_equal(boxes[j],anchor),'Failure/no-violation result is own anchor')
 require(np.array_equal(boxes[~active],np.tile(anchor,(4,1))[~active]),'Inactive coordinates cannot change')
 return boxes
def audit_arrays(d,pool,scores,trace):
 require(tuple(d['arm_names'].tolist())==ARMS,'Exact eight arm roles')
 require(np.asarray(d['q_source']).shape==() and np.issubdtype(np.asarray(d['q_source']).dtype,np.integer),'Scalar integer source query')
 for key in ('actual_changed_axes','geometry_increment_axes'):
  require(np.asarray(d[key]).shape==(8,6) and np.asarray(d[key]).dtype==bool,'Boolean axis metadata')
 raw=finite_array(d['arm_boxes_raw6'],(8,6),np.float32);q=int(d['q_source']);aq=np.asarray(d['arm_query_index'])
 require(q==int(np.argmax(scores)) and aq.shape==(8,) and np.issubdtype(aq.dtype,np.integer) and ((aq>=0)&(aq<256)).all(),'Source/arm query domain')
 require(np.array_equal(d['source_boxes_raw6'],pool) and np.array_equal(d['source_text_scores256'],scores),'Immutable source pool/scores bitexact')
 require(aq[0]==q and np.array_equal(raw[0],pool[q]),'Original q0 source readout')
 require(np.array_equal(d['arm_boxes_eval6'],evalcopy(raw)),'Official float32 size clamp only on eval copy')
 require(np.array_equal(d['arm_geometry_anchor_raw6'],pool[aq]),'Every arm source-query anchor')
 require(np.array_equal(d['actual_changed_axes'],raw!=raw[0]) and np.array_equal(d['geometry_increment_axes'],raw!=pool[aq]),'Association versus geometry changed flags')
 require(aq[3]==aq[4]==aq[5],'Main common associated q1')
 for f in ('arm_support_sha256','arm_mask_sha256','arm_association_accepted'):require(np.array_equal(d[f][3:6],np.repeat(d[f][3],3)),'Main common support/mask/association gate')
 branches=trace['geometry_trace']['branches'];mapping=[(None,0),('proposal',1),('proposal',2),('graph',1),('graph',2),('graph',3),('relation_disabled',2),('relation_flip',2)]
 for name,b in branches.items():
  br=np.asarray(b['arm_boxes_raw6'],np.float32);bq=np.asarray(b['arm_query_index']);anchor=pool[int(b['q_association'])]
  require(br.shape==(5,6) and np.array_equal(br[0],pool[q]),'Branch source bridge')
  if not b['association_accepted']:
   require(np.array_equal(br,np.tile(pool[q],(5,1))) and np.all(bq==q),'Unaccepted whole branch stays q0')
  else:
   qq=int(b['q_association']);require(np.all(bq[1:]==qq) and np.array_equal(br[1],pool[qq]),'Accepted association original proposal')
   geo=geometry_arrays(b['geometry'],anchor,qq);require(np.array_equal(br[2:],geo[1:]),'Delivered branch geometry arrays')
 for j,(name,index) in enumerate(mapping):
  if name is None:continue
  b=branches[name];require(np.array_equal(raw[j],np.asarray(b['arm_boxes_raw6'][index],np.float32)) and aq[j]==b['arm_query_index'][index] and str(d['arm_status'][j])==b['arm_status'][index],'NPZ equals sealed branch trace')
  require(bool(d['arm_association_accepted'][j])==b['association_accepted'],'Association gate flag')
  if b['association_accepted']:
   require(str(d['arm_support_sha256'][j])==b['diagnostics']['selected_support_sha256'] and str(d['arm_mask_sha256'][j])==b['diagnostics']['selected_mask_sha256'],'Selected support/mask binding')
  else:require(str(d['arm_support_sha256'][j])==str(d['arm_mask_sha256'][j])=='0'*64,'Undefined support hash placeholder')
 if not trace['geometry_trace']['relation_flip_applicable']:
  require(np.array_equal(raw[7],raw[4]) and aq[7]==aq[4] and trace['geometry_trace']['nonapplicable_flip_exact_fullgraph_array_reuse'],'Nonapplicable flip exact full result')
 return raw,aq

def semantic_replay(sem,alias,row,meta,payload,counts):
 text=read(sem/'calls_PRIVATE'/(alias+'__text.json'));rr=text['result']
 graph=parse_caption_graph(rr['raw_answer_PRIVATE'],row['caption']) if rr['ended_eos'] is True else {'status':'NA_text_output_budget_nonEOS','entities':[],'attributes':[],'relations':[]}
 require(json_safe(graph)==text['parsed'],'Exact frozen v4 text parse')
 parsed=[];maskids={o['mask_id'] for o in meta['objects']}
 for name,call in [('text',text),*[(f'visual_order{i}',read(sem/'calls_PRIVATE'/(alias+f'__visual_order{i}.json'))) for i in range(2)]]:
  r=call['result'];budget=1024 if name=='text' else 2048
  require(r['max_new_tokens']==budget and 0<=r['generated_tokens']<=budget and r['input_tokens']<=8192,'Actual frozen v4 call budgets')
  counts[name.split('_')[0]+'_calls']+=1;counts[name.split('_')[0]+'_EOS']+=r['ended_eos'];counts[name.split('_')[0]+'_nonEOS']+=not r['ended_eos']
  require(call['bindings']==payload['bindings'] and call['scene_metadata_sha256']==payload['scene_metadata_sha256'],'Call→case source binding')
  if name!='text':
   cand=call['candidate_mapping_PRIVATE'];require(len(cand)==len(maskids) and {x['mask_id'] for x in cand}==maskids and len({x['candidate_id'] for x in cand})==len(cand),'Complete numbered allmask inventory')
   out=parse_visual_graph(r['raw_answer_PRIVATE'],cand,graph) if r['ended_eos'] is True else {'status':'NA_visual_output_budget_nonEOS','role_proposals':{},'attribute_claims':{}}
   require(json_safe(out)==call['parsed'],'Exact frozen v4 visual parse');parsed.append(out)
 visual=combine_visual_graphs(parsed)
 expected={'text_graph':graph,'visual_parsed':parsed,'combined_visual':visual,'proposal_consensus':proposal_consensus(visual),'graph_match':match_scene_graph(graph,meta,visual),'graph_relation_disabled':match_scene_graph(graph,meta,visual,relation_check=False),'graph_relation_flip':match_relation_flip_diagnostic(graph,meta,visual),'source_q0_boxes_scores_used_for_semantic_selection':False,'all_mask_inventory_preserved':True,'agreement_is_not_correctness':True}
 require(json_safe(expected)==payload['semantics'],'All frozen semantic branches replay from already sealed compact observations')
 counts['text:'+graph['status']]+=1;counts['graph:'+expected['graph_match']['status']]+=1
 return expected

def mapping_readout_check(branch,semantic,meta,pool,scores):
 """Independently reduce recorded count matrices; no point geometry recompute.
 These are provenance-bound predicted observations, not independent identity GT.
 """
 require(branch['semantic_proposed_target_mask_id']==semantic['selected_target_mask_id'],'Branch uses its actual semantic proposal')
 m=branch['mapping']
 if m is None:
  require(not branch['association_accepted'],'No mapping cannot accept association');return
 objects=meta['objects'];n=np.asarray([o['n_exclusive'] for o in objects],int);M=len(n)
 count=np.asarray(m['inside_exclusive_counts']);require(count.shape==(256,M) and np.issubdtype(count.dtype,np.integer) and (count>=0).all() and (count<=n).all(),'Recorded exclusive count domain')
 den=count.sum(1);foreign=den[:,None]-count;frac=foreign/np.maximum(den,1)[:,None]
 pos=count/np.maximum(n,1)[None,:];positive=(pos>=.9)&(n[None,:]>=8)&(pool[:,3:]>0).all(1)[:,None]
 pollution=(foreign>=3)&(frac>.1);feasible=positive&~pollution
 require(np.array_equal(m['foreign_exclusive_counts'],foreign) and np.array_equal(m['pollution_rejected'],pollution) and np.array_equal(m['positive_feasible'],positive) and np.array_equal(m['feasible'],feasible),'Independent fixed count-based gates')
 groups=[g['query_indices'] for g in m['groups']];require(sorted(q for g in groups for q in g)==list(range(256)),'Groups partition original256 query slots')
 member=np.asarray(m['membership']);require(member.shape==(256,) and all(member[q]==i for i,g in enumerate(groups) for q in g),'Group membership indices')
 groupmasks=[{int(j) for q in g for j in np.flatnonzero(feasible[q])} for g in groups]
 mid=semantic['selected_target_mask_id'];j=next(i for i,o in enumerate(objects) if o['mask_id']==mid);qs=np.flatnonzero(feasible[:,j]);gids={int(member[q]) for q in qs}
 canonical_q=sorted(qs.tolist(),key=lambda q:(-float(scores[q]),q))[0] if len(gids)==1 and len(groupmasks[next(iter(gids))])==1 else None
 recorded=next(r for r in m['mask_mappings'] if r['mask_id']==mid)
 require(recorded['canonical_query_index']==canonical_q and branch['association_accepted']==(canonical_q is not None),'Reciprocal uniqueness and original-score canonical member')
 if canonical_q is not None:require(branch['q_association']==canonical_q and branch['selected_target_mask_id']==mid,'Readout uses actual reciprocal query and mask')

def main(a):
 global CHECKS
 CHECKS=0
 tick=time.perf_counter();require(not a.output.exists(),'New audit receipt only')
 direct,ds=verify(a.predictions,a.seal_sha256);sem,ss=verify(a.semantic,a.semantic_seal_sha256)
 require(ds['state']=='sealed_R15_eight_direct_arms_before_GT' and ds['n_cases']==ds['n_samples']==204 and ds['n_arms']==8,'Formal204 direct only')
 require(ss['state']=='sealed_R15_semantic204_three_calls_before_GT' and ss['n_cases']==204 and ss['n_model_calls']==612,'Formal204/612 semantic only, never engineering')
 require(ds['semantic_seal_sha256']==sha(sem/'SEALED.json') and ds['source_checkpoint_sha256']==SOURCE_SHA,'Actual producer/source provenance')
 dp=read(direct/'registered_plan.json');sp=read(sem/'registered_plan.json')
 require(sha(direct/'registered_plan.json')==ds['plan_sha256'] and sha(sem/'registered_plan.json')==ss['plan_sha256'],'Actual plan byte identity')
 require(dp['GT_read_permitted'] is False and sp['GT_read_permitted'] is False and dp['model_execution_permitted'] is False and dp['CPU_geometry_execution_permitted'] is True,'No label/direct-model producer scope')
 for seal in (ds,ss):
  for p,h in seal['program_sha256'].items():require(sha(resolve(p))==h,'Immutable executed code closure')
 require(ss['model_frozen_parameters_buffers_unchanged'] is True and ss['no_generated_code_GT_model_update_or_sourceROI'] is True,'Frozen executed VLM flags')
 require(ss['model_files_sha256']==sp['model_files_sha256'],'Executed asset digest declarations; no independent weight reload')
 for key in ('manifest','case_index','frame_index'):
  require(dp[key+'_sha256']==sp[key+'_sha256']==sha(resolve(dp[key])),'Identical bound input registries')
 require(sha(a.manifest)==dp['manifest_sha256'] and sha(a.cases)==dp['case_index_sha256'],'Explicit command input binding')
 rows=read(a.manifest);case_rows=read(a.cases);cases={c['id']:c for c in case_rows};require(len(rows)==len(cases)==204 and set(cases)=={r['id'] for r in rows},'All204unique source expressions')
 require(collections.Counter(c['role'] for c in cases.values())=={'fresh64_recent_content_relative_only':64,'R14_regression140_exposed':140},'Separate64/140 roles')
 require(collections.Counter((c['role'],c['platform']) for c in cases.values())=={('fresh64_recent_content_relative_only','drone'):32,('fresh64_recent_content_relative_only','quad'):32,('R14_regression140_exposed','drone'):64,('R14_regression140_exposed','quad'):76},'Four cohort/domain denominators')
 expected_seals={str(Path(c[k])/'SEALED.json') for c in cases.values() for k in ('source_dir','mask_dir')};require(len(expected_seals)==8 and set(ds['cache_seals_sha256'])==set(ss['cache_seals_sha256'])==expected_seals and ds['cache_seals_sha256']==ss['cache_seals_sha256'],'Eight shared capture closure')
 capture_files=0;captures={}
 for p,h in ds['cache_seals_sha256'].items():
  root,s=verify(resolve(p).parent,h);captures[str(root.resolve())]=s;capture_files+=len(s['files'])
  if 'source_checkpoint_sha256' in s:require(s['source_checkpoint_sha256']==SOURCE_SHA and s['source_unchanged'] and s['source_repeat_bitexact'] and s['strict_FP32_flags']==FLAGS,'Actual frozen source capture proof')
 aliases={c['case_alias'] for c in cases.values()};require(len(aliases)==204,'Distinct public-ready aliases')
 for folder,suffix in [('cases_PRIVATE','.npz'),('trace_PRIVATE','.json')]:require({p.stem for p in (direct/folder).glob('*'+suffix)}==aliases,'Complete direct case/trace coverage')
 for folder in ('cases_PRIVATE','scene_metadata_PRIVATE'):require({p.stem for p in (sem/folder).glob('*.json')}==aliases,'Complete semantic case/meta coverage')
 require({p.name for p in (sem/'calls_PRIVATE').glob('*.json')}=={alias+'__'+kind+'.json' for alias in aliases for kind in ('text','visual_order0','visual_order1')},'Exactly612 closed call records')
 reserved=read(R/'research/round15/input/UNUSED_B32_GPU_INPUTS_PRIVATE.json');require(not ({r['id'] for r in reserved}&set(cases)),'Reserved B absent from formalG')
 counters=collections.defaultdict(collections.Counter);semcounts=collections.Counter();branchstatuses=collections.Counter();ngeometry=0
 for row in rows:
  require(set(row)=={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'},'Strict9key input-only manifest')
  c=cases[row['id']];alias=c['case_alias'];require(c['platform']==row['platform'] and c['scene_PRIVATE']==row['scene'],'Manifest case join')
  src=resolve(c['source_dir']);cp=src/'source_cache'/(row['id']+'.npz');source_seal=captures[str(src.resolve())];require(sha(cp)==source_seal['files']['source_cache/'+row['id']+'.npz'],'Source cache member binding')
  with np.load(cp,allow_pickle=False) as z:cache={k:z[k].copy() for k in ('sample_id','boxes_raw6','text_scores','query_index','manifest_row_sha256')}
  require(str(cache['sample_id'])==row['id'] and cache['manifest_row_sha256'].tobytes().hex()==canonical(row),'Actual source text/sampling manifest exact identity')
  pool=finite_array(cache['boxes_raw6'],(1,256,6),np.float32)[0];scores=finite_array(cache['text_scores'],(1,256),np.float32)[0]
  with np.load(direct/'cases_PRIVATE'/(alias+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  require(str(d['sample_id'])==row['id'] and str(d['case_alias'])==alias and str(d['case_role'])==c['role'] and str(d['platform'])==row['platform'],'Direct identity/cohort fields')
  tr=read(direct/'trace_PRIVATE'/(alias+'.json'));payload=read(sem/'cases_PRIVATE'/(alias+'.json'));meta=read(sem/'scene_metadata_PRIVATE'/(alias+'.json'))
  require(payload['sample_id_PRIVATE']==row['id'] and payload['case_alias']==alias and payload['role']==c['role'],'Actual semantic identity')
  require(canonical(meta)==payload['scene_metadata_sha256']==str(d['scene_metadata_sha256'])==tr['scene_metadata_sha256'],'Canonical metadata bridge')
  require(str(d['semantic_case_sha256'])==sha(sem/'cases_PRIVATE'/(alias+'.json'))==tr['semantics_member_sha256'],'Sealed semantic case byte bridge')
  require(tr['bindings']==payload['bindings'] and str(d['source_cache_sha256'])==sha(cp)==tr['bindings']['source_cache_sha256'],'Direct/source/cache trace bridge')
  require(str(d['mask_cache_sha256'])==sha(resolve(c['mask_dir'])/(c['frame_alias']+'.npz'))==tr['bindings']['mask_cache_sha256'],'Actual selected mask producer byte bridge')
  expected=semantic_replay(sem,alias,row,meta,payload,semcounts)
  raw,aq=audit_arrays(d,pool,scores,tr);require(int(d['q_source'])==int(cache['query_index']),'Same source q0 cache')
  objects={o['mask_id']:o for o in meta['objects']}
  semantic_key={'proposal':'proposal_consensus','graph':'graph_match','relation_disabled':'graph_relation_disabled','relation_flip':'graph_relation_flip'}
  for bname,b in tr['geometry_trace']['branches'].items():
   mapping_readout_check(b,expected[semantic_key[bname]],meta,pool,scores)
   if b['association_accepted']:
    o=objects[b['selected_target_mask_id']];require(o['support_sha256']==b['diagnostics']['selected_support_sha256'] and o['mask_sha256']==b['diagnostics']['selected_mask_sha256'] and o['n_exclusive']==b['diagnostics']['selected_exclusive_support_n'],'Whole-scene selected support provenance')
    ngeometry+=1
   branchstatuses[bname+':'+str(b['diagnostics'].get('reason'))]+=1
  pop=c['role']+'|'+c['platform'];counts=counters[pop];counts['cases']+=1
  for j,name in enumerate(ARMS):
   counts[name+':query_changed']+=aq[j]!=int(d['q_source']);counts[name+':actual_changed']+=np.any(raw[j]!=raw[0]);counts[name+':geometry_increment']+=np.any(raw[j]!=pool[aq[j]])
   counts[name+':association_accepted']+=bool(d['arm_association_accepted'][j]);counts[name+':status:'+str(d['arm_status'][j])]+=1
  require(tr['geometry_trace']['all_no_parameter_update'] is True and tr['geometry_trace']['no_old_G_endpoint_used'] is True,'Direct-only/no legacy path')
 out={'state':'PASS_R15_FORMAL204_EIGHT_ARM_UNLABELLED_PROCESS_AND_INCREMENT_AUDIT_NOT_TASK_EFFECT',
  'prediction_seal_sha256':sha(direct/'SEALED.json'),'semantic_seal_sha256':sha(sem/'SEALED.json'),'plan_sha256':ds['plan_sha256'],'semantic_plan_sha256':ss['plan_sha256'],'manifest_sha256':sha(a.manifest),'case_index_sha256':sha(a.cases),'input_plan_sha256':sha(R/'research/round15/input/INPUT_PLAN_LOCK.json'),'auditor_sha256':sha(__file__),
  'checks':CHECKS,'n_samples':204,'n_arms':8,'n_semantic_calls':612,'capture_directories_verified':8,'capture_member_files_hashed':capture_files,'direct_member_files_hashed':len(ds['files']),'semantic_member_files_hashed':len(ss['files']),
  'counters':{k:{n:int(v) for n,v in z.items()} for k,z in counters.items()},'semantic_process_counts':dict(semcounts),'branch_status_counts':dict(branchstatuses),'accepted_branch_geometries_checked':ngeometry,
  'all_source_pool_q0_and_same_q1_increment_contracts_passed':True,'all204retained_no_acceptance_subset_gate':True,'reservedB_not_used_in_G':True,
  'center_size_bound_scope':'Fixed .25*anchorSize capped .5m; sizes source..1.2source; final f32 quantization permits only one outward ULP of rounded bound.',
  'no_GT_index_annotations_taskIoU_model_GPU_G_resolve_or_raw_sensor_decode':True,
  'quality_or_identity_or_progression_claimed':False,'asset_proof_scope':'Recorded11file digest declarations and frozen producer code/flags; no independent weight reload or inference.',
  'seconds':time.perf_counter()-tick,'issues':[]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(json_safe(out),indent=2,allow_nan=False)+'\n');print(json.dumps({'state':out['state'],'sha256':sha(a.output),'counts':out['counters']}))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','semantic','manifest','cases','output'):p.add_argument('--'+k,type=Path,required=True)
 for k in ('seal-sha256','semantic-seal-sha256'):p.add_argument('--'+k,required=True)
 main(p.parse_args())
