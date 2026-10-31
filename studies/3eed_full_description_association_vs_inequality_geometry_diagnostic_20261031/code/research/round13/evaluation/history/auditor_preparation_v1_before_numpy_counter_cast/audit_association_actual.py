"""R13 sealed-A audit. Preparation reads no in-progress outputs; CLI after release.
Replays frozen discrete decisions, not VLM/model/optimization/task geometry.
"""
from pathlib import Path
import argparse,collections,hashlib,json,re
import numpy as np
B=Path(__file__).resolve().parents[3]
ARMS=('source','full_order0_raw','full_two_numbering_consensus','noun_two_numbering_consensus','deranged_two_numbering_consensus')
QUERIES=('full','noun','deranged');LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def verify(folder):
 s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  p=Path(rel);assert not p.is_absolute() and '..' not in p.parts;z=(folder/p).resolve();assert z.is_relative_to(folder.resolve()) and sha(z)==h
 return s

def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if (a[3:]<=0).any() or (b[3:]<=0).any():return 0.
 v=np.maximum(np.minimum(a[:3]+a[3:]/2,b[:3]+b[3:]/2)-np.maximum(a[:3]-a[3:]/2,b[:3]-b[3:]/2),0).prod();u=a[3:].prod()+b[3:].prod()-v
 return float(v/u) if u>0 else 0.
def grouping(boxes,scores):
 groups=[];membership=np.full(256,-1,np.int64)
 for q in sorted(range(256),key=lambda q:(-float(scores[q]),q)):
  g=next((g for g in groups if iou(boxes[q],boxes[g['representative_query_index']])>=.5),None)
  if g is None:g={'group_id':len(groups),'representative_query_index':q,'query_indices':[],'representative_text_score':float(scores[q]),'representative_raw_size_positive':bool((boxes[q,3:]>0).all())};groups.append(g)
  g['query_indices'].append(q);membership[q]=g['group_id']
 return groups,membership

def parse(raw,candidates):
 text=str(raw).strip();valid={c['candidate_id'] for c in candidates}
 if text=='UNKNOWN':return {'parse_status':'explicit_UNKNOWN','candidate_id':None}
 if re.fullmatch(r'C\d{3}',text) and text in valid:return {'parse_status':'valid_candidate_ID','candidate_id':text}
 return {'parse_status':'invalid_or_unlisted_output','candidate_id':None}
def mapped(parsed,candidates,mapping,q0):
 c=next((c for c in candidates if c['candidate_id']==parsed['candidate_id']),None);m=next((m for m in mapping['masks'] if c is not None and m['mask_index']==c['mask_index']),None)
 ok=m is not None and m['mapping_status']=='mapped_unique_geometry_hypothesis';q=m['representative_query_index'] if ok else q0
 return q,c,m,'mapped_original_query' if ok else 'fallback_q0_'+(m['mapping_status'] if m is not None else parsed['parse_status'])
def main(a):
 assert sha(a.association/'SEALED.json')==a.expected_seal_sha256
 s=verify(a.association);p=read(a.plan);assert sha(a.plan)==s['plan_sha256']
 assert s['state']=='sealed_R13_all116_frozen_association6calls_before_GT' and s['n_samples']==116 and s['n_model_calls']==696 and s['n_queries']==3 and s['numberings']==2
 assert tuple(s['arm_names'])==ARMS and s['only_original_source_query_boxes_selected'] and s['model_frozen_parameters_buffers_unchanged'] and not s['GT_or_optimizer_or_neural_update_executed']
 assert p['stage']=='R13_frozen_full_description_association' and not p['GT_read_permitted'] and p['model_dtype']=='bfloat16' and p['attn_implementation']=='sdpa' and p['formal_preflight_policy']=='rerun_all116_in_separate_output'
 assert s['program_sha256']==p['program_sha256'] and s['cache_seals_sha256']==p['cache_seals_sha256'] and s['model_files_sha256']==p['model_files_sha256']
 for rel,h in p['program_sha256'].items():assert sha(B/rel)==h
 for path,h in p['cache_seals_sha256'].items():assert sha(B/path)==h;verify((B/path).parent)
 for key in ('manifest','case_index','frame_index'):assert sha(p[key])==p[key+'_sha256']==s[key+'_sha256']
 # Actual asset byte hashing, no weights/tensors loaded or interpreted.
 for rel,h in p['model_files_sha256'].items():assert sha(Path(p['model_dir'])/rel)==h
 rows=read(p['manifest']);cases=read(p['case_index']);cases={c['id']:c for c in cases};frames=read(p['frame_index']);assert len(rows)==len(cases)==116
 binding=read(a.association/'RUN_BINDINGS.json');assert binding['plan_sha256']==sha(a.plan) and binding['scope']=='formal116_rerun_separate_output_696calls' and binding['n_cases']==116 and binding['two_numberings']==2 and binding['queries']==list(QUERIES)
 assert binding['program_sha256']==p['program_sha256'] and binding['model_files_sha256']==p['model_files_sha256']
 receipt=read(a.association/'MODEL_LOAD_RECEIPT.json');assert receipt['local_only_safetensors_no_remote_code'] and receipt['frozen_eval_no_updates'] and receipt['modelweights_files_rehashed'] and receipt['formal_scope']==binding['scope'];assert not any(receipt['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
 actual={str(f.relative_to(a.association)) for f in a.association.rglob('*') if f.is_file()};assert actual==set(s['files'])|{'SEALED.json'}
 stats=collections.defaultdict(collections.Counter);mapping_stats=collections.defaultdict(collections.Counter);parse_stats=collections.defaultdict(collections.Counter);prompts=set();expectedcalls=set();tokens=[];times=[];allids={r['id']:r for r in rows};checks=0
 for row in rows:
  case=cases[row['id']];alias=case['case_alias'];tag=case['role']+'|'+row['platform'];t=read(a.association/'cases'/(alias+'.json'));stats[tag]['n']+=1
  assert t['id_PRIVATE']==row['id'] and t['role']==case['role'] and t['platform']==row['platform'] and t['scene_PRIVATE']==row['scene'] and t['no_3Dbox_regression_or_parameter_updates']
  cp=B/case['source_dir']/'source_cache'/(row['id']+'.npz');mp=B/case['mask_dir']/(case['frame_alias']+'.npz');assert sha(cp)==t['bindings']['source_cache_sha256'] and sha(mp)==t['bindings']['mask_cache_sha256']
  frame=frames[case['frame_alias']];assert t['bindings']['image_sha256']==frame['image_sha256'] and t['bindings']['point_sha256']==frame['point_sha256']
  assert sha(row['imagepath'])==frame['image_sha256'] and sha(row['pointpath'])==frame['point_sha256'] and sha(row['metapath'])==frame['metadata_sha256']
  with np.load(cp,allow_pickle=False) as z:pool=z['boxes_raw6'][0].copy();scores=z['text_scores'][0].copy();q0=int(z['query_index'])
  with np.load(a.association/'cases'/(alias+'.npz'),allow_pickle=False) as z:out={k:z[k].copy() for k in z.files}
  assert q0==int(scores.argmax())==t['q0']==int(out['source_q0']) and str(out['sample_id'])==row['id'];assert tuple(out['arm_names'])==ARMS and np.array_equal(out['source_boxes_raw6'],pool) and np.array_equal(out['source_text_scores256'],scores)
  with np.load(mp,allow_pickle=False) as z:masks=z['masks_bool'].copy();rects=z['boxes_xyxy'].copy()
  maskhash=[ah(m) for m in masks];mapping=t['mapping'];groups,membership=grouping(pool,scores);assert mapping['groups']==groups and np.array_equal(mapping['membership256'],membership) and mapping['source_q0']==q0 and mapping['source_group']==int(membership[q0]);checks+=4
  assert len(mapping['masks'])==len(masks)
  for j,m in enumerate(mapping['masks']):
   assert m['mask_index']==j and m['mask_sha256']==maskhash[j] and len(m['candidate_pairs'])==len(groups)
   eligible=[]
   for gid,v in enumerate(m['candidate_pairs']):
    assert v['group_id']==gid and v['representative_query_index']==groups[gid]['representative_query_index'];e=bool(v['projected_box_valid'] and v['projected_2D_iou']>=.1 and v['nearest_depth_visible_points_in_box_and_mask']>=3);assert v['eligible']==e
    if e:eligible.append(v)
   eligible.sort(key=lambda v:(-v['projected_2D_iou'],v['group_id']));assert len(eligible)==m['eligible_group_n'];best=eligible[0] if eligible else None;gap=best['projected_2D_iou']-eligible[1]['projected_2D_iou'] if len(eligible)>1 else None;good=best is not None and (gap is None or gap>=.05)
   status='mapped_unique_geometry_hypothesis' if good else 'NA_no_eligible_querygroup' if not eligible else 'NA_ambiguous_querygroup_2D_IoU_gap';assert status==m['mapping_status'];assert m['representative_query_index']==(best['representative_query_index'] if good else None);assert m['selected_group_id']==(best['group_id'] if good else None);assert m['query_group_indices']==(groups[best['group_id']]['query_indices'] if good else [])
   mapping_stats[tag][status]+=1;checks+=5+len(m['candidate_pairs'])*2
  order=sorted(range(len(masks)),key=lambda j:(hashlib.sha256((p['association_config']['candidate_order_salt']+'|'+row['id']+'|'+maskhash[j]).encode()).hexdigest(),j));cs=t['candidate_panels']
  for k,ix in enumerate((order,list(reversed(order)))):
   expected=[{'candidate_id':f'C{n+1:03d}','mask_index':j,'bbox_xyxy':rects[j].astype(float).tolist(),'mask_sha256':maskhash[j]} for n,j in enumerate(ix)];assert cs[k]==expected;checks+=1
  donor=allids[case['shuffled_donor_id_PRIVATE']];assert donor['caption']==case['shuffled_caption']!=row['caption'] and (donor['platform'],donor['scene'])==(row['platform'],row['scene'])
  nounmatch=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',row['caption'],re.I);noun=nounmatch.group(0) if nounmatch else 'object';queries={'full':row['caption'],'noun':noun,'deranged':case['shuffled_caption']};parsed={};cb={}
  for cond in QUERIES:
   parsed[cond]=[];cb[cond]=[]
   for k in (0,1):
    rel='calls/'+alias+'__'+cond+'__order'+str(k)+'.json';expectedcalls.add(rel);call=read(a.association/rel);v=call['result'];b=call['binding'];assert call['state']=='complete_frozen_VLM_single_call_no_GT_or_update';assert b['case_alias']==alias and b['case_id_PRIVATE']==row['id'] and b['condition']==cond and b['numbering']==k and b['plan_sha256']==sha(a.plan) and b['bindings']==t['bindings'];assert b['candidate_id_to_mask_index']=={c['candidate_id']:c['mask_index'] for c in cs[k]}
    h=hashlib.sha256(queries[cond].encode()).hexdigest();assert b['query_sha256']==v['query_sha256']==h;choice=parse(v['raw_answer_PRIVATE'],cs[k]);assert choice==v['parsed'] and v==t['calls'][cond][k];parsed[cond].append(choice);cb[cond].append(b);parse_stats[tag+'|'+cond][choice['parse_status']]+=1
    assert 0<v['input_token_n']<=8192 and v['max_new_tokens']==8 and 0<=v['generated_token_n']<=8;grid=v['image_grid_thw'];assert len(grid)==2 and all(g[0]==1 and 0<g[1]*g[2]//4<=1024 for g in grid);assert v['image_visualtoken_n']==[g[0]*g[1]*g[2]//4 for g in grid];tokens.append(v['input_token_n']);times.append(v['seconds_model_generate']);prompts.add(v['serialized_prompt_sha256']);stats[tag][cond+'|ended_with_eos']+=bool(v['ended_with_eos']);checks+=8
   assert t['calls'][cond][0]['serialized_prompt_sha256']==t['calls'][cond][1]['serialized_prompt_sha256'];checks+=1
  assert len({b['original_RGB_array_sha256'] for x in cb.values() for b in x})==1
  for k in (0,1):assert len({cb[cond][k]['panel_RGB_array_sha256'] for cond in QUERIES})==1
  selected=[q0]
  rawq,c,m,status=mapped(parsed['full'][0],cs[0],mapping,q0);saved=t['full_order0_raw'];assert saved['q_selected']==rawq and saved['readout_status']==status and np.array_equal(saved['box_raw6'],pool[rawq]);selected.append(rawq)
  for cond in QUERIES:
   chosen=[next((c for c in cs[k] if c['candidate_id']==parsed[cond][k]['candidate_id']),None) for k in (0,1)];same=all(c is not None for c in chosen) and chosen[0]['mask_index']==chosen[1]['mask_index'];pc=parsed[cond][0] if same else {'candidate_id':None,'parse_status':'two_numberings_disagree_or_UNKNOWN'};q,c,m,status=mapped(pc,cs[0],mapping,q0);saved=t['consensus'][cond]
   assert saved['q_selected']==q and saved['readout_status']==status and saved['two_numberings_same_physical_mask']==same and saved['selected_mask_index_by_order']==[c['mask_index'] if c else None for c in chosen] and np.array_equal(saved['box_raw6'],pool[q]);selected.append(q);stats[tag][cond+'|same_mask_consensus']+=same;stats[tag][cond+'|query_changed']+=q!=q0;stats[tag][cond+'|box_changed']+=not np.array_equal(pool[q],pool[q0]);stats[tag][cond+'|fallback']+=status!='mapped_original_query';checks+=5
  assert np.array_equal(out['arm_query_index'],selected) and np.array_equal(out['arm_boxes_raw6'],pool[selected]);checks+=2
 assert {str(f.relative_to(a.association)) for f in (a.association/'calls').glob('*.json')}==expectedcalls and len(expectedcalls)==696
 journal=[json.loads(x) for x in (a.association/'calls.jsonl').read_text().splitlines() if x];assert len(journal)==696 and {x['path'] for x in journal}==expectedcalls;assert all(sha(a.association/x['path'])==x['sha256'] for x in journal)
 engineering=None
 if a.engineering:
  es=verify(a.engineering);assert es['state']=='sealed_R13_frozen_association_first2_engineering_no_GT' and es['n_samples']==2 and es['n_model_calls']==12 and es['scope']=='engineering_inputonly_first2_12calls_not_formal116';assert es['model_files_sha256']==s['model_files_sha256'] and es['program_sha256']==s['program_sha256'];engineering={'seal_sha256':sha(a.engineering/'SEALED.json'),'calls':12,'separate_output_rerun_not_reused_as_formal_calls':True}
 result={'state':'PASS_R13_FORMAL116_ASSOCIATION_UNLABELLED_AUDIT','association_seal_sha256':sha(a.association/'SEALED.json'),'plan_sha256':sha(a.plan),'auditor_sha256':sha(Path(__file__)),'n_cases':116,'n_model_calls':696,'finite_discrete_checks':checks,'all_seal_members_program_assets_capture_input_bindings_verified':True,'replayed_same_mask_numbering_consensus_and_originalquerybox_readout':True,'dedup_and_mapping_decisions_scope':'Greedygroups replayed from savedsourcepool. Mapping eligibility/uniquechoice replayed from sealed projected-IoU/support metrics; not independent rawpoint/calibration reprojection or semantic identity validation.','parameter_state_scope':'Frozenrequiresgrad/eval/inference and parameter versions/bufferbytehash from producerproof; not a parameter-memory bytehash audit.','pixel_prompt_scope':'All image/panel files bytehashed; percall arrayhash consistency and exactliteral/noun/deranged querySHA checked. PIL image-array preparation and tokenizer prompt serialization not independently reexecuted.','counts':{k:dict(v) for k,v in stats.items()},'parse_status_histograms':{k:dict(v) for k,v in parse_stats.items()},'mapping_status_histograms':{k:dict(v) for k,v in mapping_stats.items()},'input_token_min_max':[min(tokens),max(tokens)],'generation_seconds_sum':sum(times),'distinct_serialized_prompt_hashes':len(prompts),'engineering':engineering,'no_GT_index_taskscore_VLM_reexecution_task_IoU_geometry_optimization_or_predictionchanges':True,'issues':[]}
 assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':checks,'counts':result['counts'],'sha256':sha(a.output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for x in ('association','plan','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--expected-seal-sha256',required=True);p.add_argument('--engineering',type=Path);main(p.parse_args())
