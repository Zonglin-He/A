"""Final prediction audit draft. Imports never open data; exact root grant first."""
import argparse,json,hashlib,time,datetime,re,collections,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
R=Path(__file__).resolve().parents[3]
MAIN=['source','R24_query_mlp_lr1_s16','R24_memory_alpha1','R30_beta1','R30_memory_E_only','R30_memory_allA_zeroU']
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
EXT=['source','Tent_BN_statistics_first_pre','Tent_clean_post_port','SAR_LN_first_pre','SAR_LN_clean_post_port','MEMO_BN_prior_only','MEMO_augmentation_only','MEMO_AdamW1e5_clean_post_port','Reg_inspired_teacher_pre','Reg_inspired_teacher_post','Reg_inspired_student_post','Reg_inspired_direct_pseudo']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def hardstop():assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,2,1,30,tzinfo=datetime.timezone.utc)
def verify(p):
 s=read(p/'SEALED.json')
 for rel,h in s['files'].items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(p/rel)==h,(p,rel)
 return s
def key(z):
 t=z['tokens'][z['token_mask']].mean(0);t/=max(float(np.linalg.norm(t)),1e-12);v=z['query']*t;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);return v
def libraries():
 A=read(R/'research/round24/input/D208_PRIVATE.json');fb=read(R/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');assert len(A)==208 and len(fb)==22;result={}
 for pl in ['drone','quad']:
  rr=[x for x in A if x['platform']==pl];zz=[npz(R/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(x['id']+'.npz')) for x in rr];K=np.stack([key(z) for z in zz]);v24=[];v30=np.zeros((104,256),np.float32);ct=np.zeros_like(v30);lookup={x['id']:i for i,x in enumerate(rr)}
  for x,z in zip(rr,zz):
   b=npz(R/'crossplatform/round23/bank272_attempt001/samples_PRIVATE'/(x['id']+'.npz'));f=z['scores'].astype(float)+2*b['vision']+2*b['position']+2*b['relation'];v24.append(f-z['scores'])
  ff=[f for f in fb if f['platform']==pl]
  for f in ff:
   i=lookup[f['id_PRIVATE']];P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];s=zz[i]['scores'];dv=max(.1-float(s[P].max()-s[N].max()),0)/2;v30[i,P]+=dv;v30[i,N]-=dv;ct[i,P]+=1;ct[i,N]+=1
  v30/=np.maximum(ct,1);E=np.array(sorted({lookup[f['id_PRIVATE']] for f in ff}));assert len(E)==(8 if pl=='drone' else 11);U=np.array([i for i in range(104) if i not in set(E)]);assert np.count_nonzero(v30[U])==0;result[pl]={'A_keys':K,'V24_dense_G_probability_increment':np.array(v24,np.float32),'V30_preference_probability_increment':v30,'E_indices':E,'U_indices':U}
 return result
def sample_check(r,z):
 p=Path(r['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);assert len(raw)>0 and raw.ndim==2 and raw.shape[1]>=3 and np.isfinite(raw[:,:3]).all();raw=np.asarray(raw,np.float32);ix=np.random.default_rng(r['seed']).choice(len(raw),16384,replace=len(raw)<16384).astype(np.int64);xyz=raw[ix,:3].copy()
 if r['platform']=='drone':xyz[:,2]+=1.8
 xyz=np.ascontiguousarray(xyz,np.float32);assert np.array_equal(ix,z['point_indices']) and hashlib.sha256(xyz.tobytes()).hexdigest()==str(z['point_sample_sha256'])
def head_scores(z,state):
 h=z['h'].astype(float)
 for j in [0,2,4]:
  h=h@state[str(j)+'.weight'].astype(float).T+state[str(j)+'.bias'].astype(float)
  if j<4:h=np.maximum(h,0)
 h/=np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12);return (softmax(h@z['tokens'].astype(float).T/.07,axis=-1)*z['token_mask']).sum(-1)
def main():
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);a.add_argument('--predictions',required=True);a.add_argument('--authorization',required=True);a.add_argument('--code-lock',required=True);a.add_argument('--out',required=True);arg=a.parse_args();hardstop()
 # No final manifest, input, prediction, media or XYZ read before this check.
 grant=read(arg.authorization);assert grant['independent_CPU_audit_authorized'] is True and grant['independent_code_lock_sha256']==sha(arg.code_lock)
 locked=read(arg.code_lock)
 for rel,h in {**locked['program_sha256'],**locked['fixed_provenance_sha256']}.items():assert sha(R/rel)==h,rel
 assert grant['plan_sha256']==sha(arg.plan) and grant['prediction_seal_sha256']==sha(Path(arg.predictions)/'SEALED.json');plan=read(arg.plan);assert plan['execution_mode']=='final_once' and grant['manifest_sha256']==plan['manifest_sha256'];assert sha(plan['manifest_path'])==plan['manifest_sha256'];rows=read(plan['manifest_path']);assert len(rows)==plan['expected_n']==384;B=Path(arg.predictions);O=Path(arg.out);O.mkdir(exist_ok=False,parents=True);start=time.perf_counter();seal=verify(B)
 assert seal['plan_sha256']==sha(arg.plan)
 for rel,h in {**plan['program_sha256'],**plan['old_inputs_sha256']}.items():assert sha(R/rel)==h,rel
 for n in ['source','frames','main','external','oldG/textM64_attempt001','oldG/preparedM64_attempt001','oldG/dinoM64_attempt001','oldG/evidenceM64_attempt001','oldG/bankM64_attempt001']:verify(B/n)
 import torch
 from transformers import RobertaTokenizerFast
 torch.set_num_threads(4);tok=RobertaTokenizerFast.from_pretrained(R/'crossplatform/upstream/data/roberta_base',local_files_only=True);libs=libraries();states={(pl,arm):npz(R/plan[field].format(platform=pl)) for pl in ['drone','quad'] for arm,field in [('R24_query_mlp_lr1_s16','R24_state_template'),('R30_beta1','R30_state_template')]};closure=read(B/'source/CLOSURE_AUDIT.json');assert closure['outside_projection_sha256_before']==closure['outside_projection_sha256_after'] and closure['full_source_fingerprint_before']==closure['full_source_fingerprint_after'] and closure['full_model_forwards']==384*3
 bindings={x['id_PRIVATE']:x for x in read(B/'frames/FRAME_BINDINGS_PRIVATE.json')};assert set(bindings)=={x['id'] for x in rows};frame_seal=read(B/'frames/SEALED.json');assert frame_seal['n']==frame_seal['model_calls']==384 and frame_seal['model_unchanged'] and frame_seal['weights_sha256']==read(R/'research/round12/experiment/MASK_CAPTURE_PLAN_LOCK.json')['weights_sha256'];assert plan['fresh_frames'] and plan['max_MaskRCNN_calls']==384
 for pl,expected in libs.items():
  saved=npz(B/'main/memory_PRIVATE'/(pl+'_library.npz'))
  for k,v in expected.items():assert np.array_equal(v,saved[k]),(pl,k)
 records=[];headrows=[];memoryrows=[];similarity_cells=0;max_head_error=0.;head_qdiff=0;memory_ties=0;evidence_outputs=[]
 for r in rows:
  hardstop();sid=r['id'];pl=r['platform'];z=npz(B/'source/samples_PRIVATE'/(sid+'.npz'));m=npz(B/'main/predictions_PRIVATE'/(sid+'.npz'));names=MAIN+[x+'_plus_current_oldG' for x in MAIN];assert m['arm_names'].tolist()==names and m['scores'].shape==(12,256) and np.isfinite(m['scores']).all();assert np.array_equal(m['queries'],m['scores'].argmax(-1)) and np.array_equal(m['boxes'],z['boxes'][m['queries']]);assert np.array_equal(m['scores'][0],z['scores']);sample_check(r,z)
  bind=bindings[sid];assert all(sha(r[path])==bind[field] for path,field in [('imagepath','image_sha256'),('pointpath','point_sha256'),('metapath','metadata_sha256')]);mp=B/'frames/masks_PRIVATE'/(sid+'.npz');assert sha(mp)==bind['mask_cache_sha256'];ins=npz(mp);N=len(ins['scores']);hw=tuple(ins['original_hw']);assert ins['masks_bool'].shape==(N,*hw) and ins['masks_bool'].dtype==np.bool_ and ins['boxes_xyxy'].shape==(N,4) and np.isfinite(ins['boxes_xyxy']).all() and np.all(ins['scores']>=.5) and np.all((ins['class_ids']>0)&(ins['class_ids']<91)) and np.all(np.diff(ins['instance_indices'])>0);attempt=read(B/'frames/attempts_PRIVATE'/(sid+'.json'));assert attempt['status']=='completed' and attempt['model_forward_entered'];camera=read(B/'frames/camera_PRIVATE'/(sid+'.json'));assert {'image_intrinsic','image_extrinsic'}<=set(camera)<={'image_intrinsic','image_extrinsic','camera_distortion'} and np.shape(camera['image_intrinsic']) in [(3,3),(3,4)] and np.shape(camera['image_extrinsic'])==(4,4)
  off=np.array(tok([r['caption']+' . not mentioned'],return_offsets_mapping=True)['offset_mapping'][0]);match=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',r['caption']+' . not mentioned',re.I);mask=(off[:,1]>match.start())&(off[:,0]<match.end()) if match else off[:,1]>off[:,0];assert len(off)==len(z['tokens']) and np.array_equal(mask,z['token_mask']);source_cpu=(softmax(z['query'].astype(float)@z['tokens'].astype(float).T/.07,axis=-1)*mask).sum(-1);assert abs(source_cpu-z['scores']).max()<2e-5
  for arm in ['R24_query_mlp_lr1_s16','R30_beta1']:
   v=head_scores(z,states[pl,arm]);actual=m['scores'][names.index(arm)];err=float(abs(v-actual).max());assert err<2e-5;different=int(v.argmax())!=int(actual.argmax());head_qdiff+=different;max_head_error=max(max_head_error,err);gap=float(np.sort(actual)[-1]-np.sort(actual)[-2]);assert not different or gap<=2*err+1e-12;headrows.append({'alias':r.get('alias',sid),'platform':pl,'arm':arm,'max_CPU64_probability_error':err,'CPU64_native_argmax_diff':different,'actual_native_top_tie_count':int((actual==actual.max()).sum()),'actual_native_top_gap':gap,'actual_GPU_argmax_preserved':True})
  bank=npz(B/'oldG/bankM64_attempt001/samples_PRIVATE'/(sid+'.npz'));FG=z['scores'].astype(float)+2*bank['vision']+2*bank['position']+2*bank['relation'];delta=FG-z['scores'].astype(float);assert np.array_equal(FG,m['oldG_F64']) and np.array_equal(delta,m['shared_deltaG_F64']) and np.array_equal(m['scores'][names.index('source_plus_current_oldG')],FG)
  prep=read(B/'oldG/preparedM64_attempt001/cases_PRIVATE'/(sid+'.json'));pa=npz(B/'oldG/preparedM64_attempt001/arrays_PRIVATE'/(sid+'.npz'));ev=read(B/'oldG/evidenceM64_attempt001/cases_PRIVATE'/(sid+'.json'));pool=prep['mask_pool_PRIVATE'];vv=np.zeros(256)
  if pool:
   e=np.array([np.mean([x['visual_evidence'] for x in ev['targets_PRIVATE'][str(mi)]]) for mi in pool]);w=pa['weights'];raw=w@e;mapped=w.sum(1)>0;q0=int(z['scores'].argmax());anchor=raw[q0] if mapped[q0] else float(np.median(e));vv[mapped]=np.clip((raw[mapped]-anchor)/2,-2,2)
  assert np.array_equal(vv,bank['vision'])
  def evidence_walk(x):
   if isinstance(x,dict):
    if {'log_odds_yes_no','image_sha256','prompt_sha256','input_tokens'}<=set(x):evidence_outputs.append((x['image_sha256'],x['prompt_sha256'],x['input_tokens'],x['log_odds_yes_no']));return
    for v in x.values():evidence_walk(v)
   elif isinstance(x,list):
    for v in x:evidence_walk(v)
  evidence_walk(ev)
  lib=libs[pl]
  for arm,field,ix in [('R24_memory_alpha1','V24_dense_G_probability_increment',np.arange(104)),('R30_memory_E_only','V30_preference_probability_increment',lib['E_indices']),('R30_memory_allA_zeroU','V30_preference_probability_increment',np.arange(104))]:
   saved=npz(B/'main/memory_PRIVATE'/(sid+'_'+arm+'.npz'));sim=torch.einsum('qc,anc->qan',torch.from_numpy(key(z)),torch.from_numpy(lib['A_keys'][ix]));similarity_cells+=sim.numel();near,which=sim.max(-1);best,obs=near.topk(min(4,len(ix)),-1);reported=np.array(saved['A_row_indices']);lookup={int(g):i for i,g in enumerate(ix)};ri=np.vectorize(lookup.__getitem__)(reported);rq=np.array(saved['A_query_indices']);rr=torch.from_numpy(ri);qq=torch.from_numpy(rq);qindex=torch.arange(256)[:,None];actual_similarity=sim[qindex,rr,qq];assert np.array_equal(actual_similarity.numpy(),saved['cosine']);assert torch.equal(actual_similarity,near.gather(1,rr));cut=best[:,-1:];assert torch.all(actual_similarity>=cut);assert all(len(set(row))==len(row) for row in ri);assert torch.all(actual_similarity[:,:-1]>=actual_similarity[:,1:]);boundary=(near==cut).sum(1)>1;memory_ties+=int(boundary.sum());weight=((actual_similarity-1)/.1).exp();V=torch.from_numpy(lib[field][ix]);corr=(V[rr,qq]*weight).sum(-1)/weight.sum(-1).clamp_min(1e-12);assert np.array_equal(weight.numpy(),saved['weights']) and np.array_equal(corr.numpy(),saved['correction_probability_units']) and np.array_equal(m['scores'][names.index(arm)],(z['scores']+corr.numpy()).astype(float));memoryrows.append({'alias':r.get('alias',sid),'platform':pl,'arm':arm,'all256_legal_nearest_queries_top4_distinct_observations':True,'boundary_tie_queries':int(boundary.sum()),'actual_indices_used_not_replaced':True,'weights_correction_native_probability_units_exact':True})
  for arm in MAIN[1:]:assert np.array_equal(m['scores'][names.index(arm+'_plus_current_oldG')],m['scores'][names.index(arm)]+delta)
  ex=npz(B/'external/predictions_PRIVATE'/(sid+'.npz'));assert ex['arm_names'].tolist()==EXT and ex['scores'].shape==(12,256) and ex['boxes'].shape==(12,6) and np.isfinite(ex['scores']).all() and np.isfinite(ex['boxes']).all();assert ex['box_pools'].shape==(12,256,6) and np.isfinite(ex['box_pools']).all();assert ex['query_override'].shape==ex['box_override'].shape==(12,);assert np.flatnonzero(ex['query_override']).tolist()==[EXT.index('MEMO_augmentation_only')] and np.flatnonzero(ex['box_override']).tolist()==[EXT.index('Reg_inspired_direct_pseudo')]
  for j,arm in enumerate(EXT):
   if not ex['query_override'][j]:assert int(ex['queries'][j])==int(ex['scores'][j].argmax())
   if not ex['box_override'][j]:assert np.array_equal(ex['boxes'][j],ex['box_pools'][j,int(ex['queries'][j])])
  for arm in ['source','SAR_LN_first_pre','Reg_inspired_teacher_pre']:assert np.array_equal(ex['scores'][EXT.index(arm)],z['scores'])
  assert np.array_equal(ex['box_pools'][0],z['boxes']) and np.array_equal(ex['source_pool'],z['boxes']) and np.array_equal(ex['source_native_probability'],z['scores']);records.append({'alias':r.get('alias',sid),'id_PRIVATE':sid,'platform':pl,'point_sample_and_token_mask_exact':True,'main12_queries_original_boxes_exact':True,'sharedG_source_literalFG_and_all6_sameDelta_exact':True,'G_mask_pool_size':len(pool),'G_nonzero_delta_queries':int(np.count_nonzero(delta)),'G_direct_changed_source_query':int(FG.argmax())!=int(z['scores'].argmax()),'external12_each_actual_box_pool_and_declared_overrides_exact':True});dump(O/'AUDIT_PROGRESS_PRIVATE.json',{'completed':len(records),'expected':384})
 calls=sorted((B/'oldG/evidenceM64_attempt001/numeric_calls_PRIVATE').glob('*.json'));numeric=[]
 for j,f in enumerate(calls,1):
  x=read(f);assert x['attempt']==j and x['status']=='completed' and x['model_forward_entered'] and x['generated_tokens']==0;assert float(np.float32(x['yes_logsumexp'])-np.float32(x['no_logsumexp']))==x['raw_yes_no_logodds']==x['log_odds_yes_no'];numeric.append((x['image_sha256'],x['prompt_sha256'],x['input_tokens'],x['log_odds_yes_no']))
 assert collections.Counter(numeric)==collections.Counter(evidence_outputs) and len(calls)<=plan['max_numeric_Qwen_calls'];dino=sorted((B/'oldG/dinoM64_attempt001/calls_PRIVATE').glob('*.json'));assert len(dino)<=plan['max_DINO_calls']
 for j,f in enumerate(dino,1):
  x=read(f);raw=npz(f.with_name(f.stem+'_raw.npz'));assert x['attempt']==j and x['status']=='completed' and x['model_forward_entered'] and all(np.isfinite(v).all() for v in raw.values())
 traces=read(B/'external/TRACE_PRIVATE.json');assert len(traces)==384;steps=0;counts={'Tent':42816,'SAR_LN_only':24768,'MEMO_Transformer_AdamW':144527419,'Reg_inspired':1734};skip=collections.Counter()
 for x in traces:
  assert x['source_bridge_exact'] and set(x['diagnostics'])==set(counts)
  for method,d in x['diagnostics'].items():assert d['steps'] in [0,1] and d['selected_parameters']==counts[method] and d['nonselected_parameters_unchanged'];steps+=d['steps'];skip[method]+=int(d['steps']==0)
 assert steps<=4*384 and read(B/'external/SEALED.json')['source_unchanged'];result={'state':'PASS_R46_INDEPENDENT_FINAL_PREDICTION_CONTRACT','n':384,'main_arms':12,'external_protocol_controls':12,'actual_full_model_source_head_forwards':1152,'fresh_MaskRCNN_calls':384,'fresh_all384_image_point_metadata_mask_bindings_complete':True,'fresh_camera_allowlist_only_and_mask_schema_nativeorder_checked':True,'head_CPU64_maxerror':max_head_error,'head_CPU64_vs_actual_native_argmax_differences':head_qdiff,'actual_native_GPU_argmax_preserved':True,'memory_similarity_cells_recomputed_CPU32':similarity_cells,'memory_boundary_tie_query_counts':memory_ties,'numeric_Qwen_calls':len(calls),'nonnumeric_VLM_calls':0,'DINO_calls':len(dino),'external_logical_steps':steps,'external_skips':dict(skip),'all_external_full_box_pools_and_explicit_override_contracts_checked':True,'G_boundary':'Raw numeric evidence, vision residual and shared readout arithmetic independently recomputed; positional/reference geometry engine remains pinned implementation with sealed bindings, not a semantic correctness certificate.','external_boundary':'No independent rerun of backbone or optimizer; complete sealed outputs/pools and static pinned implementation/parameter restoration receipts audited.','prediction_seal_sha256':sha(B/'SEALED.json'),'plan_sha256':sha(arg.plan),'manifest_sha256':plan['manifest_sha256'],'audit_authorization_sha256':sha(arg.authorization),'no_GT_taskscore_model_optimizer_or_method_change':True,'seconds':time.perf_counter()-start};dump(O/'ALL384_CONTRACT_ROWS_PRIVATE.json',records);dump(O/'ALL768_HEAD_ARITHMETIC_SCALARS_PRIVATE.json',headrows);dump(O/'ALL1152_MEMORY_ARITHMETIC_SCALARS_PRIVATE.json',memoryrows);dump(O/'AUDIT_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
