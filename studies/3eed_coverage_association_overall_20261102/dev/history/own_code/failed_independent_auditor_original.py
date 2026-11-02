"""Independent sealed seven-readout replay; no author R50 algorithm import/GT."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round50/experiment'
PRED=ROOT/'crossplatform/round50/predicted336_attempt001';PREP=ROOT/'crossplatform/round50/prepared336_attempt001';VIS=ROOT/'crossplatform/round50/visual336_attempt001';OLD=ROOT/'crossplatform/round49/stage336_attempt001';OUT=HERE/'prediction_audit_attempt001'
from audit_prepared50 import exact,oldw,oldread
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(folder):
 ss=read(folder/'SEALED.json')
 for rel,h in ss['files'].items():assert sha(folder/rel)==h
 return ss
def conditional(s,a,rho,pool,e,pr):
 active=rho>0;cue=a@e[pool];rel=a@pr[pool];q0=int(s.argmax());cv=np.zeros(256);cp=np.zeros(256);cr=np.zeros(256);av=ar=None
 if len(pool):
  av=cue[q0] if active[q0] else float(np.median(e[pool]));ar=rel[q0] if active[q0] else np.median(pr[pool],axis=0)
  cv[active]=np.clip((cue[active]-av)/2,-2,2);cp[active]=np.clip(rel[active,0]-ar[0],-2,2);cr[active]=np.clip(rel[active,1]-ar[1],-2,2)
 v=rho*cv;p=rho*cp;r=rho*cr
 assert (v[~active]==0).all() and (p[~active]==0).all() and (r[~active]==0).all()
 return {'scores':s.astype(float)+2*v+2*p+2*r,'vision':v,'position':p,'relation':r,'conditional_vision_clipped':cv,'conditional_position_clipped':cp,'conditional_relation_clipped':cr,'mapped':active,'rho':rho,'anchor_v':av,'anchor_PR':ar,'q0_mapped':bool(active[q0])}
def view_means(views):
 assert len(views)==2;d=[];f=[];g=[]
 for view in views:
  for key in ('full','gray'):
   q=view[key];assert q['generated_tokens']==0 and type(q['input_tokens']) is int and 0<q['input_tokens']<=8192 and type(q['log_odds_yes_no']) in (int,float) and np.isfinite(q['log_odds_yes_no'])
   for k in ('image_sha256','prompt_sha256'):assert isinstance(q[k],str) and len(q[k])==64 and all(c in '0123456789abcdef' for c in q[k])
  delta=view['full']['log_odds_yes_no']-view['gray']['log_odds_yes_no'];assert type(view['visual_evidence']) in (int,float) and delta==view['visual_evidence'];d.append(delta);f.append(view['full']['log_odds_yes_no']);g.append(view['gray']['log_odds_yes_no'])
 return float(np.mean(d)),float(np.mean(f)),float(np.mean(g))
def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_INDEPENDENT_PREDICTION_AUDIT_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_SEVEN_PREDICTION_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__)
 for rel,h in auth['references_sha256'].items():assert sha(ROOT/rel)==h
 assert auth['prediction_seal_sha256']==sha(PRED/'SEALED.json') and auth['visual_seal_sha256']==sha(VIS/'SEALED.json') and auth['prepared_seal_sha256']==sha(PREP/'SEALED.json') and auth['prep_audit_sha256']==sha(HERE/'prepared_audit_attempt001/AUDIT_RESULT.json')
 assert read(HERE/'prepared_audit_attempt001/AUDIT_RESULT.json')['state']=='PASS_R50_INDEPENDENT_PREP_GEOMETRY_ENTRY_ASSOCIATION_QUALITY_NO_GT'
 plan=read(EXP/'PREDICT_PLAN_LOCK.json');assert auth['prediction_plan_sha256']==sha(EXP/'PREDICT_PLAN_LOCK.json')
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 ps=verify(PRED);vs=verify(VIS);verify(PREP);assert ps['arms']==ARMS and ps['n']==vs['n']==336
 inv=ROOT/'research/round49/inventory_attempt001';verify(inv);legacybindings=read(inv/'SOURCE_BINDINGS_PRIVATE.json')
 bindings=read(PRED/'BINDINGS_PRIVATE.json')
 for rel,h in bindings.items():
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==h
  if rel in legacybindings:assert h==legacybindings[rel]
 assert not OUT.exists();OUT.mkdir(parents=True);rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');stats=[];newmasks=oldmasks=0;expected_calls=set();prompt_checks=0
 try:
  for i,row in enumerate(rows):
   rid=row['id'];a=load(PREP/'arrays_PRIVATE'/(rid+'.npz'));pc=read(PREP/'cases_PRIVATE'/(rid+'.json'));z=load(PRED/'samples_PRIVATE'/(rid+'.npz'));case=read(PRED/'cases_PRIVATE'/(rid+'.json'));v=read(VIS/'cases_PRIVATE'/(rid+'.json'));old=load(OLD/'samples_PRIVATE'/(rid+'.npz'))
   assert v['id_PRIVATE']==case['id_PRIVATE']==pc['id_PRIVATE']==rid and v['prep_array_sha256']==sha(PREP/'arrays_PRIVATE'/(rid+'.npz')) and v['prep_case_sha256']==sha(PREP/'cases_PRIVATE'/(rid+'.json'))
   assert v['actual_subject_query_sha256']==pc['actual_subject_query_sha256'] and v['wide_pool_PRIVATE']==a['wide_pool'].tolist()
   base=ROOT/'crossplatform'/('round24' if row['stage']=='M' else 'round23');suffix='M64' if row['stage']=='M' else '272';ev=read(base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'));pr=read(base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'));text=read(ROOT/pc['actual_text_source_PRIVATE'])
   assert v['image_sha256']==pr['bindings']['image_sha256'] and v['mask_cache_sha256']==pr['bindings']['mask_cache_sha256'] and hashlib.sha256(text['subject_query'].encode()).hexdigest()==pc['actual_subject_query_sha256']
   e=a['image_e'].copy();full=a['image_full'].copy();gray=a['image_gray'].copy();wide=a['wide_pool'];pool=a['old_pool'];assert set(v['targets_PRIVATE'])=={str(m) for m in wide} and set(v['mask_sha256_by_index_PRIVATE'])=={str(m) for m in wide}
   oldcase=read(OLD/'cases_PRIVATE'/(rid+'.json'));mp=ROOT/oldcase['mask_cache_path_PRIVATE'];assert sha(mp)==oldcase['mask_cache_sha256']
   with np.load(mp,allow_pickle=False) as masks:H,W=map(int,masks['original_hw'])
   promptset=set()
   for m in wide:
    key=str(m);assert v['mask_sha256_by_index_PRIVATE'][key]==str(a['mask_sha256'][m]);views=v['targets_PRIVATE'][key]
    if key in ev['targets_PRIVATE']:
     for before,after in zip(ev['targets_PRIVATE'][key],views):
      for k in ('full','gray','visual_evidence'):assert before[k]==after[k]
     oldmasks+=1
    else:
     newmasks+=1
     for view_index,view in enumerate(views):
      padding=0. if view_index==0 else .2;x1,y1,x2,y2=map(float,a['detector_boxes'][m]);ww=x2-x1;hh=y2-y1
      left=max(0,int(x1-padding*ww));top=max(0,int(y1-padding*hh));right=min(W,int(x2+padding*ww));bottom=min(H,int(y2+padding*hh));size=[right-left,bottom-top];assert min(size)>0
      for mode in ('full','gray'):
       name=f'{rid}__m{int(m)}__v{view_index}__{mode}.json';assert name not in expected_calls;expected_calls.add(name);record=read(VIS/'calls_PRIVATE'/name);assert record['state']=='COMPLETED' and record['value']==view[mode]
       bb=record['binding'];assert bb['id_PRIVATE']==rid and bb['mask_index_PRIVATE']==int(m) and bb['mask_sha256']==str(a['mask_sha256'][m]) and bb['crop_view']==view_index and bb['mode']==mode and bb['image_size']==size and bb['image_sha256']==view[mode]['image_sha256'] and bb['prompt_sha256']==view[mode]['prompt_sha256'] and bb['plan_sha256']==vs['plan_sha256']
       if mode=='gray':assert hashlib.sha256(np.full((size[1],size[0],3),127,np.uint8).tobytes()).hexdigest()==bb['image_sha256']
       prompt_checks+=1
    for view in views:
     promptset.update(view[mode]['prompt_sha256'] for mode in ('full','gray'))
    e[m],full[m],gray[m]=view_means(views)
   assert len(promptset)<=1
   for key,val in [('image_e',e),('image_full',full),('image_gray',gray),('source_boxes',a['source_boxes']),('source_scores',a['source_scores']),('old_pool',pool),('wide_pool',wide),('raw_mask_PR',a['raw_mask_PR'])]:exact(z[key],val,key)
   assert np.isfinite(e[wide]).all();s=a['source_scores'];boxes=a['source_boxes'];rawpr=a['raw_mask_PR'];S=a['legacy_support'];C=a['legacy_coverage'];I=a['legacy_projection_iou'];ow,_,_=oldw(pool,S,C,I,boxes);ww,_,_=oldw(wide,S,C,I,boxes)
   states=[{'scores':s.astype(float),'vision':np.zeros(256),'position':np.zeros(256),'relation':np.zeros(256),'mapped':np.zeros(256,bool)},oldread(s,ow,pool,e,rawpr),oldread(s,ww,wide,e,rawpr)];states[1]['mapped']=ow.sum(1)>0;states[2]['mapped']=ww.sum(1)>0
   for k in (3,10):
    for tag,pp in [('old',pool),('wide',wide)]:
     aa=a[f'{tag}_k{k}_conditional_association'];rho=a[f'{tag}_k{k}_rho'];states.append(conditional(s,aa,rho,pp,e,rawpr))
     for f in ('conditional_association','rho','unknown_mass','unknown_prior_mass','out_of_pool_foreground_mass','unassigned_mass','ambiguous_mass'):exact(z[f'{tag}_k{k}_{f}'],a[f'{tag}_k{k}_{f}'],f)
   scores=np.stack([st['scores'] for st in states]);queries=np.argmax(scores,axis=1).astype(np.int64);exact(scores,z['scores'],'seven_scores');exact(queries,z['queries'],'seven_queries');exact(boxes[queries],z['selected_original_boxes'],'seven_original_boxes');assert z['arm_names'].tolist()==ARMS
   for key in ('vision','position','relation','mapped'):exact(z[key],np.stack([st[key] for st in states]),key)
   for key in ('rho','conditional_vision_clipped','conditional_position_clipped','conditional_relation_clipped'):exact(z[key],np.stack([st[key] for st in states[3:]]),key)
   exact(scores[1],old['arm_scores'][7],'R49oldG');assert queries[1]==old['queries'][7]
   for j,st in enumerate(states[3:]):
    an=case['new_anchor'][j];assert an['arm']==ARMS[j+3] and an['q0_mapped']==st['q0_mapped'] and an['fallback_original_maskmedian']==bool(len(pool if j in (0,2) else wide) and not st['q0_mapped']) and an['anchor_v']==st['anchor_v']
    assert (an['anchor_PR'] is None and st['anchor_PR'] is None) or np.array_equal(an['anchor_PR'],st['anchor_PR'])
   stats.append({'row':f'D{i+1:04d}','platform':row['platform'],'newmask_count':len(wide)-len(pool),'seven_readouts_exact':True})
 except Exception as exc:
  dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':len(stats),'no_task_GT_read':True});raise
 assert len(stats)==336 and newmasks==1239 and len(expected_calls)==4956 and {p.name for p in (VIS/'calls_PRIVATE').glob('*.json')}==expected_calls
 vc=read(VIS/'COUNTS.json');assert vc['total_new_input_numeric_calls']==4956 and vc['new_numeric_calls']+vc['resumed_numeric_calls']==4956 and vc['new_reference_calls']==0 and vc['optimizer_steps']==0 and vc['generated_tokens']==0 and vc['model_unchanged'] is True
 dump(OUT/'ANONYMOUS_ROWS.json',stats);res={'state':'PASS_R50_INDEPENDENT_SEVEN_READOUT_SOURCE_AND_EVIDENCE_AUDIT_NO_GT','n':336,'arms':7,'full_score_values_bitexact':336*7*256,'selected_query_box_values_bitexact':336*7,'old_scored_mask_records_reused_exact':oldmasks,'new_scored_masks':newmasks,'unique_completed_atomic_numeric_call_records':len(expected_calls),'numeric_call_binding_size_prompt_and_gray_content_checks':prompt_checks,'incomplete_atomic_records':0,'original_two_view_full_minus_gray_arithmetic_exact':True,'oldG_and_source_endpoint_bitexact':True,'conditional_rho_afterclip_and_unknown_parts_exact':True,'new_model_forward_not_recomputed_or_semantic_identity_certified':True,'new_full_RGB_crop_hash_not_independently_reconstructed_from_pixels':True,'no_GT_RGBpixels_rawXYZ_models_or_protected_FINAL_read':True,'prediction_seal_sha256':sha(PRED/'SEALED.json'),'visual_seal_sha256':sha(VIS/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',res);print(json.dumps(res))
if __name__=='__main__':main()
