"""Root-authorized post-seal CPU replay and624task values. No feedback change."""
import sys,json,hashlib,time,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
P=ROOT/'crossplatform/round44/direct208_attempt001';EXP=ROOT/'research/round44/phase1';ARMS=['source','unrestricted_oldG','eligible_fixed_source_rep']
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(folder):
 s=read(folder/'SEALED.json')
 for n,h in s['files'].items():assert sha(folder/n)==h,(folder,n)
 return s
def independent_decision(s,F,groups):
 q0=int(np.argmax(s));qg=int(np.argmax(F));g0=[(str(m),g) for m,g in groups.items() if q0 in g];gg=[(str(m),g) for m,g in groups.items() if qg in g];m0=[m for m,g in g0];mg=[m for m,g in gg];S=g0[0][1] if len(g0)==1 else [];G=gg[0][1] if len(gg)==1 else [];r0=sorted(S,key=lambda j:(-float(s[j]),j))[0] if S else None;rg=sorted(G,key=lambda j:(-float(s[j]),j))[0] if G else None
 reasons=['no_corrective_nomination_same_q' if qg==q0 else None,'non_strict_G_tie' if not F[qg]>F[q0] else None,'source_query_not_exactly_one_full_group' if len(g0)!=1 else None,'G_query_not_exactly_one_full_group' if len(gg)!=1 else None,'same_instance_group_no_negative' if len(g0)==len(gg)==1 and m0[0]==mg[0] else None,'overlapping_group_members_no_negative' if set(S)&set(G) else None];why=next((x for x in reasons if x),'eligible_distinct_unique_groups');eligible=why=='eligible_distinct_unique_groups'
 return {'source_q0':q0,'raw_G_q':qg,'G_F64_gap_vs_q0':float(F[qg]-F[q0]),'source_membership_masks':m0,'G_membership_masks':mg,'source_group_members':S,'G_group_members':G,'source_group_rep':r0,'G_group_rep':rg,'eligible':eligible,'reason':why,'query':int(rg) if eligible else q0,'accepted_preference':{'positive_queries':G,'negative_queries':S,'positive_mask':mg[0],'negative_mask':m0[0],'positive_source_rep':rg,'negative_source_rep':r0} if eligible else None,'distinct_mask_hypothesis_not_gold_identity':True,'unknown_not_labelled_negative':not eligible}
def main():
 start=time.perf_counter();seal=verify(P);plan=read(EXP/'PLAN_LOCK.json');auth=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_PHASE1_DIRECT_USE_ONLY' and plan['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json') and seal['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for n,h in {**plan['program_sha256'],**plan['input_sha256'],**auth['refs_sha256']}.items():assert sha(ROOT/n)==h,n
 gate=read(ROOT/'research/round44/qualification/RESULT.json');assert gate['state']=='PASS_EXACT_SOURCE_ARRAY_COMPATIBILITY_ONLY' and gate['boxes_exact']==gate['scores_exact']==gate['n']==208
 locations={'source':ROOT/'crossplatform/round24/capture336_attempt001','bank':ROOT/'crossplatform/round23/bank272_attempt001','geometry':ROOT/'crossplatform/round37/prepared208_attempt001','history':ROOT/'crossplatform/round29/direct208_attempt001'};seals={n:read(p/'SEALED.json') for n,p in locations.items()};cohort=read(ROOT/'research/round24/input/D208_PRIVATE.json');assert len(cohort)==208 and len({r['id'] for r in cohort})==208 and collections.Counter(r['platform'] for r in cohort)=={'drone':104,'quad':104};decisions=[];arrays={};reasons=collections.Counter();floatcases=[]
 for r in cohort:
  rel='samples_PRIVATE/'+r['id']+'.npz';sp=locations['source']/rel;bp=locations['bank']/rel;gp=locations['geometry']/'cases_PRIVATE'/(r['id']+'.json');ga=locations['geometry']/'arrays_PRIVATE'/(r['id']+'.npz');hp=locations['history']/'predictions_PRIVATE'/(r['id']+'.npz')
  for name,path in [('source',sp),('bank',bp),('geometry',gp),('geometry',ga),('history',hp)]:assert sha(path)==seals[name]['files'][str(path.relative_to(locations[name]))]
  z=npz(sp);b=npz(bp);case=read(gp);geometry=npz(ga);h=npz(hp);p=npz(P/'predictions_PRIVATE'/(r['id']+'.npz'));author=read(P/'cases_PRIVATE'/(r['id']+'.json'));s=z['scores'];assert np.array_equal(s,geometry['source_scores']) and np.array_equal(z['boxes'],geometry['source_boxes'])
  v=(geometry['support_count']>=3)&(geometry['support_coverage']>=.15)&(geometry['projection_iou']>=.1)&(geometry['source_boxes'][:,3:]>0).all(1)[:,None];unique=v.sum(1)==1
  for m in range(len(geometry['masks_bool'])):assert case['full_groups'][str(m)]==np.flatnonzero(v[:,m]&unique).tolist()
  F=np.add(np.add(np.add(s.astype(np.float64),b['vision']*2),b['position']*2),b['relation']*2);assert np.array_equal(F,p['oldG_F64']) and np.array_equal(s,p['source_scores']);d=independent_decision(s,F,case['full_groups']);assert d==author['decision'],r['alias'];assert case['source_q0']==d['source_q0']
  for m,g in case['full_groups'].items():
   if g:assert int(case['representatives_full'][m])==sorted(g,key=lambda j:(-float(s[j]),j))[0]
  qs=[d['source_q0'],d['raw_G_q'],d['query']];assert p['arm_names'].tolist()==ARMS and np.array_equal(p['queries'],qs) and np.array_equal(p['boxes'],z['boxes'][qs]);j=h['arm_names'].tolist().index('legacy_G');assert h['queries'][j]==d['raw_G_q'] and np.array_equal(h['boxes'][j],p['boxes'][1])
  if not d['eligible']:assert d['accepted_preference'] is None and qs[2]==qs[0]
  else:assert d['source_group_rep']==d['source_q0'] and d['G_group_rep'] in d['G_group_members'] and not set(d['G_group_members'])&set(d['source_group_members']) and d['G_F64_gap_vs_q0']>0
  f32=int(np.argmax(F.astype(np.float32)))
  if f32!=d['raw_G_q']:floatcases.append({'alias':r['alias'],'platform':r['platform'],'F64_query':d['raw_G_q'],'F32_query':f32,'F64_queryscore_gap':float(F[d['raw_G_q']]-F[f32]),'diagnostic_only_no_F32_readout_substitution':True})
  reasons[d['reason']]+=1;decisions.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,qs))),'decision':d});arrays[r['id']]=p
 hist=read(P/'HISTORICAL_G_REPLAY.json');assert hist['all208_query_box_exact'] and hist['F64_F32_argmax_difference_count']==len(floatcases)==3;N=sum(r['decision']['eligible'] for r in decisions);assert N==seal['eligible_n']==32
 audit={'state':'PASS_R44_INDEPENDENT_PHASE1_READOUT_BEFORE_GT','n':208,'eligible_n':N,'eligible_by_platform':{pl:sum(r['decision']['eligible'] for r in decisions if r['platform']==pl) for pl in ['drone','quad']},'reason_counts':dict(reasons),'literal_old_F64_all208_exact':True,'historical_legacy_G_query_box_all208_exact':True,'original_source_scores_boxes_all208_exact_between_R24_R37':True,'geometry_unique_fullgroup_thresholds_reconstructed':True,'source_fixed_rep_query_boxes_624_exact':True,'F32_diagnostic_difference_count':len(floatcases),'no_F32_replacement':True,'no_new_teacher_filter_or_GT_used_in_readout_audit':True,'prediction_seal_sha256':sha(P/'SEALED.json')};dump(HERE/'READOUT_AUDIT_RESULT.json',audit);dump(HERE/'AUDITED_DECISIONS_PRIVATE.json',decisions);dump(HERE/'F64_F32_DIFFERENCES_PRIVATE.json',floatcases)
 # Only after all208 three-arm readouts are independently verified, load GT.
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};oldtask={r['id_PRIVATE']:r for r in read(ROOT/'research/round38/experiment/ALL_ROWS_PRIVATE.json')};out=[]
 for r in cohort:
  p=arrays[r['id']];i=index[r['id']];path=i['meta_path']
  if path not in containers:assert sha(path)==i['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][i['object_index']]['bbox_3d'],r['platform'],i.get('waymo_view_suffix'));boxes=p['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);iou=dict(zip(ARMS,map(float,candidate_ious(boxes,gt)[0])));assert abs(iou['source']-oldtask[r['id']]['IoU']['source'])<1e-12 and abs(iou['unrestricted_oldG']-oldtask[r['id']]['IoU']['legacy_G'])<1e-12;out.append({**next(d for d in decisions if d['id_PRIVATE']==r['id']),'IoU':iou})
 summary={};scenes=[];neg=[]
 for pl in ['drone','quad']:
  rr=[r for r in out if r['platform']==pl];summary[pl]={}
  for arm in ARMS:
   a=np.array([r['IoU'][arm] for r in rr]);s=np.array([r['IoU']['source'] for r in rr]);summary[pl][arm]={'n':len(rr),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100),'gain_vs_source_pp':float((a-s).mean()*100),'improved':int((a>s+1e-10).sum()),'harmed':int((a<s-1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr)}
  for scene in sorted({r['scene_PRIVATE'] for r in rr}):
   sr=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'eligible_n':sum(r['decision']['eligible'] for r in sr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in sr])*100) for a in ARMS},'few_existing_scenes_descriptive_only':True})
 summary['balanced_mIoU_percent']={a:.5*(summary['drone'][a]['mIoU_percent']+summary['quad'][a]['mIoU_percent']) for a in ARMS};summary['balanced_gain_vs_source_pp']=summary['balanced_mIoU_percent']['eligible_fixed_source_rep']-summary['balanced_mIoU_percent']['source']
 for r in out:
  for a in ARMS:
   for ref in ARMS:
    if a!=ref and r['IoU'][a]<r['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'arm':a,'reference':ref,'delta_pp':100*(r['IoU'][a]-r['IoU'][ref])})
 progression=N>0 and summary['balanced_gain_vs_source_pp']>0
 receipt={'state':'PASS_R44_INDEPENDENT_PHASE1_TASK_SCORING','n':208,'task_values':624,'eligible_n':N,'balanced_gain_vs_source_pp':summary['balanced_gain_vs_source_pp'],'prelocked_numeric_progression_gate_passed':progression,'gate_decision':'ROOT_MAY_CONSIDER_SEPARATE_FINITE_PHASE2_NOT_ADOPTION' if progression else 'STOP_PHASE1_NO_TTA','no_extra_coverage_or_uncertainty_gate_added':True,'phase2_training_not_authorized_by_this_receipt':True,'all_cached_source_legacy_metrics_reproduced':True,'negative_ordered_comparisons':len(neg),'prediction_seal_sha256':sha(P/'SEALED.json'),'root_phase1_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'no_new_G_GPU_VLM_model_training_or_FINAL_U_T':True,'seconds':time.perf_counter()-start};dump(HERE/'ALL208_TASK_ROWS_PRIVATE.json',out);dump(HERE/'TASK_SUMMARY.json',summary);dump(HERE/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(HERE/'ALL_NEGATIVES.json',neg);dump(HERE/'SCORING_RECEIPT.json',receipt);print(json.dumps({'audit':audit,'summary':summary,'receipt':receipt},indent=2))
if __name__=='__main__':main()
