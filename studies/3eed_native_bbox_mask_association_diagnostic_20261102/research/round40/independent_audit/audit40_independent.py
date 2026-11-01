"""Independent cached standard rectangle-IoU assignment and all-task audit."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,time,datetime,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'research/round37/independent_audit'),str(ROOT/'research/round5/audit')]
from audit37_strict_independent import rows,read,dump,sha,verify,load_npz
from independent_core_v2 import candidate_ious,transform_gt
BASE=ROOT/'crossplatform/round40';EXP=ROOT/'research/round40/experiment';OUT=Path(__file__).resolve().parent;PRED=BASE/'predictions208_attempt001';PREV=ROOT/'crossplatform/round39/inference208_attempt002';ARMS=['source','R29_direct','legacy_G','point_strict_fullmap','point_strict_oldsubset','som_strict_fullmap','som_strict_oldsubset','point_compatible_fullmap','point_compatible_oldsubset','som_compatible_fullmap','som_compatible_oldsubset','bbox_fullmap','bbox_oldsubset','bbox_IoU_fullmap','bbox_IoU_oldsubset']
def close(a,b,t=1e-10):assert abs(float(a)-float(b))<t,(a,b)
def assign(raw,case,z):
 p=raw['parsed'];grid=raw['image_grid_thw'][0];W,H=case['image_size_wh'];wp,hp=grid[2]*14,grid[1]*14;assert raw['model_resized_wh']==[wp,hp];d={'status':p['status'],'mask_index':None,'bbox_resized_unchanged':p['value'],'bbox_original_xyxy':None,'mask_IoU':None,'top_IoU':None,'top2_margin':None,'exact_max_tie_n':None,'all_masks_including_unmappable':True}
 if p['status']!='parsed':return d
 x1,y1,x2,y2=p['value'];assert 0<=x1<x2<=wp and 0<=y1<y2<=hp;box=[x1*W/wp,y1*H/hp,x2*W/wp,y2*H/hp];d['bbox_original_xyxy']=box;rects=z.get('mask_boxes');n=len(z['masks_bool'])
 if rects is None or np.shape(rects)!=(n,4) or not np.isfinite(rects).all() or np.any(rects[:,2:]<=rects[:,:2]):d['status']='missing_or_invalid_detector_box_inventory';return d
 if not n:d.update(status='no_independent_masks',mask_IoU=[],exact_max_tie_n=0);return d
 cc={x['mask_index']:x for x in case['candidates']};assert set(cc)==set(range(n)) and all(np.array_equal(cc[m]['bbox_xyxy'],rects[m]) for m in range(n))
 b=np.array(box,float);r=np.array(rects,float);inter=np.maximum(np.minimum(b[2:],r[:,2:])-np.maximum(b[:2],r[:,:2]),0).prod(1);union=np.maximum(b[2:]-b[:2],0).prod()+np.maximum(r[:,2:]-r[:,:2],0).prod(1)-inter;v=np.divide(inter,union,out=np.zeros(n,float),where=union>0);maximum=float(v.max());hit=np.flatnonzero(v==maximum);ordered=np.sort(v);d.update(mask_IoU=v.tolist(),top_IoU=maximum,top2_margin=float(maximum-ordered[-2]) if n>1 else None,exact_max_tie_n=len(hit))
 if maximum<=0:d['status']='all_rectangle_IoU_zero'
 elif len(hit)!=1:d['status']='exact_max_rectangle_IoU_tie'
 else:d.update(status='unique_positive_rectangle_IoU',mask_index=int(hit[0]))
 return d

def readout(a,case,z,subset):
 q0=int(z['source_scores'].argmax());m=a['mask_index'];table=case['old_subset_groups'] if subset else case['full_groups'];g=table.get(str(m),[]) if m is not None else [];base={'mask_index':m,'original_bbox_xyxy':a['bbox_original_xyxy'],'old_subset_contains_selected_mask':m is not None and str(m) in case['old_subset_groups'],'full_group_nonempty':m is not None and bool(case['full_groups'].get(str(m),[])),'assignment_status':a['status'],'no_second_best_if_3D_group_empty':True}
 if not g:return dict(base,query=q0,mapped=False,status=a['status'] if m is None else 'absent_from_old_subset' if subset and str(m) not in table else 'mask_without_3D_group',group_size=0,query_changed=False)
 q=max(g,key=lambda j:(float(z['source_scores'][j]),-j));return dict(base,query=int(q),mapped=True,status='mapped_original_query',group_size=len(g),query_changed=q!=q0)
def fixtures():
 raw={'parsed':{'status':'parsed','value':[0,0,10,10]},'image_grid_thw':[[1,1,1]],'model_resized_wh':[14,14]};z={'masks_bool':np.zeros((2,14,14),bool),'mask_boxes':np.array([[0,0,10,10],[5,5,14,14]],float),'source_scores':np.array([1.,0.])};case={'image_size_wh':[14,14],'candidates':[{'mask_index':i,'bbox_xyxy':z['mask_boxes'][i].tolist()} for i in range(2)],'full_groups':{'0':[],'1':[1]},'old_subset_groups':{'1':[1]}}
 a=assign(raw,case,z);assert a['mask_index']==0 and not readout(a,case,z,False)['mapped'] and readout(a,case,z,False)['query']==0
 z2=dict(z,mask_boxes=np.array([[0,0,10,10],[0,0,10,10]],float));case2=dict(case,candidates=[{'mask_index':i,'bbox_xyxy':z2['mask_boxes'][i].tolist()} for i in range(2)]);assert assign(raw,case2,z2)['status']=='exact_max_rectangle_IoU_tie'
 raw2=dict(raw,parsed={'status':'parsed','value':[11,0,14,3]});z3=dict(z,mask_boxes=np.array([[0,0,2,2],[0,3,2,5]],float));case3=dict(case,candidates=[{'mask_index':i,'bbox_xyxy':z3['mask_boxes'][i].tolist()} for i in range(2)]);assert assign(raw2,case3,z3)['status']=='all_rectangle_IoU_zero'
 assert assign(raw,case,dict(z,mask_boxes=None))['status']=='missing_or_invalid_detector_box_inventory'
 return {'all_masks_before_group_availability':True,'no_second_best_when_max_mask_group_empty':True,'exact_tie_and_allzero_UNKNOWN':True,'invalid_inventory_unknown':True}
def main():
 start=time.perf_counter();plan=read(EXP/'PLAN_LOCK.json');lock=read(EXP/'IMPLEMENTATION_LOCK.json');assert lock['plan_sha256']==sha(EXP/'PLAN_LOCK.json')
 for p,h in {**plan['input_sha256'],**lock['program_sha256']}.items():assert sha(ROOT/p)==h,p
 verify(PRED);verify(PREV);verify(ROOT/'crossplatform/round37/prepared208_attempt001');r39=read(ROOT/'research/round39/independent_audit/AUDIT_RESULT.json');assert r39['state']=='PASS_R39_ORIGINAL_CENTER_PIPELINE' and r39['prediction_seal_sha256']==sha(PREV/'SEALED.json');bindings={r['alias']:r for r in read(ROOT/'research/round39/independent_audit/ALL208_RAW_DURABILITY_BINDINGS.json')}
 jobs=[read(p) for p in (ROOT/'research/NIGHT_20261101/job_receipts').glob('*.json')];scorejob=next(x for x in jobs if x.get('command')==['crossplatform/env/bin/python','crossplatform/round40/score40.py']);assert scorejob['returncode']==0;assert (PRED/'SEALED.json').stat().st_mtime<datetime.datetime.fromisoformat(scorejob['started_utc']).timestamp();assert read(EXP/'CPU_AUDIT.json')['prediction_seal_sha256']==sha(PRED/'SEALED.json')
 original={r['id_PRIVATE']:r for r in read(EXP/'ALL_ROWS_PRIVATE.json')};index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};task=[];aflags=[];assignstatus=collections.Counter();rawexact=0;emptymax=0;maxerr=0.
 for r in rows('A208'):
  z=load_npz(ROOT/'crossplatform/round37/prepared208_attempt001/arrays_PRIVATE'/(r['id']+'.npz'));case=read(ROOT/'crossplatform/round37/prepared208_attempt001/cases_PRIVATE'/(r['id']+'.json'));p=load_npz(PRED/'predictions_PRIVATE'/(r['id']+'.npz'));prev=load_npz(PREV/'predictions_PRIVATE'/(r['id']+'.npz'));c=read(PRED/'cases_PRIVATE'/(r['id']+'.json'));old39=read(PREV/'predictions_PRIVATE'/(r['id']+'.json'));old38=read(ROOT/old39['previous_case_file']);path=ROOT/c['raw_file'];assert sha(path)==c['raw_sha256']==bindings[r['alias']]['call_sha256'];raw=read(path)['result'];assert c['bbox_parsed_original']==raw['parsed'] and c['bbox_position_values_exact'];a=assign(raw,case,z);assert a==c['assignment'];rawexact+=1;assignstatus[a['status']]+=1
  if a['mask_index'] is not None and not case['full_groups'].get(str(a['mask_index']),[]):emptymax+=1
  if a['mask_IoU'] is not None:maxerr=max(maxerr,float(np.max(np.abs(np.asarray(a['mask_IoU'])-np.asarray(c['assignment']['mask_IoU'])),initial=0)))
  assert p['arm_names'].tolist()==ARMS and prev['arm_names'].tolist()==ARMS[:13] and np.array_equal(p['queries'][:13],prev['queries']) and np.array_equal(p['boxes'][:13],prev['boxes']);assert np.array_equal(p['boxes'],z['source_boxes'][p['queries']]);q=dict(zip(ARMS,map(int,p['queries'])));rd=dict(old38['readouts'],**old39['readouts'])
  for mapping in ['fullmap','oldsubset']:
   name='bbox_IoU_'+mapping;rd[name]=readout(a,case,z,mapping=='oldsubset');assert rd[name]==c['readouts'][name] and q[name]==rd[name]['query']
  center=old39['readouts']['bbox_fullmap'];sg={'center_unknown':not center['mapped'],'center_IoU_mask_conflict':center['mask_index'] is not None and a['mask_index'] is not None and center['mask_index']!=a['mask_index']};assert sg==c['subgroups_defined_without_GT'];ref=original[r['id']];assert q==ref['queries'] and rd==ref['readouts'] and sg==ref['subgroups_defined_without_GT'];ix=index[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=p['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);vals=candidate_ious(boxes,gt)[0]
  for arm,v in zip(ARMS,vals):close(v,ref['IoU'][arm],1e-12)
  task.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':dict(zip(ARMS,map(float,vals))),'readouts':rd,'subgroups_defined_without_GT':sg});aflags.append({'alias':r['alias'],'platform':r['platform'],'assignment_status':a['status'],'top_IoU':a['top_IoU'],'top2_margin':a['top2_margin'],'exact_max_tie_n':a['exact_max_tie_n'],'all_masks_including_unmappable':True,'selected_mask_has_full_group':a['mask_index'] is not None and bool(case['full_groups'].get(str(a['mask_index']),[])),'center_unknown':sg['center_unknown'],'center_IoU_mask_conflict':sg['center_IoU_mask_conflict']})
 negative=[]
 for r in task:
  for arm in ARMS:
   for ref in ['source','R29_direct','legacy_G','point_compatible_fullmap','som_compatible_fullmap','bbox_fullmap','bbox_IoU_fullmap']:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 assert negative==read(EXP/'ALL_NEGATIVES.json');summary=read(EXP/'TASK_SUMMARY.json');coverage={};scenes=[]
 for pl in ['drone','quad']:
  part=[r for r in task if r['platform']==pl];coverage[pl]={}
  for arm in ARMS:
   v=np.array([r['IoU'][arm] for r in part]);d=v-np.array([r['IoU']['source'] for r in part]);calc={'n':104,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in part)}
   for k,vv in calc.items():close(vv,summary[pl][arm][k])
  for arm in ARMS[3:]:coverage[pl][arm]={'n':104,'mapped':sum(r['readouts'][arm]['mapped'] for r in part),'unknown_fallback':sum(not r['readouts'][arm]['mapped'] for r in part),'status_counts':dict(collections.Counter(r['readouts'][arm]['status'] for r in part))}
  for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in part})):
   ss=[r for r in part if r['scene_PRIVATE']==sc];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{arm:float(np.mean([r['IoU'][arm] for r in ss])*100) for arm in ARMS}})
 assert coverage==read(EXP/'COVERAGE_UNKNOWN.json') and scenes==read(EXP/'SCENE_ROWS.json')
 for arm in ARMS:close(np.mean([summary[pl][arm]['mIoU_percent'] for pl in ['drone','quad']]),summary['balanced_mIoU_percent'][arm])
 for rec in read(EXP/'MECHANISM_SUBGROUPS.json'):
  group=rec['subgroup'];part=[r for r in task if r['platform']==rec['platform'] and (group=='overall' or r['subgroups_defined_without_GT'][group])];assert len(part)==rec['n'];d=np.array([r['IoU']['bbox_IoU_fullmap']-r['IoU']['bbox_fullmap'] for r in part]);assert int((d>1e-10).sum())==rec['improved'] and int((d< -1e-10).sum())==rec['worsened'] and int((abs(d)<=1e-10).sum())==rec['unchanged']
  if part:
   close(d.mean()*100,rec['bboxIoU_vs_center_mean_gain_pp'])
   for arm,v in rec['mean_IoU_percent'].items():close(np.mean([r['IoU'][arm] for r in part])*100,v)
  else:assert rec['bboxIoU_vs_center_mean_gain_pp'] is None and all(v is None for v in rec['mean_IoU_percent'].values())
 rng=np.random.default_rng(40);clusters={pl:[[r for r in task if r['platform']==pl and r['scene_PRIVATE']==sc] for sc in sorted({r['scene_PRIVATE'] for r in task if r['platform']==pl})] for pl in ['drone','quad']}
 for rec in read(EXP/'FACTOR_CONTRASTS.json'):
  a,b=rec['arm'],rec['reference'];delta={pl:float(np.mean([r['IoU'][a]-r['IoU'][b] for r in task if r['platform']==pl])*100) for pl in ['drone','quad']}
  for pl,v in delta.items():close(v,rec['platform_gain_pp'][pl])
  close(np.mean(list(delta.values())),rec['balanced_gain_pp']);dist=[]
  for _ in range(2000):dist.append(100*np.mean([np.mean([r['IoU'][a]-r['IoU'][b] for j in rng.integers(0,len(cc),len(cc)) for r in cc[j]]) for pl,cc in clusters.items()]))
  assert np.allclose(np.quantile(dist,[.025,.975]),rec['descriptive_scene_bootstrap95_pp'],atol=1e-12,rtol=0)
 result={'state':'PASS_R40_CACHED_RECTANGLE_ASSIGNMENT','observations':208,'arms':15,'task_values':3120,'negative_comparisons':len(negative),'all13_R39_controls_queries_boxes_exact':True,'R39_bbox_raw_call_hashes_and_values_unchanged':208,'detector_rectangle_fullinventory_and_candidate_boxes_exact':True,'all_mask_rectangle_IoU_vectors_maxerror':maxerr,'selection_before_group_availability':True,'selected_max_mask_empty_fullgroup_no_second_best':emptymax,'assignment_status_counts':dict(assignstatus),'source3Dboxes_representatives_and_geometry_unchanged':True,'all_metrics_coverage_scene_contrasts_subgroups_reproduced':True,'security_and_geometry_fixtures':fixtures(),'prediction_seal_sha256':sha(PRED/'SEALED.json'),'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'implementation_lock_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'prediction_sealed_before_score':True,'no_new_VLM_GPU_optimizer_or_FINAL_U_T_identity_review':True,'scientific_limit':'Standard geometry association on exposed A208; not new algorithm or stronger bbox model, no TTA/default adoption.','seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result);dump(OUT/'ALL208_RECTANGLE_TASK_ROWS_PRIVATE.json',task);dump(OUT/'ALL_NEGATIVES.json',negative);dump(OUT/'ALL208_ASSIGNMENT_SCALAR_FLAGS.json',aflags);print(result)
if __name__=='__main__':main()
