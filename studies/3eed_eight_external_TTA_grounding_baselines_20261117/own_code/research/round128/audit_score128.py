"""Separate post-seal CPU audit and task scorer. No adaptation implementation import."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round128';B=R/'crossplatform/round128';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
sys.path.insert(0,str(R/'tools'));from obb_iou import oriented_iou
def dump(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metrics(y):
 y=np.asarray(y);return {'n':len(y),'mIoU_percent':float(y.mean()*100),'Acc25_percent':float((y>.25).mean()*100),'Acc50_percent':float((y>.5).mean()*100)}
def main():
 assert all((H/f'{k}_MATRIX_SEALED.json').exists() for k in ['FAST','PREPARED','CLOUDFIXER','CONTROLS','MMTTA_CORRECTED','MMTTA_INDEPENDENT']);O.mkdir(exist_ok=False);reg=[e for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];allrows=[];methodrows=[];paired=[];scenes=[];audit={'n':0,'geometry_max_error':0.,'protected_aliases_no_GT_indexing':0,'negative_updates_retained':True};resources=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt];prior={r['id_PRIVATE']:r for r in rd(R/f'research/round127/scores_attempt001/DEV_{name}_ALL_ROWS_PRIVATE.json')};gtrows=rd(R/f'research/round96/experiment/OFFICIAL_{tgt.upper()}_VAL_ORDER_PRIVATE.json');records=[]
   for family in ['fast','prepared','cloudfixer','controls','mmtta_corrected','mmtta_independent']:
    D=B/('formal_'+family)/name;seal=rd(D/'SEALED.json');assert seal['n']==128
    for path,h in seal['files'].items():assert sha(D/path)==h,path
   for ix,e in enumerate(rr):
    sid=e['row']['id'];scores={};q={};boxes=None;meta={}
    for family in ['fast','prepared','cloudfixer','controls','mmtta_corrected','mmtta_independent']:
     D=B/('formal_'+family)/name
     with np.load(D/'samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as f:
      bb=f['source_boxes'];assert bb.shape==(256,6)
      if boxes is None:boxes=bb.copy()
      else:assert np.array_equal(bb,boxes)
      assert f['scores'].shape==(len(f['arm_names']),256) and np.isfinite(f['scores']).all() and np.allclose(f['scores'].sum(1),1,atol=1e-8);assert np.array_equal(f['scores'].argmax(1),f['queries'])
      for k,p in zip(f['arm_names'],f['scores']):
       k=str(k)
       if k in scores:assert np.array_equal(scores[k],p)
       scores[k]=p.copy();q[k]=int(p.argmax())
     meta[family]=rd(D/'records_PRIVATE'/(sid+'.json'));resources.append({'source':src,'target':tgt,'slot':ix,'family':family,'protected_alias':e['protected_alias_fallback'],'record':meta[family]})
    if e['protected_alias_fallback']:
     audit['protected_aliases_no_GT_indexing']+=1;assert len(set(q.values()))==1;ar={k:float(prior[sid]['arms']['original_source']['IoU']) for k in q};assert q['source']==prior[sid]['arms']['original_source']['query_PRIVATE']
    else:
     gt=np.array(gtrows[e['row']['anno_index']]['source_GT7_float32']);bb=np.c_[boxes,np.zeros(256)].astype(float);bb[:,3:6]=np.maximum(bb[:,3:6],1e-6);ious=many_iou(bb,gt);cs,sn=np.cos(gt[6]),np.sin(gt[6]);rot=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
     for which in set(q.values()):
      independent=oriented_iou(bb[which,:3],bb[which,3:6],np.eye(3),gt[:3],gt[3:6],rot);error=abs(independent-ious[which]);audit['geometry_max_error']=max(audit['geometry_max_error'],error);assert error<1e-6
     ar={k:float(ious[v]) for k,v in q.items()};assert abs(ar['source']-prior[sid]['arms']['original_source']['IoU'])<1e-6
    refs=prior[sid]['arms'];ar['R91_prepared_A_reference']=float(refs['prepared_A']['IoU']);ar['R127_frozen_AB_full_reference']=float(refs['zero_F']['IoU']);ar['legacy_frozen_ABC_reference']=float(refs['legacy_ABC_context']['IoU']);r={'source':src,'target':tgt,'slot':ix,'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'protected_alias':e['protected_alias_fallback'],'IoUs':ar,'queries_PRIVATE':q};records.append(r);allrows.append(r);audit['n']+=1
   keys=sorted(records[0]['IoUs'])
   for k in keys:
    methodrows.append({'source':src,'target':tgt,'arm':k,**metrics([r['IoUs'][k] for r in records])})
    for j,sc in enumerate(sorted({r['scene_PRIVATE'] for r in records})):
     scenes.append({'source':src,'target':tgt,'scene_slot':j,'arm':k,**metrics([r['IoUs'][k] for r in records if r['scene_PRIVATE']==sc])})
   comparisons=[('tent_post','source'),('tent_post','BN_statistics_only'),('memo_post','source'),('memo_post_three_mean','memo_BNprior_three_mean'),('memo_post','memo_BNprior_four_view_mean'),('eata_post','BN_four_view_mean'),('eata_post','eata_zero'),('MATE_step1','MATE_prepared_zero'),('MATE_step3','MATE_prepared_zero'),('MMTTA_independent_teacher_pre','MMTTA_independent_frozen_ensemble'),('MMTTA_independent_teacher_pre','MMTTA_independent_BN_pre'),('MMTTA_independent_teacher_post','MMTTA_independent_BN_post'),('MMTTA_independent_student_post','MMTTA_independent_frozen_ensemble'),('MMTTA_correct_teacher_pre','MMTTA_frozen_ensemble'),('MMTTA_correct_teacher_pre','MMTTA_correct_BN_pre'),('MMTTA_correct_teacher_post','MMTTA_correct_BN_post'),('MMTTA_correct_student_post','MMTTA_frozen_ensemble'),('MMTTA_teacher_pre','MMTTA_frozen_ensemble'),('MMTTA_teacher_pre','MMTTA_BN_only'),('MMTTA_teacher_post','MMTTA_BN_only_post'),('MMTTA_student_post','MMTTA_BN_only'),('MMTTA_teacher_post','MMTTA_frozen_ensemble'),('UniAdapter_online','pair_prepared_zero'),('UniAdapter_online','UniAdapter_episodic'),('UniAdapter_online','UniAdapter_shuffled'),('BFTT3D','pair_prepared_zero'),('BFTT3D','BFTT3D_no_TCA'),('CloudFixer','source'),('CloudFixer','CloudFixer_frozen_direct_denoise'),('CloudFixer','CloudFixer_shuffled_deformation')]
   for k,c in comparisons:
    d=np.array([r['IoUs'][k]-r['IoUs'][c] for r in records]);paired.append({'source':src,'target':tgt,'method':k,'control':c,'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_contribution_pp':float(d.max()*100/len(d)),'largest_negative_contribution_pp':float(d.min()*100/len(d))})
   dump(O/(name+'_ALL_ROWS_PRIVATE.json'),records)
 agg=[{'group':group,'arm':k,**{m:float(np.mean([r[m] for r in methodrows if r['arm']==k and (group=='all_six_equal' or r['target']==group)])) for m in ['mIoU_percent','Acc25_percent','Acc50_percent']}} for group in ['all_six_equal','waymo','drone','quad'] for k in keys]
 for name,obj in [('ALL_METHOD_ROWS',methodrows),('ALL_PAIRED_ROWS',paired),('ALL_SCENE_ROWS',scenes),('AGGREGATE_ROWS',agg),('RESOURCE_ROWS_PRIVATE',resources)]:dump(O/(name+'.json'),obj)
 assert audit['n']==768 and audit['protected_aliases_no_GT_indexing']==44;dump(H/'TASK_AUDIT.json',audit);print(json.dumps([r for r in agg if r['group']=='all_six_equal'],indent=2),flush=True)
if __name__=='__main__':main()
