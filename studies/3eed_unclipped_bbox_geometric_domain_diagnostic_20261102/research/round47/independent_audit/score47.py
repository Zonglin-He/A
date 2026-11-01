"""Score frozen predictions only after independent domain audit and root auth."""
import sys,time,collections
from audit47 import *
sys.path.insert(0,str(R/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious
def main():
 start=time.perf_counter();verify(B);auth=read(E/'POSTSEAL_EVALUATION_AUTHORIZATION.json');assert auth['A208_task_GT_scoring_authorized'] and auth['prediction_seal_sha256']==sha(B/'SEALED.json');a=read(H/'AUDIT_RESULT.json');assert a['state']=='PASS_R47_INDEPENDENT_CACHED_DOMAIN_CONTRACT' and a['prediction_seal_sha256']==sha(B/'SEALED.json');selected={r['id_PRIVATE']:r for r in read(H/'AUDITED_SELECTED_QUERIES_PRIVATE.json')};rows=read(R/'research/round27/input/ALL528_PRIVATE.json')[:208];idx={r['id']:r for r in read(R/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};old={r['id_PRIVATE']:r for r in read(R/'research/round45/independent_audit/ALL208_TASK_ROWS_PRIVATE.json')};containers={};out=[]
 for r in rows:
  z=npz(B/'predictions_PRIVATE'/(r['id']+'.npz'));ix=idx[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=z['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);arms=z['arm_names'].tolist();v=dict(zip(arms,map(float,candidate_ious(boxes,gt)[0])))
  for k in arms[:21]:assert abs(v[k]-old[r['id']]['IoU'][k])<1e-12
  out.append(dict(selected[r['id']],IoU=v))
 summary={};scenes=[];neg=[]
 for pl in ['drone','quad']:
  rr=[r for r in out if r['platform']==pl];summary[pl]={}
  for arm in arms:
   v=np.array([r['IoU'][arm] for r in rr]);s=np.array([r['IoU']['source'] for r in rr]);summary[pl][arm]={'n':104,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float((v-s).mean()*100),'improved_vs_source':int((v>s+1e-10).sum()),'harmed_vs_source':int((v<s-1e-10).sum()),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in rr)}
  for scene in sorted({r['scene_PRIVATE'] for r in rr}):
   part=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_PRIVATE':scene,'n':len(part),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in part])*100) for a in arms},'descriptive_only_few_exposed_scenes':True})
 summary['balanced_mIoU_percent']={k:.5*(summary['drone'][k]['mIoU_percent']+summary['quad'][k]['mIoU_percent']) for k in arms};mean=summary['balanced_mIoU_percent'];coverage={pl:{arm:dict(collections.Counter(r['readouts'][arm]['status'] for r in out if r['platform']==pl)) for arm in NEW} for pl in ['drone','quad']}
 for r in out:
  for arm in arms:
   for ref in arms:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 contrast={}
 for pl,rr in [('drone',[r for r in out if r['platform']=='drone']),('quad',[r for r in out if r['platform']=='quad']),('balanced',out)]:
  mm=lambda k:float(np.mean([r['IoU'][k] for r in rr])*100);contrast[pl]={f'{c}_{mode}_domain_minus_strict_pp':mm(f'{c}_bbox_domain_{mode}_fullmap')-mm(('bbox_' if mode=='center' else 'bbox_IoU_')+'fullmap' if c=='3B' else f'7B_bbox_{mode}_fullmap') for c in ['3B','7B'] for mode in ['center','rectIoU']}
 primary=mean['7B_bbox_domain_rectIoU_fullmap'];receipt={'state':'R47_INDEPENDENT_CPU_TASK_SCORING_COMPLETE','n':208,'arms':29,'task_values':6032,'cached_strict_control_task_values_exact':4368,'all_ordered_negative_comparisons':len(neg),'primary_domain_rectIoU_mIoU':primary,'strict7B_rectIoU_mIoU':mean['7B_bbox_rectIoU_fullmap'],'legacy_G_mIoU':mean['legacy_G'],'decision':'STOP_FIXED_DOMAIN_CONFIGURATION' if primary<=mean['legacy_G'] else 'DEVELOPMENT_CANDIDATE_ONLY_NOT_AUTO_TTA','original_R45_strict_results_unchanged':True,'prediction_seal_sha256':sha(B/'SEALED.json'),'all208_exposed_development_not_new_confirmation':True,'no_GPU_model_optimizer_FINAL_U_T':True,'seconds':time.perf_counter()-start};dump(H/'ALL208_TASK_ROWS_PRIVATE.json',out);dump(H/'TASK_SUMMARY.json',summary);dump(H/'DOMAIN_CONTRASTS.json',contrast);dump(H/'COVERAGE_UNKNOWN_SUMMARY.json',coverage);dump(H/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(H/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg);dump(H/'SCORING_RECEIPT.json',receipt);print(json.dumps({'balanced_mIoU_percent':mean,'domain_contrasts':contrast,'receipt':receipt}))
if __name__=='__main__':main()
