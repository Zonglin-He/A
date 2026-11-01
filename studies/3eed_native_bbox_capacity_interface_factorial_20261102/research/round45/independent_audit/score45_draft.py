"""R45 post-seal task scoring draft; separate root authorization required."""
import sys,time,collections,numpy as np
from pathlib import Path
from audit45_draft import ROOT,HERE,P,EXP,CACHED_ARMS,POINT_ARMS,NEW_ARMS,ARMS,read,sha,verify,npz,dump
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious
def main():
 start=time.perf_counter();verify(P);auth=read(EXP/'POSTSEAL_EVALUATION_AUTHORIZATION.json');assert auth['A208_task_GT_scoring_authorized'] is True and auth['prediction_seal_sha256']==sha(P/'SEALED.json');audit=read(HERE/'AUDIT_RESULT.json');assert audit['state']=='PASS_R45_INDEPENDENT_CPU_CONTRACT' and audit['prediction_seal_sha256']==sha(P/'SEALED.json');selected={r['id_PRIVATE']:r for r in read(HERE/'AUDITED_SELECTED_QUERIES_PRIVATE.json')};rows=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')[:208];index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};old40={r['id_PRIVATE']:r for r in read(ROOT/'research/round40/experiment/ALL_ROWS_PRIVATE.json')};old42={r['id_PRIVATE']:r for r in read(ROOT/'research/round42/independent_audit/ALL208_TASK_ROWS_PRIVATE.json')};out=[]
 for r in rows:
  p=npz(P/'predictions_PRIVATE'/(r['id']+'.npz'));ix=index[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes=p['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);v=dict(zip(ARMS,map(float,candidate_ious(boxes,gt)[0])))
  for name in CACHED_ARMS:assert abs(v[name]-old40[r['id']]['IoU'][name])<1e-12
  for name in POINT_ARMS:assert abs(v[name]-old42[r['id']]['IoU'][name])<1e-12
  out.append({**selected[r['id']],'IoU':v})
 summary={};scenes=[];negative=[]
 def effects(rr):
  mean=lambda a:float(np.mean([r['IoU'][a] for r in rr])*100)
  cap_point=mean('7B_point_compatible_fullmap')-mean('point_compatible_fullmap');cap_center=mean(NEW_ARMS[0])-mean('bbox_fullmap');cap_iou=mean(NEW_ARMS[2])-mean('bbox_IoU_fullmap')
  return {'point_capacity7minus3_pp':cap_point,'primary_center_capacity7minus3_pp':cap_center,'secondary_rectIoU_capacity7minus3_pp':cap_iou,'primary_capacity_x_interface_interaction_pp':cap_center-cap_point,'secondary_capacity_x_interface_interaction_pp':cap_iou-cap_point,'3B_bboxcenter_minus_point_pp':mean('bbox_fullmap')-mean('point_compatible_fullmap'),'7B_bboxcenter_minus_point_pp':mean(NEW_ARMS[0])-mean('7B_point_compatible_fullmap')}
 for pl in ['drone','quad']:
  rr=[r for r in out if r['platform']==pl];summary[pl]={}
  for arm in ARMS:
   v=np.array([r['IoU'][arm] for r in rr]);s=np.array([r['IoU']['source'] for r in rr]);summary[pl][arm]={'n':104,'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_vs_source_pp':float((v-s).mean()*100),'improved_vs_source':int((v>s+1e-10).sum()),'harmed_vs_source':int((v<s-1e-10).sum()),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in rr)}
  for scene in sorted({r['scene_PRIVATE'] for r in rr}):
   part=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_PRIVATE':scene,'n':len(part),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in part])*100) for a in ARMS},'factorial_effects':effects(part),'descriptive_only_few_existing_scenes':True})
 summary['balanced_mIoU_percent']={a:.5*(summary['drone'][a]['mIoU_percent']+summary['quad'][a]['mIoU_percent']) for a in ARMS};contrasts={pl:effects([r for r in out if r['platform']==pl]) for pl in ['drone','quad']};contrasts['balanced']={k:.5*(contrasts['drone'][k]+contrasts['quad'][k]) for k in contrasts['drone']}
 coverage={pl:{a:dict(collections.Counter(r['readouts'][a]['status'] for r in out if r['platform']==pl)) for a in NEW_ARMS} for pl in ['drone','quad']};parse_counts={pl:dict(collections.Counter(r['parse_status'] for r in out if r['platform']==pl)) for pl in ['drone','quad']}
 for r in out:
  for arm in ARMS:
   for ref in ARMS:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:negative.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 identity=read(HERE/'IDENTITY24_JOIN_PRIVATE.json');byid={r['id_PRIVATE']:r for r in out}
 for i in identity:
  v=byid[i['id_PRIVATE']]['IoU'];i['final_IoU']=v[i['arm']];i['source_IoU']=v['source'];i['legacy_IoU']=v['legacy_G'];i['matched3B_IoU']=v['bbox_IoU_'+('oldsubset' if i['arm'].endswith('oldsubset') else 'fullmap')] if 'rectIoU' in i['arm'] else v['bbox_'+('oldsubset' if i['arm'].endswith('oldsubset') else 'fullmap')];i['gain_vs_source_pp']=100*(i['final_IoU']-i['source_IoU'])
 mean=summary['balanced_mIoU_percent'];primary=mean[NEW_ARMS[0]];secondary=mean[NEW_ARMS[2]];legacy=mean['legacy_G'];primary_stop=primary<=legacy;secondary_signal=secondary>legacy
 receipt={'state':'R45_INDEPENDENT_CPU_TASK_SCORING_COMPLETE','n':208,'arms':21,'task_values':4368,'cached_control_task_values_exact':3536,'all_ordered_negative_comparisons':len(negative),'primary_center_balanced_mIoU':primary,'secondary_rectIoU_balanced_mIoU':secondary,'legacy_balanced_mIoU':legacy,'primary_resource_stop_gate':primary_stop,'secondary_above_legacy_development_signal':secondary_signal,'decision':'STOP_FIXED_PRIMARY_RESOURCE_NO_AUTO_TTA' if primary_stop else 'PRIMARY_G_DEVELOPMENT_CANDIDATE_NOT_ADOPTION_OR_TTA','scientific_boundary':'PrimaryresourceSTOP isnotentire7Bnativefamily falsification. EffectivepredeclaredsecondarymustretaincandidatevaluebutcannotreplaceprimaryinR45;confirmationrequiresnewlock.','prediction_seal_sha256':sha(P/'SEALED.json'),'root_evaluation_auth_sha256':sha(EXP/'POSTSEAL_EVALUATION_AUTHORIZATION.json'),'all208_exposed_development_not_confirmation':True,'identity_nonexpert18clear6ambig_not_GTfilter_or_subsetrescue':True,'no_crossscene_CI_or_significance':True,'no_GPU_model_optimizer_VLM_FINAL_U_T':True,'seconds':time.perf_counter()-start};dump(HERE/'ALL208_TASK_ROWS_PRIVATE.json',out);dump(HERE/'TASK_SUMMARY.json',summary);dump(HERE/'CAPACITY_INTERFACE_CONTRASTS.json',contrasts);dump(HERE/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(HERE/'COVERAGE_UNKNOWN_SUMMARY.json',coverage);dump(HERE/'PARSE_STATUS_COUNTS.json',parse_counts);dump(HERE/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',negative);dump(HERE/'IDENTITY24_TASK_JOIN_PRIVATE.json',identity);dump(HERE/'SCORING_RECEIPT.json',receipt);print(__import__('json').dumps({'balanced_mIoU_percent':mean,'contrasts':contrasts,'receipt':receipt},indent=2))
if __name__=='__main__':main()
