"""DRAFT: post-seal task scoring requires a separate root authorization file.

No execution during preparation; no GT loaded at import time.
"""
import sys,json,time,collections,numpy as np
from pathlib import Path
from audit42_draft import ROOT,HERE,P,EXP,PRE,ARMS,read,sha,verify,npz,dump
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import transform_gt,candidate_ious
def main():
 started=time.perf_counter();verify(P);auth=read(EXP/'POSTSEAL_EVALUATION_AUTHORIZATION.json');assert auth['A208_task_GT_scoring_authorized'] is True and auth['prediction_seal_sha256']==sha(P/'SEALED.json')
 audit=read(HERE/'AUDIT_RESULT.json');assert audit['state']=='PASS_R42_INDEPENDENT_CPU_CONTRACT' and audit['prediction_seal_sha256']==sha(P/'SEALED.json');selected={r['id_PRIVATE']:r for r in read(HERE/'AUDITED_SELECTED_QUERIES_PRIVATE.json')};rows=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')[:208];index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};output=[]
 old={r['id_PRIVATE']:r for r in read(ROOT/'research/round38/experiment/ALL_ROWS_PRIVATE.json')};oldnames={'source':'source','R29_direct':'R29_direct','legacy_G':'legacy_G','3B_point_compatible_fullmap':'point_compatible_fullmap','3B_point_compatible_oldsubset':'point_compatible_oldsubset','3B_som_compatible_fullmap':'som_compatible_fullmap'}
 for r in rows:
  pred=npz(P/'predictions_PRIVATE'/(r['id']+'.npz'));assert pred['arm_names'].tolist()==ARMS;ix=index[r['id']];path=ix['meta_path']
  if path not in containers:assert sha(path)==ix['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=pred['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);vv=candidate_ious(b,gt)[0];v=dict(zip(ARMS,map(float,vv)))
  for a,o in oldnames.items():assert abs(v[a]-old[r['id']]['IoU'][o])<1e-12
  output.append({**selected[r['id']],'IoU':v})
 summary={};scenes=[];neg=[]
 for pl in ['drone','quad']:
  rr=[r for r in output if r['platform']==pl];summary[pl]={}
  for arm in ARMS:
   a=np.array([r['IoU'][arm] for r in rr]);ref=np.array([r['IoU']['source'] for r in rr]);matched=np.array([r['IoU']['3B_point_compatible_fullmap'] for r in rr]);summary[pl][arm]={'n':104,'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100),'gain_vs_source_pp':float((a-ref).mean()*100),'gain_vs_matched3B_pp':float((a-matched).mean()*100),'improved_vs_source':int((a>ref+1e-10).sum()),'harmed_vs_source':int((a<ref-1e-10).sum()),'improved_vs_matched3B':int((a>matched+1e-10).sum()),'harmed_vs_matched3B':int((a<matched-1e-10).sum()),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in rr)}
  for scene in sorted({r['scene_PRIVATE'] for r in rr}):
   part=[r for r in rr if r['scene_PRIVATE']==scene];scenes.append({'platform':pl,'scene_PRIVATE':scene,'n':len(part),'mean_IoU_percent':{a:100*float(np.mean([r['IoU'][a] for r in part])) for a in ARMS},'descriptive_only_few_existing_scenes':True})
 summary['balanced_mIoU_percent']={a:.5*(summary['drone'][a]['mIoU_percent']+summary['quad'][a]['mIoU_percent']) for a in ARMS}
 coverage={pl:{mapping:dict(collections.Counter(r['readouts'][mapping]['status'] for r in output if r['platform']==pl)) for mapping in ['fullmap','oldsubset']} for pl in ['drone','quad']}
 for r in output:
  for arm in ARMS:
   for ref in ARMS:
    if arm!=ref and r['IoU'][arm]<r['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'arm':arm,'reference':ref,'delta_pp':100*(r['IoU'][arm]-r['IoU'][ref])})
 means=summary['balanced_mIoU_percent'];new=means['7B_point_compatible_fullmap'];matched=means['3B_point_compatible_fullmap'];legacy=means['legacy_G'];source=means['source'];decision='NO_CAPACITY_OVERALL_GAIN_NO_TTA' if new<=matched else 'LIMITED_CAPACITY_INCREMENT_BELOW_LEGACY_NO_TTA' if new<=legacy else 'G_DEVELOPMENT_CANDIDATE_ONLY_REQUIRES_SEPARATE_CONFIRMATION_NO_AUTO_TTA' if new>source else 'NO_SOURCE_GAIN_NO_TTA'
 identities=read(HERE/'IDENTITY24_JOIN_PRIVATE.json');byid={r['id_PRIVATE']:r for r in output}
 for i in identities:
  v=byid[i['id_PRIVATE']]['IoU'];i['final_IoU']=v[i['arm']];i['source_IoU']=v['source'];i['matched3B_IoU']=v['3B_point_compatible_'+i['mapping']];i['gain_vs_source_pp']=100*(i['final_IoU']-i['source_IoU']);i['gain_vs_matched3B_pp']=100*(i['final_IoU']-i['matched3B_IoU'])
 dump(HERE/'IDENTITY24_TASK_JOIN_PRIVATE.json',identities)
 dump(HERE/'ALL208_TASK_ROWS_PRIVATE.json',output);dump(HERE/'TASK_SUMMARY.json',summary);dump(HERE/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg);dump(HERE/'SCENE_SUMMARIES_PRIVATE.json',scenes);dump(HERE/'COVERAGE_UNKNOWN_SUMMARY.json',coverage)
 receipt={'state':'R42_INDEPENDENT_CPU_TASK_SCORING_COMPLETE','n':208,'task_values':1664,'negative_ordered_comparisons':len(neg),'prediction_seal_sha256':sha(P/'SEALED.json'),'GT_authorization_sha256':sha(EXP/'POSTSEAL_EVALUATION_AUTHORIZATION.json'),'all_cached_controls_reproduced':True,'decision':decision,'data':'exposedA208development, notnewconfirmation','no_general_clusterCI_or_significance_claim':True,'no_identity_or_GT_filtering_of_feedback':True,'no_GPU_model_or_FINAL':'metadata fetched onlyfor208authorizedids,FINALnotread','seconds':time.perf_counter()-started};dump(HERE/'SCORING_RECEIPT.json',receipt);print(json.dumps({'summary':summary['balanced_mIoU_percent'],'receipt':receipt},indent=2))
if __name__=='__main__':main()
