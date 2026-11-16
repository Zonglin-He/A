"""All sealed R98 outcomes: one-component contrasts and balanced good/failure cases."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round99';OLD=R/'research/round98/scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
CONTRASTS=[('prepared_frozen','original_source','source_preparation'),('frozen_C','prepared_frozen','mask_only'),('frozen_FC','prepared_frozen','add_C_readout_only'),('binding_direct','prepared_frozen','G_only_F'),('lm_F','binding_direct','fit_execution_F'),('lm_C','frozen_C','weight_transfer_C'),('lm_FC','frozen_FC','weights_same_FC'),('lm_FC','direct_FC','weights_vs_same_evidence'),('lm_FC','prepared_frozen','complete_recipe')]
def main():
 assert rd(R/'research/round98/SCALAR_AUDIT.json')['n']==18620;out=[];cases=[];seen=0;cohorts={};proofs={}
 for target in ['waymo','drone','quad']:
  f=R/f'research/round97/input/TARGET_{target.upper()}_PRIVATE.json';rr=rd(f);buckets={}
  for r in rr:buckets.setdefault(r['scene'],[]).append(r)
  key=lambda s:hashlib.sha256(('R99locked128_'+s).encode()).hexdigest()
  for sc in buckets:buckets[sc].sort(key=lambda r:key(r['id']))
  scs=sorted(buckets,key=key);choice=[];i=0
  while len(choice)<128:
   for sc in scs:
    if i<len(buckets[sc]):choice.append(buckets[sc][i])
    if len(choice)==128:break
   i+=1
  choice.sort(key=lambda r:r['anno_index']);cohorts[target]=choice;proofs[target]={'source_manifest_sha256':sha(f),'n':128,'scenes':len({r['scene'] for r in choice}),'GT_used_for_choice':False};dump(H/f'input/TARGET_{target.upper()}128_PRIVATE.json',choice)
 for source in ['waymo','drone','quad']:
  for target in ['waymo','drone','quad']:
   if source==target:continue
   name=source+'_to_'+target;rr=rd(OLD/(name+'_ALL_ROWS_PRIVATE.json'));seen+=len(rr)
   for method,control,label in CONTRASTS:
    a=np.array([r['arms'][method]['IoU'] for r in rr]);b=np.array([r['arms'][control]['IoU'] for r in rr]);d=a-b;oracle=np.array([r['arms'][control]['oracle_IoU_diagnostic'] for r in rr]);out.append({'source':source,'target':target,'component':label,'method':method,'control':control,'n':len(rr),'mIoU_gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'unchanged':int((abs(d)<=1e-12).sum()),'Acc25_rescue':int(((a>.25)&(b<=.25)).sum()),'Acc25_break':int(((a<=.25)&(b>.25)).sum()),'recoverable_errors25':int(((b<=.25)&(oracle>.25)).sum()),'candidate_missing25':int((oracle<=.25).sum()),'largest_positive_contribution_pp':float(max(0,d.max())/len(rr)*100),'largest_negative_contribution_pp':float(min(0,d.min())/len(rr)*100)})
    for sign,idx in [('good',np.flatnonzero(d>1e-12)),('failure',np.flatnonzero(d< -1e-12))]:
     if not len(idx):continue
     ranked=sorted(idx,key=lambda i:(-abs(d[i]),i));chosen=list(dict.fromkeys(ranked[:2]+[ranked[len(ranked)//2]]))
     for j in chosen:
      r=rr[j];cases.append({'case':'CASE'+str(len(cases)+1).zfill(4),'source':source,'target':target,'contrast':label,'kind':sign,'row_PRIVATE':r['id_PRIVATE'],'scene_PRIVATE':r['scene_PRIVATE'],'method':method,'control':control,'method_IoU':float(a[j]),'control_IoU':float(b[j]),'change_IoU':float(d[j]),'oracle_IoU_diagnostic':float(oracle[j]),'all_arms':r['arms'],'status':r['status'],'fits':r['fits'],'selection':'two_extremes_and_median_per_direction_component_sign_descriptive_only'})
 assert seen==18620;macro=[]
 for _,_,label in CONTRASTS:
  z=[x for x in out if x['component']==label];macro.append({'component':label,'gain_pp_equal_six':float(np.mean([x['mIoU_gain_pp'] for x in z])),'positive_directions':sum(x['mIoU_gain_pp']>0 for x in z),'better':sum(x['better'] for x in z),'worse':sum(x['worse'] for x in z)})
 dump(H/'COMPONENT_ROWS.json',out);dump(H/'COMPONENT_MACRO.json',macro);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'experiment/COHORT_LOCK.json',{'policy':'scene_round_robin_then_hash_order_no_outcomes','targets':proofs,'directions':6,'directed_n':768,'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in (H/'input').glob('*.json')},'exposure':'R98 historical development, not independent confirmation','FINAL_access':False})
 lines=['# R99 Part A — 已封存R98组件断点','','仅重组已审核的18620条结果，无新模型或新GT评分。所有good/failure选择只作描述，不作为标签筛选。','', '| 因素 | 六方向等权Δ pp | 正方向 | 改善/恶化 |','|---|---:|---:|---:|']
 for x in macro:lines.append(f"| {x['component']} | {x['gain_pp_equal_six']:+.6f} | {x['positive_directions']}/6 | {x['better']}/{x['worse']} |")
 lines+=['','假设排序：先区分mask本身造成的baseline/readout变化，再检查只拟合F的末端权重是否过拟合；其次才判断绑定差分信号是否有任务信息，有限求解预算仍未排除。下一步只改变拟合视图数，沿用现有G和参数范围，保留同信息直接与错配控制。所有旧结果均为已暴露开发证据。']
 (H/'PART_A_REPORT.md').write_text('\n'.join(lines)+'\n');print({'n':seen,'cases':len(cases),'new_cohort_directed_n':768,'macro':macro})
if __name__=='__main__':main()
