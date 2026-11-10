"""Paired saved-scalar diagnostics after both sealed trials; no new predictions."""
from common66 import *
H=ROOT/'research/round66';v1=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');v2=read(H/'score_matched336_attempt001/ALL_TASK_ROWS_PRIVATE.json');p1=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');p2=read(H/'experiment/MATCHED_QUERY_DESIGN_LOCK_v2.json');strata=[];negative=[];comparisons=[];a1=[];a2=[]
for t1,t2 in zip(v1,v2):
 assert t1['id_PRIVATE']==t2['id_PRIVATE'];sid=t1['id_PRIVATE'];e1=load(ROOT/p1['evidence_root']/'samples_PRIVATE'/(sid+'.npz'));e2=load(ROOT/p2['evidence_root']/'samples_PRIVATE'/(sid+'.npz'));a1.append(bool(e1['update_allowed']));a2.append(bool(e2['update_allowed']))
 for arm in ARMS:
  diff=t2['IoU'][arm]-t1['IoU'][arm]
  if diff< -1e-10:negative.append({'row':'DEV_'+str(len(a1)).zfill(5),'platform':t1['platform'],'method':arm,'v1_IoU':t1['IoU'][arm],'v2_IoU':t2['IoU'][arm],'v2_minus_v1_pp':diff*100})
for arm in ARMS:
 d=np.array([b['IoU'][arm]-a['IoU'][arm] for a,b in zip(v1,v2)]);comparisons.append({'method':arm,'v2_minus_v1_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(a['queries'][arm]!=b['queries'][arm] for a,b in zip(v1,v2))})
a1=np.array(a1);a2=np.array(a2)
for x in [False,True]:
 for y in [False,True]:
  mask=(a1==x)&(a2==y);row={'v1_update_eligible':x,'v2_update_eligible':y,'n':int(mask.sum())}
  for version,tt in [('v1',v1),('v2',v2)]:
   for scope,arm,ref in [('R59','R59_tta1','R59'),('C','C_tta1','C_source')]:
    delta=np.array([t['IoU'][arm]-t['IoU'][ref] for t in tt]);row[version+'_'+scope+'_within_stratum_gain_pp']=float(delta[mask].mean()*100) if mask.any() else None;row[version+'_'+scope+'_full336_contribution_pp']=float(delta[mask].sum()/336*100)
  strata.append(row)
D={}
for version,suffix in [('v1',''),('v2','_QUERY')]:
 groups=read(H/('ALL_EVIDENCE_PROCESS_ROWS'+suffix+'.json'));positive=[r for r in groups if r['positive']];fullpositive=[r for r in positive if r['full_only']>0];rawpositive=[r for r in positive if r['full_only']>0 and r['category_only']>=0];D[version]={'positive_E_groups':len(positive),'positive_E_mean_corrected_full_support_positive':len(fullpositive),'positive_E_mean_corrected_full_support_nonpositive':len(positive)-len(fullpositive),'positive_E_with_full_support_positive_task_good':sum(r['postseal_best_task_IoU']>.25 for r in fullpositive),'all_positive_E_task_good':sum(r['postseal_best_task_IoU']>.25 for r in positive),'not_a_filtered_method_or_new_TTA_run':True}
D['posthoc_v1_updates_restricted_to_v2_eligible_full336_contribution_R59_pp']=sum(r['v1_R59_full336_contribution_pp'] for r in strata if r['v2_update_eligible']);D['posthoc_v2_updates_full336_gain_R59_pp']=sum(r['v2_R59_full336_contribution_pp'] for r in strata);D['this_is_saved_scalar_stratification_not_method_selection']=True
for name,val in [('PAIRED_METHOD_COMPARISONS',comparisons),('ALL_PAIRED_V2_V1_NEGATIVES',negative),('PAIRED_ELIGIBILITY_STRATA',strata),('POSTRESULT_SUPPORT_SIGN_DIAGNOSTICS',D)]:dump(H/(name+'.json'),val)
print({'strata':strata,'support_sign_diagnostics':D,'paired_negatives':len(negative)})
