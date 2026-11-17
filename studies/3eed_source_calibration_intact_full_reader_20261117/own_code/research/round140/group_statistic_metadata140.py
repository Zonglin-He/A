"""GT-free cached check: group-MEAN qualification versus actual group-MAX choice."""
from common140 import *
import numpy as np,collections
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');rows=[];counts=collections.Counter()
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;a=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));ass=a['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S=a['S'];C=a['fullreader_ownsource_C'];eligible=bool(a['fullreader_ownsource_eligible']);r={'source':src,'target':tgt,'id_PRIVATE':sid,'eligible':eligible,'groups':len(groups),'multiquery_groups':sum(len(g)>1 for g in groups),'queries_in_multiquery_groups':sum(len(g) for g in groups if len(g)>1),'S_groups':int(S.sum()),'C_groups':int(C.sum())}
  if eligible:
   for v in ['A','B']:
    prob=a['p'+v];mean=np.array([prob[g].mean() for g in groups]);maxi=np.array([prob[g].max() for g in groups]);ids=np.flatnonzero(S);topmax=int(ids[np.argmax(maxi[S])]);topmean=int(ids[np.argmax(mean[S])]);r[v+'_supported_MAX_leader_differs_MEAN']=topmax!=topmean;r[v+'_supported_MAX_leader_excluded_by_MEAN_C']=not bool(C[topmax]);r[v+'_global_MAX_leader_supported']=bool(S[ass[prob.argmax()]]);r[v+'_global_MAX_leader_excluded_when_supported']=bool(S[ass[prob.argmax()]] and not C[ass[prob.argmax()]]);r[v+'_topMAX_group_size']=len(groups[topmax])
   counts['eligible']+=1
   for k,v in r.items():
    if isinstance(v,bool) and k!='eligible':counts[k]+=v
  rows.append(r)
 out={'status':'GT_FREE_GROUP_STATISTIC_DIAGNOSTIC','n':len(rows),'counts':dict(counts),'median_group_count':float(np.median([r['groups'] for r in rows])),'median_multiquery_groups':float(np.median([r['multiquery_groups'] for r in rows])),'mean_queries_in_multiquery_groups':float(np.mean([r['queries_in_multiquery_groups'] for r in rows])),'qualification_and_partial_current_groupMEAN':True,'final_Top1_and_R136_Cinternal_rank_groupMAX':True,'not_instance_identity_or_task_gain_evidence':True,'new_target_task_scores_or_GT_read':False,'predictor_or_weights_changed':False,'FINAL_access':False};dump(H/'GROUP_STATISTIC_ROWS_PRIVATE.json',rows);dump(H/'GROUP_STATISTIC_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
