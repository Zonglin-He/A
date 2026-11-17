"""Historical complete prediction drift in supported versus UNKNOWN scope, GT-free."""
from common140 import *
import numpy as np,collections
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');rows=[];c=collections.Counter();mass=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));eligible=bool(ev['fullreader_ownsource_eligible']);old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(old['arm_names'],old['scores']));S=ev['S'][ev['assignment']];a=int(p['Frozen_A'].argmax());b=int(p['C_B_A'].argmax());r={'source':src,'target':tgt,'id_PRIVATE':sid,'eligible':eligible,'query_changed':a!=b,'A_leader_supported':bool(S[a]),'updated_leader_supported':bool(S[b]),'support_mass_delta':float(p['C_B_A'][S].sum()-p['Frozen_A'][S].sum())};rows.append(r)
  if eligible:
   c['eligible']+=1;mass.append(r['support_mass_delta'])
   if a!=b:
    c['changed']+=1
    c[('S' if S[a] else 'U')+'_to_'+('S' if S[b] else 'U')]+=1
 out={'status':'HISTORICAL_SCOPE_DRIFT_GT_FREE_METADATA_ONLY','counts':dict(c),'eligible_mean_support_mass_change':float(np.mean(mass)),'eligible_median_absolute_support_mass_change':float(np.median(abs(np.array(mass)))),'hypothesis':'partial/support-only teacher can change unknown outputs through shared Up; this does not identify good/bad changes','new_model_or_task_scores_or_GT_read':False,'R140_predictor_unchanged':True,'FINAL_access':False};dump(H/'SCOPE_DRIFT_ROWS_PRIVATE.json',rows);dump(H/'SCOPE_DRIFT_METADATA_ONLY.json',out);print(out,flush=True)
if __name__=='__main__':main()
