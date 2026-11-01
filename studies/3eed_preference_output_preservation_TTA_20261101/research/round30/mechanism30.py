import sys,numpy as np,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import *
def main():
 fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');lookup={r['id']:r for r in rows()};names={'preference_only':'b0','keep_selected':read(EXP/'SELECTED_BETA_LOCK.json')['selected']['name']};summary={};records=[];task={r['id_PRIVATE']:r for r in read(EXP/'ALL336_TASK_ROWS_PRIVATE.json') if r['split']=='A208'}
 for arm,name in names.items():
  cc=collections.Counter()
  for f in fb:
   r=lookup[f['id_PRIVATE']];p=load_npz(BASE/'screen_attempt001'/(r['platform']+'_'+name+'_A')/'predictions_PRIVATE'/(r['id']+'.npz'));s=p['scores'];q=int(p['query']);P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];other=[i for i in range(256) if i not in P and i not in N];pos=float(s[P].max());neg=float(s[N].max());omax=float(s[other].max()) if other else -1;where='positive_group' if q in P else 'negative_group' if q in N else 'other_group_or_unassigned';cc['pairs']+=1;cc['pair_satisfied']+=pos>neg;cc[where]+=1
   if f['label']=='B':cc['replacement_pairs']+=1;cc['replacement_'+where]+=1
   records.append({'arm':arm,'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'pair_alias':f['pair_alias'],'original_feedback':f['label'],'positive_size':len(P),'negative_size':len(N),'positive_max':pos,'negative_max':neg,'other_max':omax,'pair_satisfied':pos>neg,'positive_globally_highest':pos>max(neg,omax),'final_choice_group':where,'task_delta_vs_source_pp':100*(task[r['id']]['ious'][arm]-task[r['id']]['ious']['source'])})
  summary[arm]=dict(cc)
 dump(EXP/'PAIR_EXECUTION_ROWS_PRIVATE.json',records);dump(EXP/'PAIR_EXECUTION_SUMMARY.json',summary)
 # Only after sealed evidence review, join categories with pair execution, not training.
 keys=read(EXP.parent/'input/EVIDENCE22_KEY_PRIVATE.json');labels=read(BASE/'evidence_review22_attempt001/ANNOTATIONS.json');ft={(f['id_PRIVATE'],f['pair_alias']):f for f in fb};typed=[]
 for k,l in zip(keys,labels):
  f=ft[k['id_PRIVATE'],k['pair_alias']];matched=[x for x in records if x['id_PRIVATE']==k['id_PRIVATE'] and x['pair_alias']==k['pair_alias']];typed.append({'case':k['case'],'alias':k['alias'],'platform':k['platform'],'evidence_type':l['available_evidence_type'],'feedback':f['label'],'execution':{x['arm']:{z:x[z] for z in ['pair_satisfied','final_choice_group','task_delta_vs_source_pp']} for x in matched}})
 dump(EXP/'EVIDENCE22_EXECUTION_ROWS.json',typed);strata={}
 for t in sorted({x['evidence_type'] for x in typed}):
  rr=[x for x in typed if x['evidence_type']==t];strata[t]={'n_pairs':len(rr),'unique_observations':len({x['alias'] for x in rr}),'by_arm':{arm:{'pairs_satisfied':sum(x['execution'][arm]['pair_satisfied'] for x in rr),'final_positive':sum(x['execution'][arm]['final_choice_group']=='positive_group' for x in rr),'mean_A_observation_delta_pp_pair_weighted':float(np.mean([x['execution'][arm]['task_delta_vs_source_pp'] for x in rr]))} for arm in names},'not_type_isolated_transfer':True}
 dump(EXP/'EVIDENCE22_STRATA_SUMMARY.json',strata);print(json.dumps(summary));print(json.dumps(strata))
if __name__=='__main__':main()
