import sys,numpy as np,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round29'))
from common29 import *

def main():
 fb=read(BASE/'direct208_attempt001/FEEDBACK_PRIVATE.json');lookup={r['id']:r for r in rows()[:208]};task={x['id_PRIVATE']:x for x in read(EXP/'ALL272_TASK_ROWS_PRIVATE.json') if x['split']=='A208'};rr=[];summary={}
 for arm,folder in [('probability','ranknet_attempt001'),('logodds','ranknet_logodds_attempt001')]:
  cc=collections.Counter()
  for f in fb:
   r=lookup[f['id_PRIVATE']];pred=load_npz(BASE/folder/'A208_predictions/predictions_PRIVATE'/(r['id']+'.npz'));s=pred['ranknet_group_scores'];P=f['positive_group_PRIVATE'];N=f['negative_group_PRIVATE'];q=int(pred['queries'][pred['arm_names'].tolist().index('ranknet_group')]);pool=read(BASE/'groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];pos=float(s[P].max());neg=float(s[N].max());other=[i for i in range(256) if i not in P and i not in N];othermax=float(s[other].max()) if other else -1;where='positive_group' if q in P else 'negative_group' if q in N else 'other_group_or_unassigned';cc['pairs']+=1;cc['pair_satisfied']+=pos>neg;cc[where]+=1
   if f['label']=='B':cc['replacement_pairs']+=1;cc['replacement_'+where]+=1
   rr.append({'alias':r['alias'],'platform':r['platform'],'pair_alias':f['pair_alias'],'loss':arm,'label':f['label'],'positive_group_size':len(P),'negative_group_size':len(N),'positive_max':pos,'negative_max':neg,'other_max':othermax,'pair_satisfied':pos>neg,'positive_globally_beats_others':pos>max(neg,othermax),'final_choice_group':where,'positive_source_representative_PRIVATE':max(P,key=lambda i:(float(feature(r)['scores'][i]),-i)),'final_q_PRIVATE':q})
  summary[arm]=dict(cc)
 dump(EXP/'PAIR_EXECUTION_ROWS_PRIVATE.json',rr);dump(EXP/'PAIR_EXECUTION_SUMMARY.json',summary)
 # Preserve scene-level results for transfer, all methods and all frames.
 tr=[x for x in read(EXP/'ALL272_TASK_ROWS_PRIVATE.json') if x['split']=='R64'];scenes=[]
 for p in ['drone','quad']:
  for j,s in enumerate(sorted({x['scene_PRIVATE'] for x in tr if x['platform']==p})):
   rs=[x for x in tr if x['platform']==p and x['scene_PRIVATE']==s];scenes.append({'platform':p,'scene_alias':f'{p}_s{j}','n':len(rs),'mIoU_percent':{a:float(np.mean([r['ious'][a] for r in rs])*100) for a in rs[0]['q']}})
 dump(EXP/'R64_SCENE_RESULTS.json',scenes);print(json.dumps(summary));print(json.dumps(scenes))
if __name__=='__main__':main()
