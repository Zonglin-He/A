import sys,numpy as np,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round29'),str(ROOT/'research/round27')]
from common29 import *
from evaluate27 import ious

def getpref(v):
 a=[]
 for order in ['normal','reversed']:
  z=v[order]['logits'];ka,kb=('A','B') if order=='normal' else ('B','A');a.append({'d':z[kb]-z[ka],'uA':z[ka]-z['U'],'uB':z[kb]-z['U']})
 mean=np.mean([x['d'] for x in a]);label='B' if all(min(x['d'],x['uB'])>=.5 for x in a) else 'A' if all(min(-x['d'],x['uA'])>=.5 for x in a) else 'U';raw='B' if mean>1e-7 else 'A' if mean< -1e-7 else 'U';strength=min(min(x['d'],x['uB']) for x in a) if label=='B' else min(min(-x['d'],x['uA']) for x in a) if label=='A' else 0
 return {'label':label,'order_mean_AB':raw,'physical_logits':a,'strength':float(strength)}

def main():
 verify(BASE/'fullpair208_attempt001');verify(BASE/'groups208_attempt001');verify(BASE/'blind_review24_attempt001');out=BASE/'direct208_attempt001';out.mkdir(exist_ok=False);(out/'predictions_PRIVATE').mkdir();feedback=[];pairrows=[];rr=rows()[:208];oldrec={(x['id_PRIVATE'],x['pair_alias']):x for x in read(EXP/'PAIR154_ROWS_PRIVATE.json')};r28={x['alias']:x for x in read(ROOT/'research/round28/experiment/FUSION_ROWS_PRIVATE.json')}
 for r in rr:
  z=feature(r);q0=int(z['scores'].argmax());g=read(BASE/'groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'));gg=g['groups_PRIVATE'];gps={x['pair_alias']:x for x in g['pairs']};pairs=old(r,'pairs')['pairs'];fp=read(BASE/'fullpair208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['pairs'];selection={'source':q0,'legacy_G':r28[r['alias']]['q']['legacy_G'],'source_anchor':r28[r['alias']]['q']['source_anchor_zero'],'old_pair_group':q0,'full_pair_query':q0,'full_pair_group':q0,'full_pair_ordermean_query':q0};options=collections.defaultdict(list)
  for p,x in zip(pairs,fp):
   assert p['pair_alias']==x['pair_alias'];pp=getpref(x['full']);gp=gps[p['pair_alias']];a,b=map(str,p['mask_indices_PRIVATE']);oldlabel=oldrec[r['id'],p['pair_alias']]['policies']['order_agree_ABU_margin05'];pairrows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'pair_alias':p['pair_alias'],'qA_PRIVATE':q0,'qB_PRIVATE':p['qB_PRIVATE'],'mask_A_PRIVATE':int(a),'mask_B_PRIVATE':int(b),'group_eligible':gp['eligible'],**pp})
   if pp['label']=='B':options['full_pair_query'].append((pp['strength'],z['scores'][p['qB_PRIVATE']],p['qB_PRIVATE']))
   if pp['order_mean_AB']=='B':options['full_pair_ordermean_query'].append((np.mean([d['d'] for d in pp['physical_logits']]),z['scores'][p['qB_PRIVATE']],p['qB_PRIVATE']))
   if gp['eligible']:
    rep=max(gg[b],key=lambda i:(float(z['scores'][i]),-i))
    if pp['label']=='B':options['full_pair_group'].append((pp['strength'],float(z['scores'][rep]),rep))
    if oldlabel=='B':options['old_pair_group'].append((1,float(z['scores'][rep]),rep))
    if pp['label']!='U':
     positive,negative=(b,a) if pp['label']=='B' else (a,b);feedback.append({'id_PRIVATE':r['id'],'alias':r['alias'],'platform':r['platform'],'pair_alias':p['pair_alias'],'positive_group_PRIVATE':gg[positive],'negative_group_PRIVATE':gg[negative],'positive_mask_PRIVATE':int(positive),'negative_mask_PRIVATE':int(negative),'label':pp['label'],'weight':1.,'strength':pp['strength']})
  for name,opt in options.items():selection[name]=int(max(opt)[2])
  q=np.array(list(selection.values()));np.savez_compressed(out/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(selection)),queries=q,boxes=z['boxes'][q])
 dump(out/'PAIR_PREFERENCES_PRIVATE.json',pairrows);dump(out/'FEEDBACK_PRIVATE.json',feedback);seal(out,'R29_same_feedback_direct_group_rerank_predictions_before_task_scores',n=208,feedback_pairs=len(feedback),frozen_boxes=True,optimizer_steps=0)
 result=[]
 for r in rr:
  z=load_npz(out/'predictions_PRIVATE'/(r['id']+'.npz'));v=ious(r,z['boxes']);result.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'q':dict(zip(z['arm_names'],map(int,z['queries']))),'ious':dict(zip(z['arm_names'],map(float,v)))})
 dump(EXP/'DIRECT208_ROWS_PRIVATE.json',result);summary={}
 for p in ['drone','quad']:
  rs=[x for x in result if x['platform']==p];summary[p]={}
  for arm in rs[0]['q']:
   v=np.array([x['ious'][arm] for x in rs]);d=v-np.array([x['ious']['source'] for x in rs]);summary[p][arm]={'n':len(rs),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'changed':sum(x['q'][arm]!=x['q']['source'] for x in rs)}
 dump(EXP/'DIRECT208_SUMMARY.json',summary)
 blind=[];keys=read(EXP.parent/'input/BLIND24_KEY_PRIVATE.json');labs=read(BASE/'blind_review24_attempt001/ANNOTATIONS.json');pp={(x['id_PRIVATE'],x['pair_alias']):x for x in pairrows}
 for k,a in zip(keys,labs):
  x=pp[k['id_PRIVATE'],k['pair_alias']];target=next((c for c in ['A','B'] if a['target'] in k['labels_to_mask_PRIVATE'] and k['labels_to_mask_PRIVATE'][a['target']]==k['mask_'+c+'_PRIVATE']),a['target']);blind.append({'case':a['case'],'platform':k['platform'],'target_AB':target,'confidence':a['confidence'],'accepted':x['label'],'order_mean_AB':x['order_mean_AB'],'group_eligible':x['group_eligible']})
 dump(EXP/'FULLPAIR_BLIND24_ROWS.json',blind);ss={}
 for arm in ['accepted','order_mean_AB']:
  c=collections.Counter()
  for x in blind:
   t=x['target_AB'];p=x[arm]
   if t=='U':c['unresolved']+=1
   elif p=='U':c['abstained']+=1
   else:
    c['correct' if t==p else 'wrong']+=1
    if p=='B':c['B_correct' if t=='B' else 'B_wrong']+=1
  ss[arm]=dict(c)
 dump(EXP/'FULLPAIR_BLIND24_SUMMARY.json',ss);print('BLIND',json.dumps(ss));print('DIRECT',json.dumps(summary));print('FEEDBACK',dict(collections.Counter((x['platform'],x['label']) for x in feedback)))
if __name__=='__main__':main()
