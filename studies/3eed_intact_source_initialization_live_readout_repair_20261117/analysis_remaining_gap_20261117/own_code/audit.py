"""Independent raw-cache checks for error partitions and aggregate claims."""
from common import *
import numpy as np,collections
def main():
 check();rows=rd(H/'ALL_ROWS_PRIVATE.json');reg=rd(P/'INPUT_REGISTRY_PRIVATE.json');assert len(rows)==len(reg)==1798;index={(r['source'],r['target'],r['id_PRIVATE']):r for r in rows};n=0;values=0;maxerr=0.;Aheads=0;Fheads=0;flows=collections.Counter();new={}
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;r=index[(src,tgt,sid)];a=np.load(B/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));ev=np.load(B/name/'evidence_PRIVATE'/(sid+'.npz'));y=np.load(ROOT/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ass=ev['assignment'];S,C=ev['S'],ev['C'];qA=int(np.argmax(p['intact_live_A']));qF=int(np.argmax(p['intact_live_F']));Aheads+=int(np.isin(ass[qA],ev['lose']));Fheads+=int(np.isin(ass[qF],ev['lose']));assert abs(r['oracle']-max(y))<1e-15
  for arm in ['intact_live_A','intact_live_F','tta_A','tta_F','AB_mean','direct_A','direct_F','wrong_A','wrong_F']:
   assert abs(float(y[np.argmax(p[arm])])-r['IoU_'+arm])<1e-15;values+=1
  dd=np.load(H/'readouts_PRIVATE'/name/(sid+'.npz'));assert np.isfinite(dd['scores']).all() and np.allclose(dd['scores'].sum(1),1,atol=1e-12)
  for arm,prob in zip(dd['arm_names'],dd['scores']):assert abs(float(y[np.argmax(prob)])-r['IoU_'+arm])<1e-15;values+=1
  for arm in ['intact_live_A','tta_A','tta_F']:
   for threshold in [.25,.5]:
    actual=y[np.argmax(p[arm])];eligible=set(np.where(y>threshold)[0]);supported={int(j) for j in eligible if S[ass[j]]};retained={int(j) for j in eligible if C[ass[j]]}
    category=('pass' if actual>threshold else ('no_A256_good_box' if not eligible else ('good_box_but_no_B' if not r['has_B'] else ('good_box_outside_S' if not supported else ('good_box_in_S_outside_C' if not retained else 'good_box_in_C_but_wrong_rank')))))
    assert category==r['flow_'+arm+'_'+str(threshold)];flows[(arm,threshold,category)]+=1
  n+=1
 assert n==1798 and Aheads==0 and Fheads==2
 old=rd(P/'scores_attempt001/AGGREGATE_ROWS.json');agg=rd(H/'AGGREGATE_ROWS.json')
 for reported in agg:
  per=[]
  for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
   rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];w=[r['sampling_weight'] if reported['aggregation']=='design_weighted' else 1 for r in rr];y=[r[reported['arm']] for r in rr];per.append([sum(a*b for a,b in zip(w,y))/sum(w)*100,sum(a*(b>.25) for a,b in zip(w,y))/sum(w)*100,sum(a*(b>.5) for a,b in zip(w,y))/sum(w)*100])
  truth=np.mean(per,axis=0);actual=np.array([reported[k] for k in ['mIoU','Acc25','Acc50']]);err=float(abs(truth-actual).max());assert err<1e-10;maxerr=max(maxerr,err)
  if reported['arm'].startswith('IoU_'):
   match=[r for r in old if r['aggregation']==reported['aggregation'] and r['group']=='all_six_equal' and 'IoU_'+r['arm']==reported['arm']]
   if match:
    prev=np.array([match[0][k] for k in ['mIoU_percent','Acc25_percent','Acc50_percent']]);assert max(abs(prev-actual))<1e-10
 for arm in ['intact_live_A','tta_A','tta_F']:
  for t in [.25,.5]:assert sum(v for (a,th,c),v in flows.items() if a==arm and th==t)==1798
 sourcepin=rd(P/'SOURCE_STATE_PIN.json');assert all(sha(ROOT/p)==h for p,h in sourcepin['files'].items());out={'status':'PASS','n':n,'independently_checked_task_scalars':values,'max_scalar_aggregation_error':maxerr,'existing_R135_aggregates_reproduced':True,'A_top_challenged':Aheads,'F_top_challenged':Fheads,'all_six_error_partitions_exhaustive':True,'R135_source_weights_and_original_closed_state_unchanged':True,'FINAL_access':False};dump(H/'AUDIT.json',out);print(out,flush=True)
if __name__=='__main__':main()
