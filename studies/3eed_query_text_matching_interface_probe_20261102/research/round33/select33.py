import sys,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round33'),str(ROOT/'research/round27')]
from common33 import *
from evaluate27 import ious
out=BASE/'interface_screen_attempt001';verify(out);fit=read(out/'FIT_RECEIPTS.json');records=[];table=[]
for layer in ['query_mlp','text_mlp']:
 for lr in [.0001,.0005]:
  name=layer+'_lr'+str(lr).replace('.','p');pm={}
  for pl in ['drone','quad']:
   rr=[r for r in rows('DEV64') if r['platform']==pl];p=load_npz(out/'predictions_PRIVATE'/(pl+'_'+name+'_DEV64.npz'));assert p['row_ids_PRIVATE'].tolist()==[r['id'] for r in rr];values=[]
   for r,q in zip(rr,p['queries']):
    v=float(ious(r,feature(r)['boxes'][int(q)][None])[0]);values.append(v);records.append({'alias':r['alias'],'platform':pl,'layer':layer,'lr':lr,'iou':v,'q_PRIVATE':int(q)})
   pm[pl]=float(np.mean(values)*100)
  f=[x for x in fit if x['layer']==layer and x['lr']==lr];B=sum(x['B_pairs_satisfied'] for x in f);F=sum(x['B_final_positive'] for x in f);table.append({'name':name,'layer':layer,'lr':lr,'platform_mIoU_percent':pm,'balanced_mIoU_percent':float(np.mean(list(pm.values()))),'B_pairs_satisfied':B,'B_final_positive':F,'feasible':B>=5 and F>=4})
def choose(t):return sorted(t,key=lambda x:(-x['balanced_mIoU_percent'],x['layer'],x['lr']))[0] if t else None
selected=choose([x for x in table if x['feasible']]);besttext=choose([x for x in table if x['layer']=='text_mlp']);bestquery=choose([x for x in table if x['layer']=='query_mlp']);lock={'selected_feasible_primary':selected,'besttext_unconstrained_diagnostic':besttext,'bestquery_unconstrained_diagnostic':bestquery,'overall_unconstrained_diagnostic':choose(table),'text_feasible_candidate_exists':any(x['layer']=='text_mlp' and x['feasible'] for x in table),'only_exposed_DEV64_selection':True,'R_S_T_U_orFINALlabels_not_used':True,'no_further_sweep_or_budget_extension':True,'source_R30_baseline_is_selected':selected is not None and selected['layer']=='query_mlp' and selected['lr']==.0005,'fit_prediction_seal_sha256':sha(out/'SEALED.json')};dump(EXP/'DEV_INTERFACE_TABLE.json',table);dump(EXP/'DEV_INTERFACE_ROWS_PRIVATE.json',records);dump(EXP/'SELECTED_INTERFACE_LOCK.json',lock);print(json.dumps({'table':table,'selection':lock},indent=2))
