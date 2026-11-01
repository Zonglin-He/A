import sys,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round31'),str(ROOT/'research/round27')]
from common31 import *
from evaluate27 import ious
out=BASE/'masscontrol_attempt001';verify(out);records=read(EXP/'ALL_INFLUENCE_ROWS_PRIVATE.json');lookup={r['id']:dict(r,split=split) for split in ['A208','DEV64','R64','S64'] for r in rows(split)};mass={};raw=[]
for x in read(out/'MODEL_SPECS_PRIVATE.json'):
 p=load_npz(out/'predictions_PRIVATE'/(x['name']+'.npz'))
 for rid,q in zip(p['row_ids_PRIVATE'].tolist(),p['queries']):
  r=lookup[rid];v=float(ious(r,feature(r)['boxes'][int(q)][None])[0]);mass[x['platform'],x['removed_count_matched'],rid]=v;raw.append({'model':x['name'],'platform':r['platform'],'target_alias':r['alias'],'target_id_PRIVATE':rid,'split':r['split'],'iou':v,'query_PRIVATE':int(q)})
summary=[];detail=[];ss=read(EXP/'INFLUENCE19_SUMMARY.json')
for x in ss:
 rr=[r for r in records if r['feedback_alias']==x['feedback_alias']];rec={'feedback_alias':x['feedback_alias'],'platform':x['platform'],'removed_preferences':x['removed_preferences'],'uniform_coefficient':({'drone':10,'quad':12}[x['platform']]-x['removed_preferences'])/{'drone':10,'quad':12}[x['platform']],'subsets':{}}
 for r in rr:
  v=mass[x['platform'],x['removed_preferences'],r['target_id_PRIVATE']];detail.append({'feedback_alias':x['feedback_alias'],'platform':x['platform'],'target_alias':r['target_alias'],'split':r['split'],'loo_delta_remove_pp':r['delta_remove_pp'],'uniform_mass_delta_pp':100*(v-r['all_iou']),'loo_minus_uniform_mass_pp':100*(r['minus_iou']-v),'is_removed_observation':r['is_removed_observation']})
 for split in ['A208','DEV64','R64','S64','R64_S64']:
  vals=[r for r in rr if not r['is_removed_observation'] and (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']==split)];mv=[mass[x['platform'],x['removed_preferences'],r['target_id_PRIVATE']] for r in vals];rec['subsets'][split]={'n':len(vals),'loo_delta_remove_pp':float(np.mean([r['delta_remove_pp'] for r in vals])),'uniform_mass_delta_pp':float(np.mean([100*(v-r['all_iou']) for r,v in zip(vals,mv)])),'loo_minus_uniform_mass_pp':float(np.mean([100*(r['minus_iou']-v) for r,v in zip(vals,mv)]))}
 summary.append(rec)
dump(EXP/'MASS_CONTROL_ALL_ROWS_PRIVATE.json',raw);dump(EXP/'LOO_VS_MASS_ROWS.json',detail);dump(EXP/'LOO_VS_MASS_SUMMARY.json',summary)
for r in summary:print(r['feedback_alias'],r['subsets']['S64'])
