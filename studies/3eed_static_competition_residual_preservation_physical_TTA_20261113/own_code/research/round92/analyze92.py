"""Whole-set paired outcomes and exact score-margin decomposition; no GT selection rule."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round92';P=R/'crossplatform/round92/prediction336_attempt001';read=lambda p:json.loads(p.read_text())
def dump(n,x):(H/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=read(P/'SEALED.json')['arm_names'];rows=read(H/'experiment/DEV336_PRIVATE.json');method=[];comp=[];drift=[];updates=[];trials=[]
for a in arms:
 method.append({'platform':'balanced','arm':a,'mIoU_percent':su['balanced_mIoU_percent'][a],'Acc25_percent':float(np.mean([su[p][a]['Acc25_percent'] for p in ['drone','quad']])),'Acc50_percent':float(np.mean([su[p][a]['Acc50_percent'] for p in ['drone','quad']]))})
 for p in ['drone','quad']:method.append({'platform':p,'arm':a,**{k:su[p][a][k] for k in ['n','mIoU_percent','Acc25_percent','Acc50_percent','unknown_n','fallback_n','no_parameter_update_n']}})
pair=[('static_residual_formal_bt',r,'primary') for r in ['R91_frozen','R91_mean3','R91_feature3','R91_bt_dynamic','R91_raw_dynamic','same_state_bt_static_residual','static_residual_formal_wrong_bt','R59','R80_raw','R82_bt']]
for kind in ['raw','bt','wrong_raw','wrong_bt']:pair.extend([('same_state_'+kind+'_static_residual','R91_'+kind+'_dynamic','same_state_residual_hold'),('same_state_'+kind+'_residual_only','R91_frozen','same_state_residual_change_only')])
for a,b,kind in pair:
 for pl in ['balanced','drone','quad']:
  rr=tasks if pl=='balanced' else [t for t in tasks if t['platform']==pl];dv=np.array([t['IoU'][a]-t['IoU'][b] for t in rr]);comp.append({'component':kind,'arm':a,'reference':b,'platform':pl,'n':len(rr),'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv< -1e-10).sum()),'unchanged_n':int((abs(dv)<=1e-10).sum()),'query_changes':sum(t['queries'][a]!=t['queries'][b] for t in rr),'largest_positive_contribution_pp':float(max(0,dv.max())*100/len(dv)),'largest_negative_contribution_pp':float(min(0,dv.min())*100/len(dv))})
for t,r in zip(tasks,rows):
 m=read(P/'records_PRIVATE'/(r['id']+'.json'));z=np.load(P/'components_PRIVATE'/(r['id']+'.npz'));b0=z['view0_base'];r0=z['view0_residual'];i=t['queries']['R91_frozen']
 for kind in ['raw','bt','wrong_raw','wrong_bt']:
  k=m['same_state_'+kind+'_static_residual']['feature_key'];b1=z[k+'_base'];r1=z[k+'_residual'];s=b1+r1;ss=s.copy();ss[i]=-np.inf;j=int(ss.argmax());bm0=b0[i]-b0[j];rm0=r0[i]-r0[j];db=b1[i]-b1[j]-bm0;dr=r1[i]-r1[j]-rm0;err=abs(((b1+r1)[i]-(b1+r1)[j])-(bm0+rm0+db+dr));assert err<1e-10;drift.append({'row':t['row'],'platform':r['platform'],'kind':kind,'base_margin_zero':float(bm0),'residual_margin_zero':float(rm0),'base_margin_change':float(db),'residual_margin_change':float(dr),'total_margin_change':float(db+dr),'exact_margin_decomposition_error':float(err),'base_centered_drift_L2':float(np.linalg.norm((b1-b1.mean())-(b0-b0.mean()))),'residual_centered_drift_L2':float(np.linalg.norm((r1-r1.mean())-(r0-r0.mean()))),'dynamic_vs_zero_IoU_change_pp':100*(t['IoU']['R91_'+kind+'_dynamic']-t['IoU']['R91_frozen']),'held_residual_vs_zero_IoU_change_pp':100*(t['IoU']['same_state_'+kind+'_static_residual']-t['IoU']['R91_frozen']),'postseal_not_gate':True})
 for arm,rr in m.items():
  if rr['kind']!='formal_static_residual_acceptance':continue
  updates.append({'row':t['row'],'platform':r['platform'],'arm':arm,**{k:v for k,v in rr.items() if k not in ['trials','feature_key','state_path']},'attempt_count':len(rr['trials'])})
  for i0,x in enumerate(rr['trials']):trials.append({'row':t['row'],'platform':r['platform'],'arm':arm,'attempt':i0+1,**{k:v for k,v in x.items() if k not in ['feature_key','state_path']}})
scene=[];sc=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for p in ['drone','quad']:
 names=sorted({x['scene_PRIVATE'] for x in sc if x['platform']==p})
 for x in [x for x in sc if x['platform']==p]:
  for a in arms:scene.append({'platform':p,'scene_alias':p[0].upper()+'_S'+str(names.index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':a,'mIoU_percent':x['mIoU_percent'][a]})
chosen=[];why={}
def take(i,reason):
 if i not in chosen:chosen.append(i)
 why.setdefault(i,[]).append(reason)
oldcases=read(R/'research/round91/experiment/GOOD_BAD_CASES_PRIVATE.json')
for c in oldcases:
 if c['case'] in ['CASE001','CASE006','CASE010']:
  take(next(i for i,r in enumerate(rows) if r['id']==c['row']['id']),'fixed_R91_'+c['case'])
for a,b in [('static_residual_formal_bt','R91_frozen'),('static_residual_formal_bt','R91_bt_dynamic'),('same_state_bt_static_residual','R91_bt_dynamic'),('same_state_raw_static_residual','R91_raw_dynamic'),('same_state_raw_static_residual','R91_frozen')]:
 dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tasks])
 for i in [i for i in np.argsort(-dv,kind='stable') if dv[i]>1e-10][:2]:take(int(i),'gain_'+a+'_vs_'+b)
 for i in [i for i in np.argsort(dv,kind='stable') if dv[i]<-1e-10][:2]:take(int(i),'harm_'+a+'_vs_'+b)
cases=[{'case':'CASE'+str(j+1).zfill(3),'row':rows[i],'task':tasks[i],'reasons':why[i]} for j,i in enumerate(chosen)];(H/'experiment/GOOD_BAD_CASES_PRIVATE.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n');casepublic=[{'case':x['case'],'platform':x['task']['platform'],'reasons':';'.join(x['reasons']),**{a+'_IoU':x['task']['IoU'][a] for a in arms}} for x in cases]
for n,x in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',comp),('ALL_SCENE_ROWS',scene),('SCORE_MARGIN_DECOMPOSITION_ROWS',drift),('ALL_UPDATE_ROWS',updates),('ALL_ACCEPTANCE_TRIAL_ROWS',trials),('GOOD_BAD_CASE_ROWS',casepublic),('HELDOUT_ROWS',read(H/'independent_native_audit_attempt001/HELDOUT_ROWS.json'))]:dump(n,x)
s=su['balanced_mIoU_percent'];a='static_residual_formal_bt';d={'state':'R92_COMPLETED_FIXED_R91_STATIC_RESIDUAL_AND_CONSISTENT_ACCEPTANCE_DEVELOPMENT_EVALUATED','mainline':'R91_R76_full_disc','primary':a,'primary_mIoU_percent':s[a],'gains_pp':{k:s[a]-s[k] for k in ['R91_frozen','R91_mean3','R91_feature3','R91_bt_dynamic','same_state_bt_static_residual','static_residual_formal_wrong_bt']},'same_state_BT_static_vs_dynamic_pp':s['same_state_bt_static_residual']-s['R91_bt_dynamic'],'same_state_BT_residual_only_vs_zero_pp':s['same_state_bt_residual_only']-s['R91_frozen'],'own_zero_update_gain_positive':s[a]>s['R91_frozen'],'independent_confirmation':False,'robustly_adopted':False,'architecture_retained_as_user_selected_mainline':True,'source_training':False,'FINAL_access':False,'no_follow_on_queued':True};dump('DECISION',d);print(d)
