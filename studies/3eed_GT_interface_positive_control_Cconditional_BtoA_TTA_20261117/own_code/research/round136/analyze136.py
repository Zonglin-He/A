"""GT gradient comparisons are post-seal diagnostics, never an update rule."""
from common136 import *
import numpy as np,collections
def cosine(a,b):
 a=a.astype(float);b=b.astype(float);den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>1e-20 else None
def main():
 check();assert (B136/'method/SEALED.json').exists();reg=rd(H136/'INPUT_REGISTRY_PRIVATE.json');rows=[];oldanalysis={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(R136/'research/round135/analysis_remaining_gap_20261117/ALL_ROWS_PRIVATE.json')};oracle={(r['source'],r['target'],r['id_PRIVATE']):r for r in rd(H136/'ORACLE_ALL_ROWS_PRIVATE.json')}
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;ev=np.load(B136/'evidence_PRIVATE'/name/(sid+'.npz'));eligible=bool(ev['eligible']);D=B136/'method'/name;rec=rd(D/'records_PRIVATE'/(sid+'.json'));a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(a['arm_names'],a['scores']));y=np.load(R136/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];C=ev['C'];qA=int(p['Frozen_A'].argmax());ga=ass[qA];good=y>.25;old=oldanalysis[(src,tgt,sid)];r={k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'eligible':eligible,'old_A_recoverable_error':old['A_wrong_recoverable'],'good_C':old['C_has_good'],'A_top_in_C':bool(C[ga]),'A_TTA_improvement':float(y[p['C_B_A'].argmax()]-y[qA]),'C_B_actual_update':False,'wrong_actual_update':False,'C_B_proposed_A_leader_demoted':False}
  r['GT_best_in_C']=bool(((y>=y.max()-1e-12)&C[ass]).any());r['GT3_IoU_delta']=oracle[(src,tgt,sid)]['IoUs']['GT3_diagnostic']-oracle[(src,tgt,sid)]['IoUs']['Frozen_A']
  if eligible:
   gt=np.load(B136/'oracle'/name/'states_PRIVATE'/(sid+'.npz'))['gradient1'];new=np.load(D/'states_PRIVATE'/(sid+'.npz'));r['C_B_actual_update']=bool(new['C_B_A_delta'].any());r['cos_GT_C_B']=cosine(gt,new['C_B_A_gradient1']);r['C_B_accepted_steps']=sum(t['accepted'] for t in rec['fits']['C_B_A']['trace']);r['C_B_initial_loss']=rec['fits']['C_B_A']['trace'][0]['before'];r['C_B_final_loss']=rec['fits']['C_B_A']['trace'][-1]['after']
   if rec['fits']['wrong_C_B_A'].get('reused'):r['cos_GT_wrong']=r['cos_GT_C_B']
   else:r['cos_GT_wrong']=cosine(gt,new['wrong_C_B_A_gradient1']);r['wrong_actual_update']=bool(new['wrong_C_B_A_delta'].any())
   prev=R136/'crossplatform/round135/formal'/name/'states_PRIVATE'/(sid+'.npz')
   with np.load(prev) as f:
    if 'tta_A_gradient1' in f and (old['tta_A_nonzero'] or old['tta_A_update_reason']=='all_proposals_rejected'):r['cos_GT_R135']=cosine(gt,f['tta_A_gradient1'])
   if C[ga]:
    lp=np.log(np.maximum(p['Frozen_A'],1e-300));mx=np.array([lp[g].max() for g in groups])[C];pred=np.exp(mx-mx.max());pred/=pred.sum();target=ev['target'];local=int(np.flatnonzero(np.flatnonzero(C)==ga)[0]);r['C_B_proposed_A_leader_demoted']=bool(pred[local]>target[local]+1e-12)
   if old['A_wrong_recoverable']:
    ids=np.flatnonzero(good);j=int(ids[p['Frozen_A'][ids].argmax()]);l0=np.log(np.maximum(p['Frozen_A'],1e-300));l1=np.log(np.maximum(p['C_B_A'],1e-300));r['good_margin_delta']=float(l1[j]-l1[qA]-l0[j]+l0[qA]);r['C_B_rescues_Acc25']=bool(y[p['C_B_A'].argmax()]>.25)
  rows.append(r)
 summaries=[]
 for subset,rr in [('eligible',[r for r in rows if r['eligible']]),('common_R135_nonzero_task',[r for r in rows if r.get('cos_GT_R135') is not None]),('recoverable',[r for r in rows if r['eligible'] and r['old_A_recoverable_error']]),('recoverable_goodC',[r for r in rows if r['eligible'] and r['old_A_recoverable_error'] and r['good_C']]),('recoverable_GTbest_inC',[r for r in rows if r['eligible'] and r['old_A_recoverable_error'] and r['GT_best_in_C']])]:
  for key in ['cos_GT_R135','cos_GT_C_B','cos_GT_wrong']:
   vals=[r[key] for r in rr if r.get(key) is not None];summaries.append({'subset':subset,'comparison':key,'n':len(vals),'positive':sum(x>0 for x in vals),'negative':sum(x<0 for x in vals),'median':float(np.median(vals)) if vals else None,'mean':float(np.mean(vals)) if vals else None})
 flow=[]
 for name,rr in [('all',[r for r in rows]),('recoverable',[r for r in rows if r['eligible'] and r['old_A_recoverable_error']]),('recoverable_goodC',[r for r in rows if r['eligible'] and r['old_A_recoverable_error'] and r['good_C']])]:
  m=[r['good_margin_delta'] for r in rr if 'good_margin_delta' in r];flow.append({'subset':name,'n':len(rr),'nonzero_updates':sum(r['C_B_actual_update'] for r in rr),'A_leader_explicitly_demoted_by_C_teacher':sum(r['C_B_proposed_A_leader_demoted'] for r in rr),'good_margin_improves':sum(x>1e-10 for x in m),'good_margin_worsens':sum(x< -1e-10 for x in m),'rescues_Acc25':sum(r.get('C_B_rescues_Acc25',False) for r in rr)})
 scopes=[]
 for name,predicate in [('GTbest_inC',lambda r:r['eligible'] and r['GT_best_in_C']),('GTbest_outsideC',lambda r:r['eligible'] and not r['GT_best_in_C']),('C_has_Acc25good',lambda r:r['eligible'] and r['good_C']),('C_has_no_Acc25good',lambda r:r['eligible'] and not r['good_C'])]:
  total=0.;mtotal=0.;subset=[r for r in rows if predicate(r)]
  for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
   rr=[r for r in rows if r['source']==src and r['target']==tgt];den=sum(r['sampling_weight'] for r in rr);ss=[r for r in rr if predicate(r)];total+=sum(r['GT3_IoU_delta']*r['sampling_weight'] for r in ss)/den*100/6;mtotal+=sum(r['A_TTA_improvement']*r['sampling_weight'] for r in ss)/den*100/6
  scopes.append({'subset':name,'n':len(subset),'GT3_whole_cohort_contribution_pp':total,'C_B_whole_cohort_contribution_pp':mtotal,'GT3_better':sum(r['GT3_IoU_delta']>1e-12 for r in subset),'GT3_worse':sum(r['GT3_IoU_delta']< -1e-12 for r in subset)})
 dump(H136/'GRADIENT_ROWS_PRIVATE.json',rows);dump(H136/'GRADIENT_SUMMARY.json',summaries);dump(H136/'SUPERVISION_FLOW.json',flow);dump(H136/'ORACLE_SCOPE_BREAKDOWN.json',scopes);print({'gradient_alignment':summaries,'flow':flow,'oracle_scope_breakdown':scopes},flush=True)
if __name__=='__main__':main()
