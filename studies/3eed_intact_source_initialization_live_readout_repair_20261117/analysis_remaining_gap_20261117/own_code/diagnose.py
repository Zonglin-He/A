"""GT-assisted mechanism diagnosis only; no deployment selection or fitting."""
from common import *
import numpy as np,collections
ARMS=['intact_live_A','intact_live_F','tta_A','tta_F','AB_mean','direct_A','direct_F','wrong_A','wrong_F','B_teacher','base_A_no_competition','base_A_no_late_no_competition','full_A_no_late_fixed_competition']
LABEL={'intact_live_A':'A','intact_live_F':'F','tta_A':'TA','tta_F':'TF','B_teacher':'B','base_A_no_competition':'base','base_A_no_late_no_competition':'nolate_base','full_A_no_late_fixed_competition':'nolate_full'}
def flow(y,q,t,hasB,S,C,ass):
 good=y>t
 if y[q]>t:return 'pass'
 if not good.any():return 'no_A256_good_box'
 if not hasB:return 'good_box_but_no_B'
 if not (good&S[ass]).any():return 'good_box_outside_S'
 if not (good&C[ass]).any():return 'good_box_in_S_outside_C'
 return 'good_box_in_C_but_wrong_rank'
def metrics(rows,key,weighted):
 y=np.array([r[key] for r in rows]);w=np.array([r['sampling_weight'] if weighted else 1. for r in rows]);return {'mIoU':float(np.average(y,weights=w)*100),'Acc25':float(np.average(y>.25,weights=w)*100),'Acc50':float(np.average(y>.5,weights=w)*100)}
def main():
 check();seal=rd(H/'READOUTS_SEALED.json');assert seal['n']==1798
 for p,h in seal['files'].items():assert sha(H/p)==h,p
 reg=rd(P/'INPUT_REGISTRY_PRIVATE.json');pr={(r['source'],r['target'],r['id_PRIVATE'],r['kind']):r for r in rd(P/'PROCESS_ROWS_PRIVATE.json')};rows=[];flowrows=[];toprows=[];methods=[];paired=[];marginrows=[];conditionals=[];pairtot=collections.Counter();updatecounts=collections.Counter();counts=collections.Counter()
 for e in reg:
  src,tgt=e['source'],e['target'];name=src+'_to_'+tgt;sid=e['row']['id'];D=B/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));ev=np.load(D/'evidence_PRIVATE'/(sid+'.npz'));ps['B_teacher']=ev['pB'];diag=np.load(H/'readouts_PRIVATE'/name/(sid+'.npz'));ps.update(dict(zip(diag['arm_names'],diag['scores'])));y=np.load(ROOT/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];rec=rd(D/'fit_records_PRIVATE'/(sid+'.json'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S,C=ev['S'],ev['C'];q={k:int(ps[k].argmax()) for k in ARMS};qa,qf=q['intact_live_A'],q['intact_live_F'];ga,gf=ass[qa],ass[qf];w,l=ev['win'],ev['lose'];hasB=rec['has_B'];good=y>.25;ggood=np.array([good[g].any() for g in groups]);best=float(y.max());gm={v:np.array([np.log(max(float(ps[v][g].max()),1e-300)) for g in groups]) for v in ['intact_live_A','intact_live_F','B_teacher']};rep=np.array([g[ps['intact_live_F'][g].argmax()] for g in groups]);dy=y[rep[w]]-y[rep[l]];assert not (l==ga).any(),'AB-agreement cannot have A global top as loser'
  r={k:e[k] for k in ['source','target','sampling_weight','frame_PRIVATE','content_group_PRIVATE']}|{'id_PRIVATE':sid,'scene_PRIVATE':e['row']['scene'],'has_B':hasB,'active':rec['active'],'partial_active':rec['partial_active'],'pair_count':len(w),'S_groups':int(S.sum()),'C_groups':int(C.sum()),'groups':len(groups),'oracle':best,'within_A_group_oracle':float(y[groups[ga]].max()),'within_TA_group_oracle':float(y[groups[ass[q['tta_A']]]].max()),'view_oracle_A_B_F':max(float(y[q[v]]) for v in ['intact_live_A','intact_live_F','B_teacher']),'A_top_supported':bool(S[ga]),'A_top_excluded_by_C':bool(S[ga] and not C[ga]),'A_top_pair_challenged':False,'F_top_pair_challenged':bool((l==gf).any()),'A_wrong_recoverable':bool(y[qa]<=.25 and good.any()),'C_has_good':bool((ggood&C).any()),'S_has_good':bool((ggood&S).any()),'B_top_rescues_A_Acc25':bool(hasB and y[qa]<=.25 and y[q['B_teacher']]>.25),'B_top_breaks_A_Acc25':bool(hasB and y[qa]>.25 and y[q['B_teacher']]<=.25),'F_top_rescues_A_Acc25':bool(hasB and y[qa]<=.25 and y[q['intact_live_F']]>.25),'F_top_breaks_A_Acc25':bool(hasB and y[qa]>.25 and y[q['intact_live_F']]<=.25),'B_supports_good_C_over_A_top':bool(hasB and y[qa]<=.25 and (ggood&C&(gm['B_teacher']>gm['B_teacher'][ga]+1e-10)).any()),'pair_better_proxy':int((dy>1e-12).sum()),'pair_worse_proxy':int((dy< -1e-12).sum()),'pair_tied_proxy':int((abs(dy)<=1e-12).sum()),'pair_both_zero_proxy':int(((y[rep[w]]<=1e-12)&(y[rep[l]]<=1e-12)).sum()),'pair_winner_good25':int((y[rep[w]]>.25).sum()),'pair_loser_good25':int((y[rep[l]]>.25).sum())}
  for arm in ARMS:r['IoU_'+arm]=float(y[q[arm]])
  for arm in ['intact_live_A','tta_A','tta_F']:
   for t in [.25,.5]:r['flow_'+arm+'_'+str(t)]=flow(y,q[arm],t,hasB,S,C,ass)
  for arm in ['intact_live_A','intact_live_F','tta_A','tta_F','B_teacher']:
   order=np.argsort(-ps[arm],kind='stable');r['rank_best_IoU_'+arm]=int(np.min(np.flatnonzero(np.isin(order,np.flatnonzero(y>=best-1e-12)))))+1
   r['rank_first_good25_'+arm]=int(np.flatnonzero(good[order])[0])+1 if good.any() else None
   for k in [1,3,5,10,20,50,256]:r['top'+str(k)+'_'+arm]=float(y[order[:k]].max())
  old=np.load(ROOT/'crossplatform/round131/formal'/name/'features_PRIVATE'/(sid+'.npz'));na,nb,coverage=old['n_A'],old['n_B'],old['coverage_B'];assert np.array_equal(S,(na>=4)&(nb>=4)&(coverage>=.5));s1=ggood&(na>=4);s2=s1&(nb>=4);r['unsupported_good_reason']=('not_applicable' if not (r['A_wrong_recoverable'] and hasB and not r['S_has_good']) else ('A_point_support_below4' if not s1.any() else ('B_registered_support_below4' if not s2.any() else 'B_registered_fraction_below0.5')))
  mask=np.load(D/'features_PRIVATE'/(sid+'.npz'))['intact_A_mask'].astype(bool);rt=np.load(D/'route_PRIVATE'/(sid+'.npz'));roles=rt['token_roles'].astype(bool);r['native_token_count']=int(mask.sum());r['has_parsed_subject']=bool(len(roles));r['native_subject_token_overlap']=float((mask&roles[0]).sum()/max(mask.sum(),1)) if len(roles) else None;r['native_overlaps_reference_tokens']=bool(len(roles)>1 and (mask&roles[1:].any(0)).any())
  for arm in ['tta_A','tta_F']:
   process=pr.get((src,tgt,sid,arm));nonzero=bool(process and process['parameter_L2']>0);r[arm+'_nonzero']=nonzero
   if not rec['active']:reason='no_B' if not hasB else 'B_but_no_task'
   else:
    tr=rec['fits'][arm]['trace'];first=tr[0];r[arm+'_initial_objective']=first['before'];r[arm+'_initial_gradient']=first['gradient_L2']
    if nonzero:reason='nonzero_update'
    elif not rec['partial_active'] and first['before']<=1e-12:reason='pair_only_already_satisfied_in_A' if arm=='tta_A' else 'pair_only_zero_task'
    elif first['before']<=1e-12:reason='partial_task_numerically_zero_or_saturated'
    elif max(t['gradient_L2'] for t in tr)<=1e-12:reason='gradient_tiny'
    else:reason='all_proposals_rejected'
   r[arm+'_update_reason']=reason;updatecounts[(arm,reason)]+=1
  if r['A_wrong_recoverable']:
   goodids=np.flatnonzero(good);j=int(goodids[ps['intact_live_A'][goodids].argmax()]);lp0=np.log(np.maximum(ps['intact_live_A'],1e-300));lp1=np.log(np.maximum(ps['tta_A'],1e-300));m0=float(lp0[j]-lp0[qa]);m1=float(lp1[j]-lp1[qa]);r.update(good_vs_old_leader_margin0=m0,good_vs_old_leader_margin_after=m1,good_margin_change=m1-m0,absolute_change_to_initial_gap_ratio=abs(m1-m0)/max(abs(m0),1e-12),good_margin_crosses_old_leader=bool(m1>0))
  rows.append(r)
  if len(rows)%256==0:print('DIAGNOSE',len(rows),flush=True)
 assert len(rows)==1798;dump(H/'ALL_ROWS_PRIVATE.json',rows)
 numeric=['oracle','within_A_group_oracle','within_TA_group_oracle','view_oracle_A_B_F']+['IoU_'+a for a in ARMS]
 for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
  rr=[r for r in rows if r['source']+'_to_'+r['target']==name];src,_,tgt=name.split('_')
  for weighted in [False,True]:
   ag='design_weighted' if weighted else 'raw';weights=np.array([r['sampling_weight'] if weighted else 1 for r in rr]);den=weights.sum()
   for key in numeric:methods.append({'source':src,'target':tgt,'aggregation':ag,'arm':key,'n':len(rr),**metrics(rr,key,weighted)})
   for arm in ['intact_live_A','intact_live_F','tta_A','tta_F','B_teacher']:
    for k in [1,3,5,10,20,50,256]:toprows.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'K':k,**metrics(rr,'top'+str(k)+'_'+arm,weighted)})
   for arm in ['intact_live_A','tta_A','tta_F']:
    for t in [.25,.5]:
     key='flow_'+arm+'_'+str(t)
     for category in ['pass','no_A256_good_box','good_box_but_no_B','good_box_outside_S','good_box_in_S_outside_C','good_box_in_C_but_wrong_rank']:
      mask=np.array([r[key]==category for r in rr]);flowrows.append({'source':src,'target':tgt,'aggregation':ag,'arm':arm,'threshold':t,'category':category,'n':int(mask.sum()),'weighted_or_raw_percent':float(weights[mask].sum()/den*100)})
   for a,c in [('tta_A','intact_live_A'),('tta_F','intact_live_F'),('B_teacher','intact_live_A'),('base_A_no_competition','intact_live_A'),('full_A_no_late_fixed_competition','intact_live_A'),('base_A_no_late_no_competition','base_A_no_competition')]:
    d=np.array([r['IoU_'+a]-r['IoU_'+c] for r in rr]);paired.append({'source':src,'target':tgt,'aggregation':ag,'method':a,'control':c,'gain_pp':float(np.average(d,weights=weights)*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum())})
 aggregate=[]
 for ag in ['raw','design_weighted']:
  for key in numeric:
   mm=[r for r in methods if r['aggregation']==ag and r['arm']==key];aggregate.append({'aggregation':ag,'arm':key,**{m:float(np.mean([r[m] for r in mm])) for m in ['mIoU','Acc25','Acc50']}})
 topagg=[]
 for ag in ['raw','design_weighted']:
  for a in ['intact_live_A','intact_live_F','tta_A','tta_F','B_teacher']:
   for k in [1,3,5,10,20,50,256]:
    mm=[r for r in toprows if r['aggregation']==ag and r['arm']==a and r['K']==k];topagg.append({'aggregation':ag,'arm':a,'K':k,**{m:float(np.mean([r[m] for r in mm])) for m in ['mIoU','Acc25','Acc50']}})
 flowagg=[]
 for ag in ['raw','design_weighted']:
  for arm in ['intact_live_A','tta_A','tta_F']:
   for t in [.25,.5]:
    for cat in sorted({r['category'] for r in flowrows}):
     mm=[r for r in flowrows if r['aggregation']==ag and r['arm']==arm and r['threshold']==t and r['category']==cat];flowagg.append({'aggregation':ag,'arm':arm,'threshold':t,'category':cat,'n':sum(r['n'] for r in mm),'six_equal_percent':float(np.mean([r['weighted_or_raw_percent'] for r in mm]))})
 for filt,rr in [('all',rows),('B', [r for r in rows if r['has_B']]),('A_recoverable_error',[r for r in rows if r['A_wrong_recoverable']]),('A_recoverable_error_with_B',[r for r in rows if r['A_wrong_recoverable'] and r['has_B']]),('A_recoverable_error_good_C',[r for r in rows if r['A_wrong_recoverable'] and r['C_has_good']]),('A_recoverable_error_good_C_actual_update',[r for r in rows if r['A_wrong_recoverable'] and r['C_has_good'] and r['tta_A_nonzero']])]:
  conditionals.append({'subset':filt,'n':len(rr),'B_top_rescues_A':sum(r['B_top_rescues_A_Acc25'] for r in rr),'B_top_breaks_A':sum(r['B_top_breaks_A_Acc25'] for r in rr),'F_top_rescues_A':sum(r['F_top_rescues_A_Acc25'] for r in rr),'F_top_breaks_A':sum(r['F_top_breaks_A_Acc25'] for r in rr),'B_supports_good_C_over_A_top':sum(r['B_supports_good_C_over_A_top'] for r in rr),'A_top_supported':sum(r['A_top_supported'] for r in rr),'A_top_excluded_by_C':sum(r['A_top_excluded_by_C'] for r in rr),'A_top_pair_challenged':0,'F_top_pair_challenged':sum(r['F_top_pair_challenged'] for r in rr)})
 for subset,rr in [('recoverable_actual_update',[r for r in rows if r['A_wrong_recoverable'] and r['tta_A_nonzero']]),('recoverable_goodC_actual_update',[r for r in rows if r['A_wrong_recoverable'] and r['C_has_good'] and r['tta_A_nonzero']])]:
  delta=np.array([r['good_margin_change'] for r in rr]);rat=np.array([r['absolute_change_to_initial_gap_ratio'] for r in rr]);marginrows.append({'subset':subset,'n':len(rr),'margin_improves':int((delta>1e-10).sum()),'margin_worsens':int((delta< -1e-10).sum()),'unchanged':int((abs(delta)<=1e-10).sum()),'median_initial_logit_gap':float(np.median([-r['good_vs_old_leader_margin0'] for r in rr])),'median_margin_change':float(np.median(delta)),'median_absolute_change_to_gap_ratio':float(np.median(rat)),'cross_old_leader':sum(r['good_margin_crosses_old_leader'] for r in rr),'actually_good_TA':sum(r['IoU_tta_A']>.25 for r in rr)})
 summary={'n':len(rows),'B':sum(r['has_B'] for r in rows),'counterfactuals_are_posthoc_not_new_method':True,'absolute_candidate_bins':dict(collections.Counter('zero_overlap' if r['oracle']<1e-6 else ('below25' if r['oracle']<=.25 else ('25to50' if r['oracle']<=.5 else 'above50')) for r in rows)),'conditional_counts':conditionals,'margin_diagnostics':marginrows,'zero_update_reasons':[{'arm':a,'reason':why,'n':n} for (a,why),n in updatecounts.items()],'pair_quality_proxy':{k:sum(r[k] for r in rows) for k in ['pair_count','pair_better_proxy','pair_worse_proxy','pair_tied_proxy','pair_both_zero_proxy','pair_winner_good25','pair_loser_good25']},'unsupported_good_reasons':dict(collections.Counter(r['unsupported_good_reason'] for r in rows if r['unsupported_good_reason']!='not_applicable')),'native_token_alignment_proxy':{'parsed_subject_present':sum(r['has_parsed_subject'] for r in rows),'native_subject_overlap_zero':sum(r['native_subject_token_overlap']==0 for r in rows),'native_overlap_reference_tokens':sum(r['native_overlaps_reference_tokens'] for r in rows),'not_ground_truth_subject_role_audit':True},'physical_RGB_instance_correspondence_not_verified':True,'FINAL_access':False}
 for n,x in [('METHOD_ROWS',methods),('AGGREGATE_ROWS',aggregate),('TOPK_ROWS',toprows),('TOPK_AGGREGATE',topagg),('FLOW_ROWS',flowrows),('FLOW_AGGREGATE',flowagg),('PAIRED_ROWS',paired),('SUMMARY',summary)]:dump(H/(n+'.json'),x)
 dump(H/'ANALYSIS_SEALED.json',{'n':1798,'files':{str(p.relative_to(H)):sha(p) for p in H.glob('*.json') if p.name not in ['STATE.json','RUNNING.json','ANALYSIS_SEALED.json']},'new_training':False,'GPU':False});print(summary,flush=True)
if __name__=='__main__':main()
