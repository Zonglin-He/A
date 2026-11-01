"""Deferred U task score after BOTH R34/R35 full prediction/audit/closure seals."""
import sys,argparse,numpy as np
from u35_helpers import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',default=str(DEFAULT_LOCK));a=p.parse_args();check_u_lock(a.lock);verify(U_OUT);verify(U_CLOSURE);assert read(U_CLOSURE/'CLOSURE_AUDIT.json')['state']=='PASSED';audit=read(EXP/'U128_CPU_AUDIT.json');assert audit['state']=='PASSED' and audit['prediction_seal_sha256']==sha(U_OUT/'SEALED.json')
 r34=ROOT/'crossplatform/round34/scope_U128_attempt001';c34=ROOT/'crossplatform/round34/closure_U128_attempt001';verify(r34);verify(c34);assert read(c34/'CLOSURE_AUDIT.json')['state']=='PASSED';assert read(ROOT/'research/round34/experiment/CPU_AUDIT_U128.json')['state']=='PASSED'
 sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 records=[];neg=[];summary={};scenes=[];comparisons=[]
 for r in U_rows():
  z=U_feature(r);p35=load_npz(U_OUT/'predictions_PRIVATE'/(r['id']+'.npz'));p34=load_npz(r34/'predictions_PRIVATE'/(r['id']+'.npz'));q=dict(zip(ARMS,map(int,p35['queries'])));q.update({'R34_'+n:int(v) for n,v in zip(p34['arm_names'].tolist(),p34['queries']) if n not in ['source','R30_ungated']});assert int(p34['queries'][p34['arm_names'].tolist().index('source')])==q['source'];assert int(p34['queries'][p34['arm_names'].tolist().index('R30_ungated')])==q['R30_U_rows'];uq,inv=np.unique(list(q.values()),return_inverse=True);v=ious(r,z['boxes'][uq])[inv];rec={'alias':r['alias'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':q,'IoU':dict(zip(q,map(float,v)))};records.append(rec)
  for n in q:
   for ref in ['source','R30_U_rows','U_rows_matched_denominator','R34_h_samegate_QP','R34_noun_samegate_QP']:
    if n!=ref and rec['IoU'][n]<rec['IoU'][ref]-1e-10:neg.append({'alias':r['alias'],'platform':r['platform'],'configuration':n,'reference':ref,'delta_pp':100*(rec['IoU'][n]-rec['IoU'][ref])})
 for pl in ['drone','quad']:
  rr=[r for r in records if r['platform']==pl];summary[pl]={}
  for n in rr[0]['queries']:
   v=np.array([r['IoU'][n] for r in rr]);d=v-np.array([r['IoU']['source'] for r in rr]);summary[pl][n]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'delta_vs_source_pp':float(d.mean()*100),'improved':int((d>1e-10).sum()),'worsened':int((d< -1e-10).sum()),'query_changes':sum(r['queries'][n]!=r['queries']['source'] for r in rr),'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
  for j,s in enumerate(sorted({r['scene_PRIVATE'] for r in rr})):
   ss=[r for r in rr if r['scene_PRIVATE']==s];scenes.append({'platform':pl,'scene_alias':pl+'_scene'+str(j),'n':len(ss),'mIoU_percent':{n:float(np.mean([r['IoU'][n] for r in ss])*100) for n in rr[0]['queries']}})
 summary['balanced_mIoU_percent']={n:float(np.mean([summary[pl][n]['mIoU_percent'] for pl in ['drone','quad']])) for n in records[0]['queries']}
 for arm,ref in [('partial_output_keep','R30_U_rows'),('partial_output_keep','U_rows_matched_denominator'),('partial_output_keep','R34_h_samegate_QP'),('partial_output_keep','R34_noun_samegate_QP')]:
  by={pl:float(np.mean([r['IoU'][arm]-r['IoU'][ref] for r in records if r['platform']==pl])*100) for pl in ['drone','quad']};comparisons.append({'arm':arm,'reference':ref,'platform_delta_pp':by,'balanced_delta_pp':float(np.mean(list(by.values()))),'U_used_for_method_selection':False})
 dump(EXP/'U128_ALL_ROWS_PRIVATE.json',records);dump(EXP/'U128_SUMMARY.json',summary);dump(EXP/'U128_ALL_NEGATIVES.json',neg);dump(EXP/'U128_SCENE_ROWS.json',scenes);dump(EXP/'U128_PAIRED_COMPARISONS.json',comparisons);dump(EXP/'U128_SCORING_RECEIPT.json',{'state':'COMPLETED','n':128,'root_primary_lock_sha256':sha(a.lock),'R34_prediction_seal_sha256':sha(r34/'SEALED.json'),'R35_prediction_seal_sha256':sha(U_OUT/'SEALED.json'),'same_information_direct_QP_controls_included':True,'all_negative_cases_retained':True,'U_not_for_selection':True,'FINAL384_untouched':True});print('R35_U128_SUMMARY',summary)
if __name__=='__main__':main()
