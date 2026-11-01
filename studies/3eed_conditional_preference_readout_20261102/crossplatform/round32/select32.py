"""DEV-only finite support-family selection; R/S/T labels never inspected."""
import sys,numpy as np,collections,json,time
from common32 import *
import scope32 as row_module
import endpoint_scope32 as end_module
ARMS=['scope_TTA','scope_memory','global_mean_TTA','global_norm_TTA','ungated_memory']

def feasibility():
 import torch
 from transformers import RobertaTokenizerFast
 out=BASE/'feasibilityA_attempt001';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;tokenizer=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True);fb=read(ROOT/'crossplatform/round29/direct208_attempt001/FEEDBACK_PRIVATE.json');records=[]
 for kind,mod in [('row',row_module),('endpoint',end_module)]:
  for pl in ['drone','quad']:
   A=[r for r in rows() if r['platform']==pl];ff=[f for f in fb if f['platform']==pl]
   for key in mod.KEYS:
    lib=mod.build_library(A,key,tokenizer,fb)
    for r in [r for r in A if r['id'] in {f['id_PRIVATE'] for f in ff}]:
     z=feature(r);knn=mod.neighbors(mod.make_key(z,r,key,tokenizer),lib);vv,_=mod.score_row(r,z['scores'],mod.parent_scores(r),lib,knn)
     for k in mod.KS:
      for arm,s in vv[k].items():
       q=mod.group_readout(r,s)
       for f in [f for f in ff if f['id_PRIVATE']==r['id']]:records.append({'support':kind,'platform':pl,'key':key,'k':k,'arm':arm,'alias':r['alias'],'pair_alias':f['pair_alias'],'feedback':f['label'],'pair_satisfied':bool(s[f['positive_group_PRIVATE']].max()>s[f['negative_group_PRIVATE']].max()),'final_positive':q in f['positive_group_PRIVATE']})
 dump(out/'PAIR_EXECUTION.json',records);seal(out,'R32_feedback_feasibility_before_DEV_labels',pairs=22,families=8,task_labels_used=False);print('FEASIBILITY32_SEALED')

def select():
 sys.path.insert(0,str(ROOT/'research/round27'));from evaluate27 import ious
 verify(BASE/'scope400_attempt001');verify(BASE/'scope400_endpoint_attempt001');verify(BASE/'feasibilityA_attempt001');feas=read(BASE/'feasibilityA_attempt001/PAIR_EXECUTION.json');records=[];table=[]
 # All dev candidate choices scored after full family prediction seals.
 for support,folder in [('row','scope400_attempt001'),('endpoint','scope400_endpoint_attempt001')]:
  for key in row_module.KEYS:
   for k in row_module.KS:
    vals={arm:[] for arm in ARMS}
    for r in rows('DEV64'):
     p=load_npz(BASE/folder/'predictions_PRIVATE'/(r['id']+'_'+key+'.npz'));names=p['arm_names'].tolist();ix=[names.index('k'+str(k)+'_'+arm) for arm in ARMS];v=ious(r,p['boxes'][ix])
     for arm,j,val in zip(ARMS,ix,v):records.append({'support':support,'key':key,'k':k,'arm':arm,'alias':r['alias'],'platform':r['platform'],'iou':float(val),'q_PRIVATE':int(p['queries'][j])})
    for arm in ARMS:
     rr=[x for x in records if x['support']==support and x['key']==key and x['k']==k and x['arm']==arm];pm={pl:float(np.mean([x['iou'] for x in rr if x['platform']==pl])*100) for pl in ['drone','quad']};fr=[x for x in feas if x['support']==support and x['key']==key and x['k']==k and x['arm']==arm];B=[x for x in fr if x['feedback']=='B'];table.append({'support':support,'key':key,'k':k,'arm':arm,'platform_mIoU_percent':pm,'balanced_mIoU_percent':float(np.mean(list(pm.values()))),'pairs_satisfied':sum(x['pair_satisfied'] for x in fr),'B_pairs_satisfied':sum(x['pair_satisfied'] for x in B),'B_final_positive':sum(x['final_positive'] for x in B),'feasible':sum(x['pair_satisfied'] for x in B)>=5 and sum(x['final_positive'] for x in B)>=4})
 def best(rr):return sorted(rr,key=lambda x:(-x['balanced_mIoU_percent'],row_module.KEYS.index(x['key']),x['k'],ARMS.index(x['arm'])))[0]
 primary=best([x for x in table if x['support']=='endpoint' and x['arm']=='scope_TTA' and x['feasible']]);memory=best([x for x in table if x['support']=='endpoint' and x['arm']=='scope_memory']);globalcontrol=best([x for x in table if x['support']=='endpoint' and x['arm'] in ['global_mean_TTA','global_norm_TTA']]);rowbest=best([x for x in table if x['support']=='row' and x['arm']=='scope_TTA' and x['feasible']]);ungatedmemory=best([x for x in table if x['support']=='endpoint' and x['arm']=='ungated_memory']);paired=next(x for x in table if x['support']=='endpoint' and x['key']==primary['key'] and x['k']==primary['k'] and x['arm']=='scope_memory');matchedrow=next(x for x in table if x['support']=='row' and x['key']==primary['key'] and x['k']==primary['k'] and x['arm']=='scope_TTA')
 dump(EXP/'DEV_FAMILY_TABLE.json',table);dump(EXP/'DEV_FAMILY_ROWS_PRIVATE.json',records);dump(EXP/'SELECTED_SCOPE_LOCK.json',{'primary':primary,'same_gate_memory':paired,'best_DEV_memory_same_budget':memory,'best_DEV_global_amount_control':globalcontrol,'best_DEV_row_support_ablation':rowbest,'same_family_row_support_ablation':matchedrow,'best_DEV_ungated_memory':ungatedmemory,'selection_uses_exposed_DEV64_labels':True,'R64_S64_T128_used_to_select':False,'T128_labels_not_read':True,'source_R30_and_prediction_seals':[sha(BASE/f/'SEALED.json') for f in ['scope400_attempt001','scope400_endpoint_attempt001','feasibilityA_attempt001']],'selector_sha256':sha(__file__)});print(json.dumps(read(EXP/'SELECTED_SCOPE_LOCK.json'),indent=2))
if __name__=='__main__':{'feasibility':feasibility,'select':select}[sys.argv[1]]()
