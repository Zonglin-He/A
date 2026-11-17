"""Posthoc diagnosis: fitted versus unused observations, never method selection.
Uses only sealed R125 h/U and the independently audited R126 float64 reader.
"""
from core126 import *
import sys
sys.path.insert(0,str(R/'research/round118'));from core118 import rows as inputrows
sys.path.insert(0,str(R/'research/round6/source_validation_scoring'));from batch_yaw_iou import many_iou
def main():
 torch.set_num_threads(4);assert rd(H/'SCALAR_AUDIT.json')['n']==2304;D=B/'posthoc_fit_transfer_attempt001';D.mkdir(exist_ok=False);reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];heads={};pred=[];precision_changes=0
 for src in ['waymo','drone','quad']:
  st=np.load(R/col[src]['state_path']);heads[src]={k[6:]:st[k] for k in st.files if k.startswith('match.')}
 for e in reg:
  co,src,tgt,sid=e['cohort'],e['source'],e['target'],e['row']['id'];P=R/'crossplatform/round125/formal'/co/(src+'_to_'+tgt);orig=np.load(P/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(orig['arm_names'],orig['scores']));rec=rd(P/'records_PRIVATE'/(sid+'.json'));queries={}
  if e['protected_alias_fallback']:queries={v+'_'+mod+'_'+step:int(op['prepared_A'].argmax()) for v in ['A','B','F','C'] for mod in ['native','full'] for step in ['zero','corr3']}
  else:
   f=np.load(P/'features_PRIVATE'/(sid+'.npz'));rA,_=Ares(e);late=f['fixed_late'];rF=f['fixed_r'];mask=f['zero_F_mask']
   for v in ['A','B','F','C']:
    for step in ['zero','corr3']:
     pre=step+'_fit_'+v if v in ['A','B'] else step+'_'+v;hp,up=f[pre+'_h'],f[pre+'_u']
     for mod in ['native','full']:
      q=int(head64(hp,up,mask,late if mod=='full' else late*0,(rA if v in ['A','B'] else rF) if mod=='full' else rF*0,heads[src]).argmax());queries[v+'_'+mod+'_'+step]=q
      if mod=='full' and v in ['F','C']:
       want=op[('mean_context' if v=='F' else 'zero_C') if step=='zero' else step+'_'+v].argmax();precision_changes+=q!=want
  pred.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'protected_alias_fallback':e['protected_alias_fallback'],'B_active':rec.get('fit',{}).get('active',False),'C_available':rec.get('C_available',False),'queries_PRIVATE':queries})
 p=D/'PREDICTIONS_PRIVATE.json';p.write_text(json.dumps(pred,indent=2)+'\n');(D/'PREDICTIONS_SEALED.json').write_text(json.dumps({'n':len(pred),'sha256':sha(p),'posthoc_after_primary_scoring':True,'source_states_unchanged':True,'used_for_any_method_choice':False,'sealed_before_this_diagnostic_GT_read':True},indent=2)+'\n')
 gt={t:rd(R/f'research/round96/experiment/OFFICIAL_{t.upper()}_VAL_ORDER_PRIVATE.json') for t in ['waymo','drone','quad']};index={(co,t):{r['id']:r for r in inputrows(co,t)} for co in ['DEV','EXP'] for t in gt};scalars=[]
 for r in pred:
  co,src,tgt,sid=r['cohort'],r['source'],r['target'],r['id_PRIVATE'];a=np.load(R/'crossplatform/round125/formal'/co/(src+'_to_'+tgt)/'samples_PRIVATE'/(sid+'.npz'));boxes=np.c_[a['source_boxes'],np.zeros(256)].astype(float);boxes[:,3:6]=np.maximum(boxes[:,3:6],1e-6);y=many_iou(boxes,np.array(gt[tgt][index[(co,tgt)][sid]['anno_index']]['source_GT7_float32']));scalars.append({k:v for k,v in r.items() if k!='queries_PRIVATE'}|{'values':{k:float(y[q]) for k,q in r['queries_PRIVATE'].items()}})
 summary=[]
 for co in ['DEV','EXP']:
  for scope in ['all','same_AB_active_C_available']:
   rr=[r for r in scalars if r['cohort']==co and (scope=='all' or r['B_active'] and r['C_available'])]
   for v in ['A','B','F','C']:
    for mod in ['native','full']:
     direction=[]
     for src,tgt in sorted({(r['source'],r['target']) for r in rr}):
      pp=[r for r in rr if r['source']==src and r['target']==tgt];zero=[r['values'][v+'_'+mod+'_zero'] for r in pp];after=[r['values'][v+'_'+mod+'_corr3'] for r in pp];direction.append({'cohort':co,'scope':scope,'source':src,'target':tgt,'view':v,'readout':mod,'n':len(pp),'zero_mIoU_percent':sum(zero)/len(zero)*100,'updated_mIoU_percent':sum(after)/len(after)*100,'gain_pp':sum(a-b for a,b in zip(after,zero))/len(zero)*100})
     assert len(direction)==6;summary.append({'cohort':co,'scope':scope,'group':'all_six_equal','view':v,'readout':mod,'n':len(rr),'zero_mIoU_percent':sum(x['zero_mIoU_percent'] for x in direction)/6,'updated_mIoU_percent':sum(x['updated_mIoU_percent'] for x in direction)/6,'gain_pp':sum(x['gain_pp'] for x in direction)/6})
 (H/'FIT_TRANSFER_ROWS_PRIVATE.json').write_text(json.dumps(scalars,indent=2)+'\n');(H/'FIT_TRANSFER_DIAGNOSTIC.json').write_text(json.dumps({'state':'POSTHOC_DESCRIPTIVE_SEALED_STATES_NO_METHOD_SELECTION','n':2304,'precision_vs_primary_FC_query_changes':int(precision_changes),'C_is_A_fixed_hypotheses_and_GT':True,'new_model_or_parameter_update':False,'used_for_any_gate_or_hyperparameter':False,'groups':summary},indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
