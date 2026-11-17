"""Posthoc component ablation on sealed representations, never a new method."""
from fit_transfer_diagnostic126 import *
def main():
 torch.set_num_threads(4);D=B/'posthoc_fit_transfer_attempt001';base=rd(H/'FIT_TRANSFER_ROWS_PRIVATE.json');(D/'BASE_SCALARS_BEFORE_RGB_SPLIT_PRIVATE.json').write_text(json.dumps(base,indent=2)+'\n');reg={(e['cohort'],e['source'],e['row']['id']):e for e in rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json')};col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];heads={};pred=[]
 for src in ['waymo','drone','quad']:
  s=np.load(R/col[src]['state_path']);heads[src]={k[6:]:s[k] for k in s.files if k.startswith('match.')}
 for r in base:
  co,src,tgt,sid=r['cohort'],r['source'],r['target'],r['id_PRIVATE'];q={}
  if not r['protected_alias_fallback']:
   f=np.load(R/'crossplatform/round125/formal'/co/(src+'_to_'+tgt)/'features_PRIVATE'/(sid+'.npz'));rA,_=Ares(reg[(co,src,sid)]);late=f['fixed_late'];rF=f['fixed_r'];mask=f['zero_F_mask']
   for v in ['F','C']:
    for st in ['zero','corr3']:
     logp=head64(f[st+'_'+v+'_h'],f[st+'_'+v+'_u'],mask,late,rF,heads[src]);q[v+'_lateOnly_'+st]=int((logp-rF).argmax());q[v+'_fullA_'+st]=int((logp+rA-rF).argmax())
  pred.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':sid,'queries_PRIVATE':q,'alias':r['protected_alias_fallback']})
 p=D/'RGB_ADDITIONAL_PREDICTIONS_PRIVATE.json';p.write_text(json.dumps(pred,indent=2)+'\n');(D/'RGB_ADDITIONAL_SEALED.json').write_text(json.dumps({'n':2304,'sha256':sha(p),'posthoc_after_primary_scoring':True,'used_for_method_selection':False,'new_parameter_or_model':False},indent=2)+'\n');gt={t:rd(R/f'research/round96/experiment/OFFICIAL_{t.upper()}_VAL_ORDER_PRIVATE.json') for t in ['waymo','drone','quad']};idx={(co,t):{r['id']:r for r in inputrows(co,t)} for co in ['DEV','EXP'] for t in gt}
 for r,pr in zip(base,pred):
  co,src,tgt,sid=r['cohort'],r['source'],r['target'],r['id_PRIVATE']
  if r['protected_alias_fallback']:
   for v in ['F','C']:
    for mod in ['lateOnly','fullA']:
     for st in ['zero','corr3']:r['values'][v+'_'+mod+'_'+st]=r['values'][v+'_full_'+st]
  else:
   a=np.load(R/'crossplatform/round125/formal'/co/(src+'_to_'+tgt)/'samples_PRIVATE'/(sid+'.npz'));bb=np.c_[a['source_boxes'],np.zeros(256)].astype(float);bb[:,3:6]=np.maximum(bb[:,3:6],1e-6);ys=many_iou(bb,np.array(gt[tgt][idx[(co,tgt)][sid]['anno_index']]['source_GT7_float32']));r['values'].update({k:float(ys[q]) for k,q in pr['queries_PRIVATE'].items()})
 out=[]
 for co in ['DEV','EXP']:
  for scope in ['all','same_AB_active_C_available']:
   rr=[r for r in base if r['cohort']==co and (scope=='all' or r['B_active'] and r['C_available'])]
   for v in ['F','C']:
    for mod in ['native','lateOnly','fullA','full']:
     parts=[]
     for src,tgt in sorted({(r['source'],r['target']) for r in rr}):
      ss=[r for r in rr if r['source']==src and r['target']==tgt];parts.append(sum(r['values'][v+'_'+mod+'_corr3']-r['values'][v+'_'+mod+'_zero'] for r in ss)/len(ss)*100)
     out.append({'cohort':co,'scope':scope,'view':v,'readout':mod,'n':len(rr),'gain_pp_all_six_equal':sum(parts)/6})
 (H/'FIT_TRANSFER_ROWS_PRIVATE.json').write_text(json.dumps(base,indent=2)+'\n');(H/'RGB_READOUT_SPLIT_DIAGNOSTIC.json').write_text(json.dumps({'state':'POSTHOC_FIXED_REPRESENTATION_COMPONENT_DIAGNOSIS_NOT_METHOD_OR_SELECTION','n':2304,'readouts':'native:noRGB;lateOnly:add frozenlate insidequerynormalization;fullA:late+rA;full:late+rF. Onlythelast2contextsprovide different frozencompetitionvectors.','groups':out,'GT_used_only_after_new_diagnostic_seal':True,'method_or_primary_changed':False,'FINAL_access':False},indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
