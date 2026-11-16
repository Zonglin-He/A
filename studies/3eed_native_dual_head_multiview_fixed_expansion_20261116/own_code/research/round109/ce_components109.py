"""Postseal R108 attribution only; no new readout, pseudo-label or selector."""
from core109 import *
def main():
 strict97();maxerr=0.
 assert jread(R109/'research/round108/SCALAR_AUDIT.json')['n']==768;rowsout=[]
 for src in ['waymo','drone','quad']:
  st=load(R109/'crossplatform/round108'/(src+'_CE_WEIGHTS_PRIVATE.npz'));head=ThreeLayerMLP(288,256).cuda().eval().requires_grad_(False);head.load_state_dict({k:torch.as_tensor(v,device='cuda') for k,v in st.items()},strict=True)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;oldrows=jread(R109/'research/round108/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));data={r['id']:r for r in jread(R109/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')}
   for r in oldrows:
    sid=r['id_PRIVATE'];z=load(R109/'crossplatform/round108'/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(z['arm_names'],z['scores']));rec={'source':src,'target':tgt,'row_PRIVATE':sid,'protected_alias_fallback':r['status']['protected_alias_fallback'],'new_method_or_output_selected':False}
    if rec['protected_alias_fallback']:rowsout.append(rec);continue
    f=load(B106/name/'features_PRIVATE'/(sid+'.npz'));h=torch.as_tensor(f['real_fusedABC_h'],device='cuda');mask=torch.as_tensor(f['real_fusedABC_mask'],device='cuda');pm=torch.zeros(256,device='cuda',dtype=torch.bool);pm[:len(mask)]=mask
    with torch.no_grad():
     logits=head(h.T[None])[0].T.double();allmass=logits.logsumexp(-1);nonnull=logits[:,:255].logsumexp(-1);subject=logits.masked_fill(~pm[None],-torch.inf).logsumexp(-1);lnon=nonnull-allmass;lcond=subject-nonnull;ltotal=subject-allmass;qq=ltotal.softmax(0);err=float(np.max(abs(qq.cpu().numpy()-p['softtoken_fusedABC'])));maxerr=max(maxerr,err);assert err<5e-6 and int(qq.argmax())==int(p['softtoken_fusedABC'].argmax());assert torch.max(abs(lnon+lcond-ltotal))<1e-12
    base='real_frozen_fusedABC';i=int(p[base].argmax());pairs={}
    for arm in ['softtoken_fusedABC','softtoken_geometric_fusedABC','softtoken_arithmetic_fusedABC']:
     j=int(p[arm].argmax());delta=float(r['arms'][arm]['IoU']-r['arms'][base]['IoU']);cm=float(ltotal[j]-ltotal[i]);nm=float(lnon[j]-lnon[i]);tm=float(lcond[j]-lcond[i]);assert abs(cm-nm-tm)<1e-12;pairs[arm]={'query_changed':i!=j,'delta_IoU':delta,'CE_margin_new_over_BBF':cm,'nonnull_margin':nm,'conditional_subject_margin':tm,'BBF_log_margin_new_over_BBF':float(np.log(p[base][j])-np.log(p[base][i])),'nonnull_dominates_absolute_component':abs(nm)>abs(tm),'nonnull_supports_new':nm>0,'conditional_supports_new':tm>0}
    rec['pairs']=pairs;rowsout.append(rec)
   print('CE_COMPONENT109',name,flush=True)
 summary=[]
 for arm in ['softtoken_fusedABC','softtoken_geometric_fusedABC','softtoken_arithmetic_fusedABC']:
  rr=[r['pairs'][arm] for r in rowsout if 'pairs' in r]
  for group in ['all','changed','improved','harmed','same_IoU_changed_query']:
   xs=[x for x in rr if group=='all' or (group=='changed' and x['query_changed']) or (group=='improved' and x['delta_IoU']>1e-12) or (group=='harmed' and x['delta_IoU']< -1e-12) or (group=='same_IoU_changed_query' and x['query_changed'] and abs(x['delta_IoU'])<=1e-12)]
   if xs:summary.append({'arm':arm,'group':group,'n':len(xs),'nonnull_dominates_n':sum(x['nonnull_dominates_absolute_component'] for x in xs),'nonnull_supports_new_n':sum(x['nonnull_supports_new'] for x in xs),'conditional_supports_new_n':sum(x['conditional_supports_new'] for x in xs),'mean_abs_nonnull_margin':float(np.mean([abs(x['nonnull_margin']) for x in xs])),'mean_abs_conditional_subject_margin':float(np.mean([abs(x['conditional_subject_margin']) for x in xs]))})
 jdump(H109/'CE_COMPONENT_ROWS_PRIVATE.json',rowsout);jdump(H109/'CE_COMPONENT_DIAGNOSTIC.json',{'source_study':'R108 sealed predictions and already exposed postseal IoUs','n':len(rowsout),'protected_alias_fallback':sum(r['protected_alias_fallback'] for r in rowsout),'max_replay_probability_error':maxerr,'source_strict_runtime_restored':True,'failed_attempt001_retained':True,'new_predictions_or_labels_created':False,'postseal_descriptive_not_gate_or_selector':True,'identity_or_background_truth_inferred':False,'nonnull_meaning':'one minus sourceCE no-object255; unmatched-source-query mass, NOT verified foreground/object identity','groups':summary});print('CE_COMPONENT109_COMPLETE',len(rowsout))
if __name__=='__main__':main()
