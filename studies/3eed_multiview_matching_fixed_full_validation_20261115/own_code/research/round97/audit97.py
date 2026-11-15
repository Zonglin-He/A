"""Independent native replays and complete sealed-byte/source-provenance audit."""
from core97 import *
from run97 import case97,ARMS97
def main():
 summary=[];checks=[];nrows=0;allseals={};maxerr=0.
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src)
  if src!='waymo':
   root=B97/src;c=jread(root/'SOURCE_COMPLETE.json');assert c['source_train_n']==512 and c['source_val_n']==128 and c['updates']==6144 and c['original_source100_unchanged'] and not c['target_data_read'];assert sha(root/'SELECTED_SOURCE_ONLY.npz')==c['selected_sha256'];fusion=jread(root/'FUSION_HISTORY.json');head=jread(root/'HEAD_HISTORY.json');assert len(fusion)==8 and len(head)==4;assert c['fusion_selected']==max(fusion,key=lambda x:(x['source_val_mIoU_percent'],-x['epoch']));assert c['head_selected']==max(head,key=lambda x:(x['source_val_mIoU_percent'],-x['epoch']));steps=jread(root/'SOURCE_STEPS.json');assert len(steps)==6144 and all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in steps);summary.append({'source':src,'source_only_updates':len(steps),'source_selection_correct':True,'selected_sha256':c['selected_sha256']})
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   out=B97/(src+'_to_'+tgt);seal=jread(out/'SEALED.json');rows=rows97(tgt,'target');assert seal['n']==len(rows) and seal['arms']==ARMS97 and not seal['target_task_GT_read'];assert seal['source100_checkpoint_sha256']==source97(src)['checkpoint_sha256'];assert len(seal['files'])==len(rows)*2
   for f,s in seal['files'].items():assert sha(out/f)==s,(src,tgt,f)
   allseals[src+'_to_'+tgt]=sha(out/'SEALED.json');nrows+=len(rows)
   for i in [0,len(rows)//2,len(rows)-1]:
    r=rows[i];saved=load(out/'samples_PRIVATE'/(r['id']+'.npz'));restore(m,S)
    with torch.no_grad():z0,g,rec,proof=sample97(base,r)
    assert np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['point_indices'],z0['point_indices']);rt,info=role97(base,m,r,z0,g,tensor_prefix(z0),tensor_data(g,r));scores,records,cost=case97(base,m,S,r,z0,g,rt);err=float(abs(scores-saved['scores']).max());maxerr=max(maxerr,err);assert err<1e-8 and np.array_equal(scores.argmax(1),saved['queries']);old=jread(out/'records_PRIVATE'/(r['id']+'.json'))
    for arm in ['physical_raw','physical_bt','wrong_physical_raw','wrong_physical_bt']:
     for name in ['alpha','actual_parameter_L2','gradient_norm','objective_before']:assert abs(records[arm][name]-old['updates'][arm][name])<1e-8
    checks.append({'source':src,'target':tgt,'row':'ROW'+str(i+1).zfill(6),'probability_error':err,'Top1_mismatches':0})
  assert fingerprint(base)==fp;del m,base;torch.cuda.empty_cache()
 assert nrows==18620 and len(checks)==18;allseal={'state':'R97_ALL_SIX_PREDICTIONS_SEALED_AND_NATIVE_AUDITED_BEFORE_SCORING','n_directions':6,'n':nrows,'seals_sha256':allseals,'target_task_GT_read':False,'FINAL_access':False};jdump(B97/'MATRIX_SEALED.json',allseal);jdump(H97/'NATIVE_AUDIT.json',{'state':'PASS_R97_SIX_DIRECTIONS_BYTES_NATIVE_REPLAYS_AND_SOURCE_PROVENANCE','n_directions':6,'n':nrows,'new_source_preparations':summary,'native_checks':checks,'max_probability_error':maxerr,'new_task_scores':0,'FINAL_access':False});print({'state':'PASS_R97_NATIVE','n':nrows,'native_checks':len(checks),'max_error':maxerr})
if __name__=='__main__':main()
