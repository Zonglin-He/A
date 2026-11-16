"""Selected source-prepared heads are replayed on sourceVAL only, not target data."""
from core97 import *
def main():
 rows=[]
 for src in ['drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);out=B97/src;c=jread(out/'SOURCE_COMPLETE.json');zero=[];steps=jread(out/'SOURCE_STEPS.json');train=rows97(src,'TRAIN512')
  for x in steps:
   if x['stage']=='head' and x['gradient_norm']==0:
    ix=np.random.default_rng(91000+x['epoch']).permutation(512)[x['step']];r=train[int(ix)];rt=load(out/'samples_PRIVATE'/(r['id']+'_route.npz'));assert len(rt['token_roles'])==0,'zero-gradient optimizer equivalence requires no interface path';zero.append(x)
  # All positive-gradient source updates match the original R91 conditional optimizer step;
  # no-role rows have no differentiable interface path and no optimizer step in either program.
  vals=[]
  for r in rows97(src,'VAL128'):
   z0=load(out/'samples_PRIVATE'/(r['id']+'_prefix.npz'));g=load(out/'samples_PRIVATE'/(r['id']+'_geometry.npz'));rt=load(out/'samples_PRIVATE'/(r['id']+'_route.npz'));m.route=tensor_route89(rt);z,d=tensor_prefix(z0),tensor_data(g,r)
   with torch.no_grad():v=native_feature91(m,base,z,d);p,_,_=cached91(m,v,m.route,'full_disc')
   gt=transform_gt(jread(r['metapath'])['ground_info'][r['object_index']]['bbox_3d'],src,r['suffix']);boxes=z0['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);ious=candidate_ious(boxes,gt)[0];vals.append(float(ious[int(p.argmax())]))
  miou=float(np.mean(vals)*100);assert abs(miou-c['head_selected']['source_val_mIoU_percent'])<1e-9;assert fingerprint(base)==fp;rows.append({'source':src,'selected_sourceVAL_mIoU_percent':miou,'selected_epoch':c['head_selected']['epoch'],'sourceVAL_replays':len(vals),'source_preparation_attempted_updates':len(steps),'actual_optimizer_updates':sum(x['gradient_norm']>0 for x in steps),'no_interface_signal_rows':len(zero),'original_source100_unchanged':True});del m,base;torch.cuda.empty_cache()
 jdump(H97/'SOURCE_PREPARATION_AUDIT.json',{'state':'PASS_R97_SOURCE_ONLY_SELECTED_VAL_NATIVE_REPLAY_AND_EFFECTIVE_UPDATE_ACCOUNTING','rows':rows,'target_data_read':False,'FINAL_access':False});print(rows)
if __name__=='__main__':main()
