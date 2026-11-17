from core128 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');p.add_argument('--limit-per-direction',type=int,default=128);a=p.parse_args()
def main():
 reg=[e for e in jread(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];counts={};allseals={}
 for src in ['waymo','drone','quad']:
  base=base97(src);S=snapshot(base);fish,recs=fisher(base,src,S);fd=B/'fisher_PRIVATE';fd.mkdir(exist_ok=True);torch.save({k:v.cpu() for k,v in fish.items()},fd/(src+'.pt'));jdump(H/(src+'_FISHER.json'),{'source_only_unlabeled_inputs':len(recs),'target_GT_used':False,'source_checkpoint_sha256':source97(src)['checkpoint_sha256'],'records':recs})
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rr=[e for e in reg if e['source']==src and e['target']==tgt][:a.limit_per_direction]
   if a.engineering:rr=[next(e for e in rr if not e['protected_alias_fallback'])]
   name=src+'_to_'+tgt;D=B/('engineering_fast' if a.engineering else 'formal_fast')/name
   for sub in ['samples_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for i,e in enumerate(rr):
    sid=e['row']['id'];fp=D/'samples_PRIVATE'/(sid+'.npz');rp=D/'records_PRIVATE'/(sid+'.json')
    if fp.exists() and rp.exists():continue
    if e['protected_alias_fallback']:
     old=np.load(R/f'crossplatform/round127/formal/DEV/{name}/samples_PRIVATE/{sid}.npz');p0=old['scores'][list(old['arm_names']).index('original_source')];keys=['source','frozen_three_augment_mean','BN_statistics_only','eata_zero','memo_BNprior_zero','memo_BNprior_three_mean']+[m+s for m in ['tent','eata','memo'] for s in ['_post','_post_three_mean']];out={k:p0.copy() for k in keys};meta={'protected_alias_fallback':True,'no_new_protected_feature_or_media_read':True};boxes=old['source_boxes']
    else:out,meta,boxes=process(base,S,e,fish);meta['protected_alias_fallback']=False
    keys=sorted(out);arr=np.stack([out[k] for k in keys]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-8);np.savez_compressed(fp,arm_names=np.array(keys),scores=arr,source_boxes=boxes,queries=arr.argmax(1));jdump(rp,meta)
    print('R128_FAST',a.engineering,name,i+1,'steps', {m:sum(x['accepted'] for x in meta.get(m,{}).get('steps',[])) for m in ['tent','eata','memo']},flush=True)
   seal={'n':len(rr),'GT_used':False,'FINAL_access':False,'files':{str(f.relative_to(D)):sha(f) for f in sorted(D.rglob('*')) if f.is_file() and f.name!='SEALED.json'}};jdump(D/'SEALED.json',seal);allseals[name]=sha(D/'SEALED.json');counts[name]=len(rr)
  reset(base,S);del base,S,fish;torch.cuda.empty_cache()
 jdump(H/('ENGINEERING_FAST.json' if a.engineering else 'FAST_MATRIX_SEALED.json'),{'counts':counts,'seals':allseals,'all_negatives_retained':True,'FINAL_access':False})
if __name__=='__main__':B.mkdir(exist_ok=True);main()
