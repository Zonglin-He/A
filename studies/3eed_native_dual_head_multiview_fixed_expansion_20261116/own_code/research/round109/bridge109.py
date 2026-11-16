"""New-input source/cache/A256 bridge only, no repeated old fitting tests."""
from core109 import *
sys.path.insert(0,str(R109/'research/round108'));from replay108 import ce_cpu108
def main():
 checks=[];anchors={a['A']['metapath']:a for a in jread(H109/'ANCHORS_PRIVATE.json')}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=base.prediction_heads[-1].sem_cls_scores_head
  for tgt in ['waymo','drone','quad']:
   if tgt==src:continue
   rows=jread(H109/f'input/TARGET_{tgt.upper()}256_PRIVATE.json');r=next(x for x in rows if anchors[str(Path(x['metapath']).resolve())]['A']['eligible_for_extra_media']);old=load(R109/'crossplatform/round100'/(src+'_to_'+tgt)/'samples_PRIVATE'/(r['id']+'.npz'));pold=dict(zip(old['arm_names'],old['scores']))
   with torch.no_grad():z0,g,_=capture98(base,r,src);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=feature106(m,base,z,d,route,S,[]);p,logit=ce108(head,A['h'],A['mask'])
   e=float(abs(A['p'].cpu().numpy()-pold['prepared_frozen']).max());assert e<1e-7 and np.array_equal(z0['boxes'],old['source_boxes']);pc,lc=ce_cpu108(A['h'].cpu().numpy(),A['mask'].cpu().numpy(),statece108(head));ceerr=float(abs(pc-p.cpu().numpy()).max());assert ceerr<3e-6 and pc.argmax()==int(p.argmax());checks.append({'source':src,'target':tgt,'new_anchor_source_prepared_probability_error':e,'source_boxes_bitexact':True,'independent_CE_probability_error':ceerr,'source_checkpoint_sha256':source97(src)['checkpoint_sha256'],'FINAL_access':False});restore(m,S);print('BRIDGE109',src,tgt,len(checks),flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 jdump(H109/'BRIDGE.json',{'state':'PASS_NEW_EXPANSION_ANCHOR_SOURCE_NATIVE_AND_A256_CACHE_BRIDGE','n':len(checks),'checks':checks,'no_GT':True,'no_FINAL':True})
if __name__=='__main__':main()
