"""No task labels. Unique eligible mask assignment; uncertainty not merged."""
import sys,time,numpy as np
from common29 import *
from observations27 import load
sys.path.insert(0,str(ROOT/'crossplatform/round17'))
from pair_geometry import prepare

def main():
 out=BASE/'groups208_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();n=0;skip=0;start=time.perf_counter()
 for i,r in enumerate(rows()[:208]):
  pairs=old(r,'pairs')['pairs'];p={'alias':r['alias'],'pairs':[],'groups_PRIVATE':{},'ambiguous_queries_PRIVATE':[]}
  if pairs:
   frame,scene,bind=load(r);z=feature(r);raw=prepare(scene,z['boxes'],r['id']);support=np.asarray(raw['point_support_count']);coverage=np.asarray(raw['point_support_fraction']);pi=np.asarray(raw['projection_iou']);valid=(support>=3)&(coverage>=.15)&(pi>=.1)&(z['boxes'][:,3:]>0).all(1)[:,None];unique=valid.sum(1)==1;p['ambiguous_queries_PRIVATE']=np.flatnonzero(valid.sum(1)>1).tolist()
   for pair in pairs:
    a,b=pair['mask_indices_PRIVATE'];ga=np.flatnonzero(valid[:,a]&unique&(support[:,b]<=.2*np.maximum(support[:,a],1)));gb=np.flatnonzero(valid[:,b]&unique&(support[:,a]<=.2*np.maximum(support[:,b],1)));assert pair['qA_PRIVATE'] in ga
    eligible=bool(len(ga) and len(gb) and pair['qB_PRIVATE'] in gb)
    if eligible:
     # A group cannot be redefined differently per competitor; canonical unique assignment.
     for m in [a,b]:p['groups_PRIVATE'][str(m)]=np.flatnonzero(valid[:,m]&unique).tolist()
    p['pairs'].append({'pair_alias':pair['pair_alias'],'masks_PRIVATE':[a,b],'eligible':eligible,'reason':'unique_mask_groups' if eligible else 'B_representative_ambiguous_or_empty','group_A_n':len(ga),'group_B_n':len(gb)});n+=eligible;skip+=not eligible
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),p)
  if (i+1)%32==0:print('GROUPS29',i+1,flush=True)
 seal(out,'R29_unique_mapping_instance_group_hypotheses_no_GT',n=208,eligible_pairs=n,ambiguous_pairs_retained_skipped=skip,seconds=time.perf_counter()-start);print('GROUPS29_SEALED',n,skip)
if __name__=='__main__':main()
