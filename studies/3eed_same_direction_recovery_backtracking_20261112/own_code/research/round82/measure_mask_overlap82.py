"""CPU-only diagnostic of third-mask target overlap; no network or task labels."""
from core82 import *
torch.set_num_threads(4)
rows=[];coh=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json')
for i,r in enumerate(allrows80()):
 f=casepath(r) if r['id'] in DEV_IDS else ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(r['id']+'.npz')
 with np.load(f,allow_pickle=False) as z:d={k:torch.as_tensor(z[k].copy()) for k in ['depth_valid','cell','visible','query_anchor_indices']}
 d['image']=torch.empty(0);pm=[];im=[]
 for repeat in [76001,76002,76003]:p,m=masks(r,d,repeat);pm.append(p);im.append(m)
 rows.append({'row':'X'+str(i+1).zfill(5),'cohort':'DEV336' if r['id'] in DEV_IDS else coh[r['id']],'platform':r['platform'],'third_point_targets':int(pm[2].sum()),'third_point_targets_also_in_fit':int((pm[2]&(pm[0]|pm[1])).sum()),'third_RGB_targets':int(im[2].sum()),'third_RGB_targets_also_in_fit':int((im[2]&(im[0]|im[1])).sum()),'third_depth_targets':int(d['depth_valid'].sum()),'third_depth_targets_also_in_fit':int(d['depth_valid'].sum())})
summary=[]
for part in ['DEV336','VAL1280','TRAIN1024']:
 rr=[r for r in rows if r['cohort']==part]
 for kind in ['point','RGB','depth']:
  total=sum(r['third_'+kind+'_targets'] for r in rr);seen=sum(r['third_'+kind+'_targets_also_in_fit'] for r in rr);summary.append({'cohort':part,'kind':kind,'n':len(rr),'third_targets':total,'third_targets_also_in_fit':seen,'overlap_fraction':seen/max(total,1)})
dump(H82/'MASK_TARGET_OVERLAP_ROWS.json',rows);dump(H82/'MASK_TARGET_OVERLAP_SUMMARY.json',summary);print(summary)
