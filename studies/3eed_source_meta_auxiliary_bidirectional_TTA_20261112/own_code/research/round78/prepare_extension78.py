"""Locked larger cohorts: reuse R69 source prefixes, measure new RGB geometry only."""
from core78 import *
from prepare76 import geometry
import time
p=read(H78/'experiment/EXTENSION_LOCK.json');pin(p);rr=read(ROOT/p['manifest']);out=B78/'extension_geometry2304_attempt001';assert not out.exists();out.mkdir();(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();start=time.perf_counter();cost=[];required=['vis0','pos0','lang0','textpad','xyz','query_points_sample_inds','query_anchor_xyz','point_indices','token_mask','boxes','scores']+['position'+str(j) for j in range(6)]
for i,r in enumerate(rr):
 f=ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz')
 with np.load(f,allow_pickle=False) as zz:z={k:zz['A_'+k].copy() for k in required}
 data,proof=geometry(r,z);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**data);dump(out/'cases_PRIVATE'/(r['id']+'.json'),proof);cost.append({'platform':r['platform'],'seed_raw_max_error':proof['seed_raw_max_error'],'visible_seeds':proof['visible_seeds'],'RGB_cells_with_depth':proof['RGB_cells_with_depth']})
 if(i+1)%128==0:print('GEOMETRY_EXT78',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
dump(out/'COUNTS.json',{'n':2304,'seconds':time.perf_counter()-start,'new_full_model_forwards':0,'new_target_GT_or_FINAL_read':False,'cohort_coverage':cost});seal(out,'R78_FIXED_EXTENSION_NEW_RAW_RGB_GEOMETRY_CACHED_SOURCE_PREFIXES_NO_GT',n=2304)
