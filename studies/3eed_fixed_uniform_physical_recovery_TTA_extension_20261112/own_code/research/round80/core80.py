"""Fixed R76 e6 + R79 uniform physical-recovery SGD, no source training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/round79'))
from core79 import *
H80=ROOT/'research/round80';B80=ROOT/'crossplatform/round80'
ARMS80=['source','frozen_bi','frozen_direct3','uniform_sgd1','uniform_wrong_sgd1','frozen_ABC','R59']
EXT_KEYS=['vis0','pos0','lang0','textpad','xyz','query_points_sample_inds','query_anchor_xyz','point_indices','token_mask','boxes','scores']+['position'+str(j) for j in range(6)]
def allrows80():return target_rows()+read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')
def item80(r):
 if r['id'] in DEV_IDS:
  z0=prefix(r);data=load(casepath(r))
 else:
  f=ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz');old=load(f);z0={k:old['A_'+k] for k in EXT_KEYS};data=load(ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
 return z0,tensor_prefix(z0),tensor_data(data,r)
DEV_IDS={r['id'] for r in target_rows()}
def uniform_weights80(d,pm,im):
 # Exactly the R79 uniform local aggregation; never compute query saliency.
 return weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform']
def old80(r):
 p=(ROOT/'crossplatform/round79/prediction464_attempt001' if r['id'] in DEV_IDS else ROOT/'crossplatform/round69/predicted2304_attempt001')
 z=load(p/'samples_PRIVATE'/(r['id']+'.npz'));return z,dict(zip(z['arm_names'],z['scores']))
