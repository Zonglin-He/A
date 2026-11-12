"""Same locked method and physically indexed R69 A-prefixes on larger cohorts."""
from core78 import *
KEYS_EXT=['vis0','pos0','lang0','textpad','xyz','query_points_sample_inds','query_anchor_xyz','point_indices','token_mask','boxes','scores']+['position'+str(j) for j in range(6)]
def ext_rows():return read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')
def ext_prefix(r):
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:return {k:z['A_'+k].copy() for k in KEYS_EXT}
def ext_item(r):
 z0=ext_prefix(r);return z0,tensor_prefix(z0),tensor_data(load(B78/'extension_geometry2304_attempt001/samples_PRIVATE'/(r['id']+'.npz')),r)
