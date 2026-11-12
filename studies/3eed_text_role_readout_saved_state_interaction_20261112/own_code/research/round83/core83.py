"""Two saved parameter states, fixed query-token readouts, no optimization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round82'))
from core82 import *
from roles83 import roles,spanmask
H83=ROOT/'research/round83';B83=ROOT/'crossplatform/round83'
ARMS83=['source','frozen_native','BT_native','frozen_background','BT_background','frozen_role','BT_role','frozen_mismatch','BT_mismatch','frozen_direct3','frozen_ABC','R59']
def prefix83(r):
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as f:z0={k:f['A_'+k].copy() for k in EXT_KEYS}
 return z0,tensor_prefix(z0),tensor_data(load(ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(r['id']+'.npz')),r)
def lse83(x):
 a=x.astype(np.float64);hi=a.max(-1);return hi+np.log(np.exp(a-hi[:,None]).sum(-1))
def probability83(logit):
 v=np.exp(logit-logit.max());return v/v.sum()
def readouts83(S,role,native_probability):
 name=np.array(role['name'],bool);bg=np.array(role['background'],bool);attr=np.array(role['attribute'],bool);sh=np.array(role['mismatch'],bool);out={'native':native_probability}
 if not role['usable']:return {k:native_probability for k in ['native','background','role','mismatch']}
 a=lse83(S[:,name])-np.log(name.sum());b=lse83(S[:,bg])-np.log(bg.sum());out['background']=probability83(a-b)
 for key,mask in [('role',attr),('mismatch',sh)]:out[key]=probability83(.5*a+.5*(lse83(S[:,mask])-np.log(mask.sum()))-b) if mask.any() else out['background']
 return out
