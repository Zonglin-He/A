from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round75'
sys.path.insert(0,str(ROOT/'research/round66'));from common66 import read,dump,load,sha,seal,pin
ADAPT=['dense','output','shuffled']
ARMS=['source','frozen_mask_probability','frozen_mask_token','frozen_mask_representation','frozen_ABC','R59','dense_full','output_full','shuffled_full','dense_ensemble','output_ensemble','shuffled_ensemble','held_mask_source','held_mask_dense','held_mask_output','held_mask_shuffled']
def rows():return read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
def original(r):
 with np.load(ROOT/'crossplatform/round66/prefusion336_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:return {k[2:]:z[k].copy() for k in z.files if k.startswith('A_') and not k.startswith('A_class_')}
def mask_for(r,z,view):
 ix=np.asarray(z['query_points_sample_inds'],int).reshape(-1);assert len(np.unique(ix))==256;n=z['xyz'].shape[1];available=np.setdiff1d(np.arange(n),ix);seed=int(hashlib.sha256(('R75_anchor_preserving_20percent_nonanchor|'+r['id']+'|'+view).encode()).hexdigest()[:16],16);m=np.zeros(n,bool);m[np.random.default_rng(seed).choice(available,int(len(available)*.2),replace=False)]=True;assert not m[ix].any();return m
