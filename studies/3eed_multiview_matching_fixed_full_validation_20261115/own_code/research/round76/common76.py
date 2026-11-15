from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round76';BASE=ROOT/'crossplatform/round76'
sys.path.insert(0,str(ROOT/'research/round66'));from common66 import read,dump,sha,load,seal,pin
MODES=['independent','rgb_to_point','bidirectional']
ARMS=['source','independent_frozen','rgb_to_point_frozen','bidirectional_frozen','bidirectional_reverse_off','bidirectional_wrong_RGB','bidirectional_frozen_budget3','bidirectional_TTA1','bidirectional_TTA3','bidirectional_TTA3_wrong_correspondence','frozen_ABC','R59']
def source_rows(split):return read(H/'input'/('SOURCE_'+split+'_PRIVATE.json'))
def target_rows():return read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
def casepath(r):return BASE/'prepared976_attempt001/samples_PRIVATE'/(r['id']+'.npz')
def prefixpath(r):return (BASE/'source_prefix640_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if r['platform']=='waymo' else ROOT/'crossplatform/round66/prefusion336_attempt001/samples_PRIVATE'/(r['id']+'.npz')
def prefix(r):
 with np.load(prefixpath(r),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files} if r['platform']=='waymo' else {k[2:]:z[k].copy() for k in z.files if k.startswith('A_') and not k.startswith('A_class_')}
