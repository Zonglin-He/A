from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round74'
sys.path.insert(0,str(ROOT/'research/round72'));from common72 import read,dump,sha,load,seal,pin,perm
ARMS=['source','frozen_AB','R59','CLIP_reference','corr_RGB','corr_shuffled','corr_noRGB']
ADAPT=['corr_RGB','corr_shuffled','corr_noRGB']
def rows():return read(H/'input/MECHANISM64_PRIVATE.json')
def ab(r):
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:
  return {v:{k:z[v+'_'+k].copy() for k in ['h','query','tokens','token_mask','boxes','scores','query_anchor_xyz','point_indices']} for v in 'AB'}
def cview(r):return load(ROOT/'crossplatform/round69/C_random2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),['h','query','tokens','token_mask','boxes','query_anchor_xyz','point_indices'])
