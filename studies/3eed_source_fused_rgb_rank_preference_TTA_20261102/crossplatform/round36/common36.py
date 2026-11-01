"""R36 fixed 2x2, A/DEV only. No U or FINAL reader exists here."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'crossplatform/round25')]
import common32 as current
from common32 import read,dump,sha,seal,verify,load_npz
BASE=ROOT/'crossplatform/round36';EXP=ROOT/'research/round36/experiment';MODES=['rgb','no_rgb']
def rows(split='A208'):
 assert split in ['A208','DEV64'];return current.rows(split)
def feature(r):return current.feature(r)
def rgb_feature(r):
 p=ROOT/'crossplatform/round25/rgb_features_attempt002/samples_PRIVATE'/(r['id']+'.npz') if r['stage']!='Q' else BASE/'rgbDEV64_attempt001/samples_PRIVATE'/(r['id']+'.npz')
 return load_npz(p)
def source_config(mode):
 return next(c for c in read(ROOT/'research/round25/experiment/FINAL_SOURCE_SELECTED_LOCK.json')['selected'] if c['seed']==0 and c['variant']==mode)
def source_path(mode):
 c=source_config(mode);p=ROOT/'crossplatform/round25'/c['training_folder']/'states_PRIVATE'/c['state_name'];assert sha(p)==c['state_sha256'];return p
def check_plan():
 p=read(EXP/'PLAN_LOCK.json')
 for k,v in p['program_sha256'].items():assert sha(ROOT/k)==v,k
 for k,v in p['input_sha256'].items():assert sha(ROOT/k)==v,k
 return p
def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def captured(r,mode):return load_npz(BASE/'capture272_attempt001/samples_PRIVATE'/(r['id']+'_'+mode+'.npz'))
def groups(r):
 if r['stage']=='Q':return []
 return list(read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'].values())
def grouped_readout(r,s,s0):
 used=set();options=[]
 for ix in groups(r):
  used.update(ix);rep=max(ix,key=lambda i:(float(s0[i]),-i));options.append((float(np.max(s[ix])),-rep,rep))
 for i in range(256):
  if i not in used:options.append((float(s[i]),-i,i))
 return int(max(options)[2])
def tensorbank(rr):
 import torch
 zz=[feature(r) for r in rr];im=[rgb_feature(r) for r in rr];T=max(len(z['tokens']) for z in zz);N=len(rr);tokens=np.zeros((N,T,64),np.float32);pad=np.ones((N,T),bool);mask=np.zeros((N,T),bool)
 for i,z in enumerate(zz):L=len(z['tokens']);tokens[i,:L]=z['tokens'];pad[i,:L]=False;mask[i,:L]=z['token_mask']
 d={'h':np.stack([z['h'] for z in zz]),'tokens':tokens,'padding':pad,'mask':mask,'text_mean':np.stack([z['text_h'][1:-1].mean(0) for z in zz]),'scores':np.stack([z['scores'] for z in zz]),'rgb':np.stack([z['rgb'] for z in im]),'global':np.stack([z['global_token'] for z in im]),'geom':np.stack([z['geom'] for z in im]),'valid':np.stack([z['valid'] for z in im])}
 return {k:torch.tensor(v,device='cuda') for k,v in d.items()}
def fp_nonrank(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():
  if not k.startswith('rank.'):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
