from pathlib import Path
import sys,hashlib,re
import numpy as np
import torch
R=Path(__file__).resolve().parent;ROUND=R.parent;P=ROUND.parent;U=P/'upstream';sys.path[:0]=[str(P),str(U)]
from src.joint_det_dataset import WAYMO_SYNONYMS,M3ED_SYNONYMS,get_positive_map
from tta import LEXICON
from utils.transform_waymo import transform_to_front_view
CLASS_NAMES=['car','pedestrian','bus','othervehicle','truck','cyclist']
HEAD_NAMES=['proposal_','0head_','1head_','2head_','3head_','4head_','last_']
SENTINEL=' . not mentioned'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_caption(c):return ' '.join(c.replace(',',' ,').split())
def noun_caption(c):
 m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',c,re.I);return (m.group() if m else c),m is None

def class_masks(tokenizer,caption,platform):
 # Exact released-loader leading/trailing spaces and synonym-priority order.
 formatted=' '+caption+' ';tok=tokenizer(formatted,return_tensors='pt');dictionary=WAYMO_SYNONYMS if platform=='waymo' else M3ED_SYNONYMS
 masks=[];found=[];spans=[]
 for cls in CLASS_NAMES:
  positions=[]
  for word in dictionary.get(cls,[cls]):
   positions.extend((m.start(),m.end(),word) for m in re.finditer(r'\b'+re.escape(word)+r'\b',formatted,re.I))
  found.append(bool(positions))
  if positions:
   beg,end,word=positions[0];mask=get_positive_map(tok,[torch.tensor([beg,end])])[0]>0;spans.append([beg,end])
  else:mask=np.zeros(256,bool);spans.append([-1,-1])
  masks.append(mask)
 masks=np.asarray(masks,bool);return masks,np.asarray(found,bool),np.any(masks,axis=1),np.asarray(spans,np.int64),tok['input_ids'][0].numpy()

def input_masks(tokenizer,text):
 tok=tokenizer(text,return_offsets_mapping=True,return_tensors='pt');off=tok['offset_mapping'][0].numpy();m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',text,re.I)
 mask=(off[:,1]>m.start())&(off[:,0]<m.end()) if m else off[:,1]>off[:,0]
 out=np.zeros(256,bool);assert len(mask)<=256,'Actual token length exceeds official256score capacity';out[:len(mask)]=mask
 return out,tok['input_ids'][0].numpy(),tok['attention_mask'][0].numpy(),off

def sample_points(row):
 path=Path(row['lidar_path']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4)
 assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
 # Deterministic per-expression Generator, explicit departure from original worker/global RNG order.
 ix=np.random.default_rng(row['sampling_seed']).choice(len(raw),16384,replace=len(raw)<16384);xyz=raw[ix,:3].copy()
 if row['platform']=='drone':xyz[:,2]+=1.8
 if row['platform']=='waymo':xyz,_=transform_to_front_view(xyz,np.zeros((1,7)),['F','FL','FR','SL','SR'][row['waymo_view_suffix']])
 return xyz.astype(np.float32),ix.astype(np.int64),len(raw)
