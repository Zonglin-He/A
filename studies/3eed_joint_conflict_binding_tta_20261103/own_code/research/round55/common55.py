"""R55 fixed observation contradictions; no task labels or instruction execution."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
COLORS=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black']
VARIANTS=('real','RGBshuffle','endpointshuffle')
ARMS=['source','zero_step']+[a for v in VARIANTS for a in [v+'_direct',v+'_tta1_raw',v+'_tta1_joint',v+'_tta3_raw',v+'_tta3_joint']]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def seal(out,state,**extra):
 dump(out/'SEALED.json',{'state':state,**extra,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
def pin(plan):
 for group in ('programs_sha256','inputs_sha256'):
  for p,h in plan.get(group,{}).items():assert sha(ROOT/p)==h,('pin',p)
def rng(s):return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little'))
def colors_conflict(means,counts,cover,requests):
 """Only stable high alternative support, never mere absence of requested color."""
 n=means.shape[1];result=np.zeros((len(requests),n),float)
 # Broad hues far apart; adjacent hues and white/gray are deliberately not opposites.
 hue={'red':0.,'orange':.08,'yellow':.16,'green':.33,'cyan':.5,'blue':.66,'purple':.78,'pink':.91}
 for e,req in enumerate(requests):
  if len(req)!=1 or req[0] not in COLORS:continue
  color=req[0];k=COLORS.index(color);alts=[]
  for j,c in enumerate(COLORS):
   if color in hue and c in hue and min(abs(hue[c]-hue[color]),1-abs(hue[c]-hue[color]))>=.24:alts.append(j)
   if (color,c) in [('white','black'),('black','white')]:alts.append(j)
  if not alts:continue
  alt=means[:,:,alts].max(-1);wanted=means[:,:,k]
  known=(counts>=16).all(0)&(cover>=.35).all(0)
  stable=(alt>=.45).all(0)&(wanted<=.20).all(0)&((alt-wanted)>=.30).all(0)
  # Fixed evidence confidence; missing support is neutral. No learned UNKNOWN mass.
  strength=np.minimum(.95,counts.min(0)/(counts.min(0)+16.))*np.clip((alt-wanted).min(0)/.5,0,1)
  result[e,known&stable]=strength[known&stable]
 return result

def relation_conflict(intervals,valid,overlap,edges,entity_names):
 n=len(intervals);out=np.zeros((len(entity_names)-1,n,n),float)
 lo,hi=intervals[:,0],intervals[:,1]
 separate=(overlap<=.1)&valid[:,None]&valid[None,:]
 # Matrices always index [target i, reference j]. Graph's actual directed endpoints are respected.
 for edge in edges:
  if edge.get('frame')!='camera_image' or edge.get('predicate') not in ('LEFT_OF','RIGHT_OF'):continue
  a,b=edge.get('subject'),edge.get('reference')
  if a=='T' and b in entity_names[1:]:ref=b;left_is_target=True
  elif b=='T' and a in entity_names[1:]:ref=a;left_is_target=False
  else:continue
  if edge['predicate']=='RIGHT_OF':left_is_target=not left_is_target
  contradiction=(lo[:,None]>hi[None,:]+.02) if left_is_target else (lo[None,:]>hi[:,None]+.02)
  out[entity_names.index(ref)-1]=np.maximum(out[entity_names.index(ref)-1],np.where(separate&contradiction,.9,0.))
 return out

def joint_conflicts(color,geo):
 if len(color)<2:return np.empty((0,256,256),np.float32)
 return (1-(1-geo)*(1-color[0][None,:,None])*(1-color[1:,None,:])).clip(0,.95).astype(np.float32)
