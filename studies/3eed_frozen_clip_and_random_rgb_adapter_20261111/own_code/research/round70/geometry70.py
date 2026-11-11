"""Only source geometry, calibrated RGB and deterministic literal text parsing."""
import sys,re
from common70 import *
from PIL import Image
sys.path.insert(0,str(ROOT/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points
sys.path.insert(0,str(ROOT/'research/round66'))
from groups66 import support,overlap,LEXICON
from scipy.sparse import csr_matrix

def appearance(text):
 # No VLM/LLM parse cache. Preserve literal attributes until a relation cue.
 s=re.sub(r'^\s*(?:in (?:the|this) (?:image|picture|scene)[, ]*|there (?:is|are)\s+|(?:find|locate|identify)\s+)','',text,flags=re.I).strip()
 m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',s,re.I)
 if m is None:return text,'UNKNOWN_NO_LITERAL_ENTITY_FULL_TEXT'
 rel=re.search(r'\b(?:next to|beside|behind|in front of|near|closest to|nearest to|(?:to (?:the )?)?(?:left|right) of|on (?:the )?(?:left|right)|between)\b',s[m.end():],re.I)
 end=m.end()+rel.start() if rel else len(s);phrase=s[:end].strip(' ,.;')
 if len(phrase)<3:return text,'UNKNOWN_EMPTY_FULL_TEXT'
 return phrase,'LITERAL_PREFIX_BEFORE_RELATION_NOT_ROLE_TRUTH'

def groups(boxes,x):
 a=support(x,boxes);counts=a.sum(1);inter=(csr_matrix(a.astype(np.float32))@csr_matrix(a.astype(np.float32)).T).toarray();u=counts[:,None]+counts[None]-inter;j=np.divide(inter,u,out=np.zeros_like(inter),where=u>0);ok=(overlap(boxes,boxes)>=.7)&(j>=.7)&(inter>=4);gs=[];unused=set(range(256))
 for q in range(256):
  if q not in unused:continue
  g=[q];unused.remove(q)
  for k in sorted(unused):
   if ok[k,g].all():g.append(k)
  for k in g[1:]:unused.remove(k)
  gs.append(np.array(g,np.int64))
 return gs,a

def rects(boxes,spec,wh):
 signs=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]])
 xyz=(boxes[:,None,:3]+signs[None]*boxes[:,None,3:]/2).reshape(-1,3);p=project_model_points(xyz,spec,wh);uv=p['uv'].reshape(-1,8,2);d=p['depth'].reshape(-1,8);out=np.zeros((len(boxes),4),float);fraction=np.zeros(len(boxes));valid=np.zeros(len(boxes),bool);W,HH=wh
 for i in range(len(boxes)):
  if not (boxes[i,3:]>0).all() or not (d[i]>.1).all() or not np.isfinite(uv[i]).all():continue
  b=np.r_[uv[i].min(0),uv[i].max(0)];c=np.clip(b,[0,0,0,0],[W,HH,W,HH]);size=c[2:]-c[:2]
  if (size<2).any():continue
  out[i]=c;fraction[i]=np.prod(size)/max(np.prod(b[2:]-b[:2]),1e-12);valid[i]=True
 return out,fraction,valid

def prepare(r,z):
 with Image.open(r['imagepath']) as im:image=im.convert('RGB');rgb=np.array(image);wh=image.size
 raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy();x=raw.copy()
 if r['platform']=='drone':x[:,2]+=np.float32(1.8)
 spec=CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),missing_distortion_policy='assume_zero_declared')
 p=project_model_points(x.astype(float),spec,wh);ids=np.flatnonzero(p['in_image']);W,HH=wh;xy=np.floor(p['uv'][ids]).astype(int);cell=(xy[:,1]//4)*((W+3)//4)+xy[:,0]//4;depth=np.full(((HH+3)//4)*((W+3)//4),np.inf);np.minimum.at(depth,cell,p['depth'][ids]);visible=ids[p['depth'][ids]<=depth[cell]+.3+.01*p['depth'][ids]]
 gs,bags=groups(z['boxes'].astype(float),x.astype(float));rep=np.array([g[np.argmax(z['scores'][g])] for g in gs]);mapping=np.empty(256,np.int64)
 for k,g in enumerate(gs):mapping[g]=k
 groupbags=np.array([bags[g].any(0) for g in gs]);owners=groupbags.sum(0);rr,frac,ok=rects(z['boxes'][rep].astype(float),spec,wh);reliability=np.zeros(len(gs));projectrel=np.zeros(len(gs));viscounts=[]
 for k,b in enumerate(groupbags):
  vv=visible[b[visible]];n=len(np.unique(np.floor(p['uv'][vv]).astype(int),axis=0));total=int(b[ids].sum());viscounts.append(n)
  if not ok[k]:continue
  projectrel[k]=frac[k]
  if n<4:continue
  purity=float(np.mean(1/owners[vv])) if len(vv) else 0.;reliability[k]=projectrel[k]*n/(n+8)*len(vv)/max(total,1)*purity
 seed=z['xyz'][0];sp=project_model_points(seed.astype(float),spec,wh);valid=sp['in_image'].copy();sx=np.floor(np.nan_to_num(sp['uv'],nan=-100)).astype(int);valid&=(sx[:,0]>=8)&(sx[:,0]<W-8)&(sx[:,1]>=8)&(sx[:,1]<HH-8);ii=np.flatnonzero(valid);cc=(sx[ii,1]//4)*((W+3)//4)+sx[ii,0]//4;valid[ii]&=sp['depth'][ii]<=depth[cc]+.3+.01*sp['depth'][ii];ii=np.flatnonzero(valid)
 patches=np.array([rgb[y-8:y+8,x0-8:x0+8].transpose(2,0,1) for x0,y in sx[ii]],np.uint8) if len(ii) else np.empty((0,3,16,16),np.uint8);order=perm(len(ii),'R70seedpartition|'+r['id']);fit=order[:min(256,len(ii)//2)];held=order[min(len(ii)//2,len(ii)):];validB=len(fit)>=16 and len(held)>=16
 subject,status=appearance(r['caption']);payload={'boxes':z['boxes'],'scores':z['scores'],'group_representatives':rep,'query_group':mapping,'rects':rr.astype(np.float32),'rect_valid':ok,'reliability':reliability,'project_reliability':projectrel,'visible_support_counts':np.array(viscounts),'seed_indices':ii,'seed_pixels':sx[ii],'patches':patches,'fit_patch_indices':fit,'held_patch_indices':held,'B_eligible':np.array(validB)}
 return payload,{'subject_PRIVATE':subject,'subject_parse_status':status,'group_count':len(gs),'supported_visual_groups':int((reliability>0).sum()),'visible_seeds':len(ii),'fit_seeds':len(fit),'held_seeds':len(held),'B_eligible':bool(validB),'missing_distortion_assumed_zero':spec.distortion is None,'sparse_LiDAR_visibility_not_true_RGB_mask':True}

def crops(image,b,scale=1.):
 W,HH=image.size;c=(b[:2]+b[2:])/2;s=(b[2:]-b[:2])*scale/2;v=np.r_[c-s,c+s];lo=np.floor(v[:2]).astype(int);hi=np.ceil(v[2:]).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,[W,HH]);im=image.crop((*lo,*hi));side=max(im.size);out=Image.new('RGB',(side,side),(127,127,127));out.paste(im,((side-im.width)//2,(side-im.height)//2));return out
