"""Native calibrated image crops; no detector, semantic expert or GT labels."""
from common151 import *
import numpy as np
from PIL import Image
sys.path.insert(0,str(R/'research/round105'));import geometry105 as G
from common151 import R,H,B,rd,sha,dump,check,utc

def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def groups(ass):return [np.flatnonzero(ass==i) for i in range(int(ass.max()+1))]
def target_geometry(e,z):
 sensor=e['A_sensor'];other=e['pair_PRIVATE']['B'];imgs=[]
 for r in [sensor,other]:
  assert r['eligible_for_extra_media'] and not r['protected_image_content_alias'];assert sha(r['imagepath'])==r['image_sha256'];imgs.append(Image.open(r['imagepath']).convert('RGB'))
 whA,whB=[im.size for im in imgs];ca,cb=G.camera(sensor),G.camera(other);pa=G.project_model_points(z['xyz'][0].astype(float),ca,whA);rr=rd(R/e['registration_path']);assert sha(R/e['registration_path'])==e['registration_sha256'];rawA=G.transform_points(z['xyz'][0].astype(float),G.inverse_se3(G.raw_to_model_transform(ca)));rawB=G.transform_points(rawA,G.inverse_se3(np.asarray(rr['forward']['T_raw_B_to_A'])));modelB=G.raw_to_model_points(rawB,other['platform'],int(other.get('suffix') or 0));pb=G.project_model_points(modelB,cb,whB);src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;bb=np.load(R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz'));gg=[]
 for tag,pp,wh,im in [('A',pa,whA,imgs[0]),('B',pb,whB,imgs[1])]:
  vis=bb['norm_'+tag+'image_visible'];g=G.graph_kernel(pp,wh,vis)
  for k in ['p2i','i2p','visible']:assert np.array_equal(g[k],bb['norm_'+tag+'image_'+k])
  for k in ['p2i_weight','i2p_weight']:assert abs(g[k]-bb['norm_'+tag+'image_'+k]).max(initial=0)<1e-5
  g['image']=np.asarray(im.resize((384,256),Image.Resampling.BILINEAR));g['query_anchor_indices']=z['query_points_sample_inds'];gg.append(g)
 return gg,[pa,pb],imgs

def select_recipes(z,ev,graphs,projections,images):
 ass=ev['assignment'];gs=groups(ass);uniq,inv=np.unique(z['xyz'][0],axis=0,return_inverse=True);inside=(abs(uniq[None]-z['boxes'][:,None,:3])<=np.maximum(z['boxes'][:,None,3:6],1e-6)/2).all(-1);sup=np.array([inside[g].any(0) for g in gs]);vis=[]
 for g in graphs:
  v=np.zeros(len(uniq),bool);np.logical_or.at(v,inv,g['visible']);vis.append(v)
 anchor=(ev['pA']+ev['pB'])/2;order=sorted(np.flatnonzero(ev['S']),key=lambda g:(-float(anchor[gs[g]].max()),int(g)));chosen=[];recipes=[]
 for group in order:
  if any((sup[group]&sup[x]).sum()/max((sup[group]|sup[x]).sum(),1)>.5 for x in chosen):continue
  counts=[int((sup[group]&v).sum()) for v in vis]
  if max(counts)<4:continue
  view=0 if counts[0]>=counts[1] else 1;ids=np.flatnonzero(sup[group][inv]&graphs[view]['visible']);uv=projections[view]['uv'][ids];W,HH=images[view].size;lo=uv.min(0);hi=uv.max(0);center=(lo+hi)/2;span=np.maximum(hi-lo,1)*1.5;w=max(span[0],span[1]*1.5,48.);h=w/1.5;w=min(int(np.ceil(w)),W);h=min(int(np.ceil(h)),HH);x=int(np.clip(np.floor(center[0]-w/2),0,W-w));y=int(np.clip(np.floor(center[1]-h/2),0,HH-h));box=[x,y,x+w,y+h];assert w>=1 and h>=1 and (uv[:,0]>=x-1e-8).all() and (uv[:,0]<x+w+1e-8).all() and (uv[:,1]>=y-1e-8).all() and (uv[:,1]<y+h+1e-8).all();recipes.append(dict(group_PRIVATE=int(group),view=int(view),ROI_PRIVATE=box,visible_unique_support=counts[view],original_wh_PRIVATE=[W,HH]));chosen.append(group)
  if len(chosen)==3:break
 return recipes

def wrong_crop(g,salt):
 bad={k:v.copy() for k,v in g.items()};ids=np.flatnonzero(g['visible']);rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16));order=rng.permutation(ids);perm=np.arange(len(g['visible']))
 if len(order)>1:perm[order]=np.roll(order,1)
 inverse=np.argsort(perm);bad['p2i']=g['p2i'][perm];bad['p2i_weight']=g['p2i_weight'][perm];bad['i2p']=np.where(g['i2p']>=0,inverse[g['i2p'].clip(0)],-1);assert np.array_equal((bad['p2i']>=0).any(1),(g['p2i']>=0).any(1));return bad

def crop_banks(graphs,projections,images,recipes,salt):
 high=[];low=[];wrong=[];dups=[]
 for j,rec in enumerate(recipes):
  v=rec['view'];x,y,r,b=rec['ROI_PRIVATE'];w,h=r-x,b-y;pp=projections[v];uv=pp['uv']-np.array([x,y]);visible=graphs[v]['visible']&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);g=G.graph_kernel(dict(uv=uv,depth=pp['depth']),(w,h),visible);g['query_anchor_indices']=graphs[v]['query_anchor_indices'];g['image']=np.asarray(images[v].resize((384,256),Image.Resampling.BILINEAR,box=(x,y,r,b)));lo={k:a.copy() for k,a in g.items()};W,HH=images[v].size;box=(x*384/W,y*256/HH,r*384/W,b*256/HH);lo['image']=np.asarray(Image.fromarray(graphs[v]['image']).resize((384,256),Image.Resampling.BILINEAR,box=box));high.append(g);low.append(lo);wrong.append(wrong_crop(g,salt+'|crop'+str(j)));dups.append({k:a.copy() for k,a in graphs[v].items()})
 return dict(high_local=graphs+high,low_local=graphs+low,wrong_local=graphs+wrong,duplicate_local=graphs+dups)
