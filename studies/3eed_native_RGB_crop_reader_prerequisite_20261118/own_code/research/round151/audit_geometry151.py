"""Independent camera/ROI/sparse-graph reconstruction, no inference or GT."""
from common151 import *
import numpy as np,torch
from PIL import Image
argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round141'));from audit141 import cam,tx,raw_to_model_transform,project_model_points,sparse_graph,merged,cnn,read135,tensors;sys.argv=argv
from common151 import R,H,B,rd,sha,dump,check,utc

def independently_build(e,z,ev,recipes,decode=True):
 src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;bb=np.load(R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz'));sensors=[e['A_sensor'],e['pair_PRIVATE']['B']];images=[];wh=[]
 for ob in sensors:
  assert ob['eligible_for_extra_media'] and not ob['protected_image_content_alias'];assert sha(ob['imagepath'])==ob['image_sha256'];im=Image.open(ob['imagepath']);wh.append(im.size);images.append(im.convert('RGB') if decode else im)
 ca,cb=[cam(x) for x in sensors];Ma,Mb=[raw_to_model_transform(x) for x in [ca,cb]];rr=rd(R/e['registration_path']);rawB=tx(tx(z['xyz'][0],np.linalg.inv(Ma)),np.linalg.inv(np.asarray(rr['forward']['T_raw_B_to_A'])));points=[project_model_points(z['xyz'][0],ca,wh[0]),project_model_points(tx(rawB,Mb),cb,wh[1])];graphs=[]
 for v,tag in enumerate(['A','B']):
  visibility=bb['norm_'+tag+'image_visible'];g=sparse_graph(points[v],points[v],wh[v],visibility_override=visibility)
  for k,a in g.items():
   old=bb['norm_'+tag+'image_'+k]
   if k.endswith('weight'):assert abs(a-old).max(initial=0)<1e-5
   else:assert np.array_equal(a,old)
  if decode:g['image']=np.asarray(images[v].resize((384,256),Image.Resampling.BILINEAR))
  graphs.append(g)
 _,inv=np.unique(z['xyz'][0],axis=0,return_inverse=True);xyz=np.unique(z['xyz'][0],axis=0);ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];support=[];nA=[];nB=[]
 for group in groups:
  ins=np.logical_or.reduce([(abs(xyz-z['boxes'][q,:3])<=np.maximum(z['boxes'][q,3:6],1e-6)/2).all(-1) for q in group]);support.append(ins);raw=ins[inv];nA.append(len(set(inv[raw&graphs[0]['visible']].tolist())));nB.append(len(set(inv[raw&graphs[1]['visible']].tolist())))
 score=np.array([(ev['pA'][g]+ev['pB'][g]).max()/2 for g in groups]);order=np.lexsort((np.arange(len(groups)),-score));chosen=[];expected=[]
 for group in order:
  if not ev['S'][group] or any(np.count_nonzero(support[group]&support[h])/max(np.count_nonzero(support[group]|support[h]),1)>.5 for h in chosen):continue
  c=[nA[group],nB[group]]
  if max(c)<4:continue
  v=int(c[1]>c[0]);ii=np.flatnonzero(support[group][inv]&graphs[v]['visible']);uv=points[v]['uv'][ii];center=(uv.min(0)+uv.max(0))/2;span=np.maximum(np.ptp(uv,axis=0),1)*1.5;W,HH=wh[v];w0=max(float(span[0]),float(span[1])*1.5,48);w=min(int(np.ceil(w0)),W);h=min(int(np.ceil(w0/1.5)),HH);x=int(np.clip(np.floor(center[0]-w/2),0,W-w));y=int(np.clip(np.floor(center[1]-h/2),0,HH-h));expected.append(dict(group_PRIVATE=int(group),view=v,ROI_PRIVATE=[x,y,x+w,y+h],visible_unique_support=c[v],original_wh_PRIVATE=[W,HH]));chosen.append(group)
  if len(chosen)==3:break
 assert recipes==expected,('recipe_mismatch',src,tgt)
 if not decode:return None
 hi=[];lo=[];bad=[];dup=[]
 for j,rec in enumerate(recipes):
  v=rec['view'];x,y,r,b=rec['ROI_PRIVATE'];WW,HH=r-x,b-y;uv=points[v]['uv']-[x,y];inside=(uv[:,0]>=0)&(uv[:,0]<WW)&(uv[:,1]>=0)&(uv[:,1]<HH)&(points[v]['depth']>0);p=dict(uv=uv,depth=points[v]['depth'],in_image=inside);visible=graphs[v]['visible']&inside;g=sparse_graph(p,p,(WW,HH),visibility_override=visible);g['image']=np.asarray(images[v].resize((384,256),Image.Resampling.BILINEAR,box=(x,y,r,b)));l={k:a.copy() for k,a in g.items()};W,H0=wh[v];l['image']=np.asarray(Image.fromarray(graphs[v]['image']).resize((384,256),Image.Resampling.BILINEAR,box=(x*384/W,y*256/H0,r*384/W,b*256/H0)));d={k:a.copy() for k,a in g.items()};ids=np.flatnonzero(visible);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R151crop|crop'+str(j)).encode()).hexdigest()[:16],16));order=rng.permutation(ids);perm=np.arange(1024)
  if len(order)>1:perm[order]=np.roll(order,1)
  inverse=np.argsort(perm);d['p2i']=g['p2i'][perm];d['p2i_weight']=g['p2i_weight'][perm];d['i2p']=np.where(g['i2p']>=0,inverse[g['i2p'].clip(0)],-1);hi.append(g);lo.append(l);bad.append(d);dup.append(graphs[v])
 swap=[{k:a.copy() for k,a in g.items()} for g in hi]
 for j,g in enumerate(swap):g['image']=hi[(j+1)%len(hi)]['image'].copy()
 return dict(high_local=graphs+hi,low_local=graphs+lo,wrong_local=graphs+bad,duplicate_local=graphs+dup,region_swap_local=graphs+swap)
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');seal=rd(B/'RECIPES_SEALED.json');n=0;nb=0
 for f,h in seal['files'].items():assert sha(B/f)==h
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];recipes=rd(B/'recipes_PRIVATE'/name/(sid+'.json'))['recipes']
  if e['has_B']:
   z=dict(np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz')));ev=dict(np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz')));independently_build(e,z,ev,recipes,False);nb+=1
  else:assert recipes==[]
  n+=1
  if n%128==0:print('CROP151_AUDIT',n,flush=True)
 dump(H/'PREPARATION_AUDIT.json',dict(status='PASS',n=n,B=nb,independent_ROI_selection_and_camera_projection=True,GT_read=False));print('CROP151_AUDIT_COMPLETE',n,nb,flush=True)
if __name__=='__main__':main()
