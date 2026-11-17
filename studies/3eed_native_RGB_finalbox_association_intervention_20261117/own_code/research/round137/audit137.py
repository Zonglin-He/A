"""Independent CPU native RGB/fusion, pooling and complete-readout audit."""
from common137 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();saved=sys.argv;sys.argv=[saved[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=saved
def linear(x,W,n):return F.linear(x,W[n+'.weight'],W[n+'.bias'])
def adapted(x,W,n):return x+linear(F.gelu(linear(x,W,n+'.down')),W,n+'.up')
def source_rgb(P,d,W):
 x=torch.from_numpy(d['image'].copy()).permute(2,0,1)[None].float()/255;x=(x-.5)*2
 for j,ng in [(0,4),(3,4),(6,8)]:x=F.gelu(F.group_norm(F.conv2d(x,W[f'rgb.{j}.weight'],W[f'rgb.{j}.bias'],stride=2,padding=1),ng,W[f'rgb.{j+1}.weight'],W[f'rgb.{j+1}.bias']))
 I=adapted(x.flatten(2).transpose(1,2)[0],W,'image_adapter');P=adapted(P,W,'point_adapter');idx=torch.from_numpy(d['i2p'].copy()).long();ww=torch.from_numpy(d['i2p_weight'].copy());valid=(idx>=0)&(ww>0);Q=linear(I,W,'point_to_image.q');K=linear(P,W,'point_to_image.k')[idx.clamp_min(0)];V=linear(P,W,'point_to_image.v')[idx.clamp_min(0)];logits=(Q[:,None]*K).sum(-1)/8+ww.clamp_min(1e-12).log();a=logits.masked_fill(~valid,-1e4).softmax(-1)*valid;a=a/a.sum(-1,keepdim=True).clamp_min(1e-12);II=I+W['gate_I'].tanh()*linear((a[:,:,None]*V).sum(1),W,'point_to_image.o')*valid.any(1)[:,None]
 idx=torch.from_numpy(d['p2i'].copy()).long();ww=torch.from_numpy(d['p2i_weight'].copy())*(idx>=0);ww=ww/ww.sum(-1,keepdim=True).clamp_min(1e-12);return (II[idx.clamp_min(0)]*ww[:,:,None]).sum(1)
def main():
 check();torch.set_num_threads(4);root=B/('engineering' if args.engineering else 'direct');seal=rd(root/'SEALED.json');n=0;noB=0;editrows=0;editqueries=0;mxrgb=0.;mxlate=0.;mxprob=0.
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W=tensors(dict(np.load(R/'crossplatform/round135'/(src+'_SOURCE_MODEL_PRIVATE.npz'))))
  for D in sorted(root.glob(src+'_to_*')):
   for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
    sid=p.stem;scores=np.load(p);ps=dict(zip(scores['arm_names'],scores['scores']));ff=np.load(D/'features_PRIVATE'/p.name);rec=rd(D/'records_PRIVATE'/(sid+'.json'));z=np.load(R/'crossplatform/round131/formal'/D.name/'prefix_PRIVATE'/p.name);old=np.load(R/'crossplatform/round135/formal'/D.name/'features_PRIVATE'/p.name);oo=np.load(R/'crossplatform/round135/formal'/D.name/'samples_PRIVATE'/p.name);op=dict(zip(oo['arm_names'],oo['scores']));rt=tensors(dict(np.load(R/'crossplatform/round135/formal'/D.name/'route_PRIVATE'/p.name)));assert np.array_equal(scores['source_boxes'],oo['source_boxes']) and np.array_equal(scores['queries'],scores['scores'].argmax(1));assert np.array_equal(ps['Frozen_A'],op['intact_live_A']) and np.array_equal(ps['Frozen_F'],op['intact_live_F']);key=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();d=np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz'));xyz=z['xyz'][0];box=scores['source_boxes'];inside=((xyz[None]>=box[:,None,:3]-box[:,None,3:]/2)&(xyz[None]<=box[:,None,:3]+box[:,None,3:]/2)).all(-1);ids=z['query_points_sample_inds'].astype(int);anchorinside=inside[np.arange(256),ids];visible=d['visible'].astype(bool)&(d['p2i']>=0).any(-1);sup=inside&visible[None];cnt=sup.sum(-1);edited=(~anchorinside|(d['p2i'][ids]<0).all(-1))&(cnt>=4)
    if not rec['has_B']:edited[:]=False;noB+=1
    assert np.array_equal(sup,ff['support']) and np.array_equal(edited,ff['edited']) and np.array_equal(anchorinside,ff['anchor_inside']);perm=ff['permutation'];assert np.array_equal(np.sort(perm),np.arange(256)) and np.array_equal(edited[perm],edited);editrows+=bool(edited.any());editqueries+=int(edited.sum())
    for view in ['A','F']:
     P=torch.from_numpy(z['vis0'][0].copy())
     if view=='F':P=torch.where(torch.from_numpy(old['valid_B'])[:,None],(P+torch.from_numpy(old['B_vis']))/2,P)
     rgb=source_rgb(P,d,W);err=float(abs(rgb.numpy()-ff[view+'_point_RGB']).max());mxrgb=max(mxrgb,err);assert err<3e-5,(D.name,sid,view,'RGB',err);late=W['gate_late'].tanh()*linear(rgb[ids],W,'late')*torch.from_numpy((d['p2i'][ids]>=0).any(-1))[:,None];pooled=torch.from_numpy(sup.astype(np.float32))@rgb/torch.from_numpy(cnt.clip(1)).float()[:,None];patch=W['gate_late'].tanh()*linear(pooled,W,'late');new=torch.where(torch.from_numpy(edited)[:,None],patch,late);wrong=torch.where(torch.from_numpy(edited)[:,None],new[perm],late)
     for name,v in [('late',late),('rebind',new),('wrong',wrong)]:
      err=float(abs(v.numpy()-ff[view+'_'+name]).max());mxlate=max(mxlate,err);assert err<3e-5,(D.name,sid,view,name,err)
     a=tensors({k:old['intact_'+view+'_'+k] for k in ['h','u','late','local','own','text','relation_text','mask']})
     for name,ll in [('rebind',new),('wrong_rebind',wrong)]:
      expect=read135(a,rt,W,late=ll) if rec['has_B'] else ps['Frozen_A'];err=float(abs(expect-ps[view+'_'+name]).max());mxprob=max(mxprob,err);assert err<5e-6,(D.name,sid,view,name,err)
    if not rec['has_B']:
     for name in ['A_rebind','F_rebind','A_wrong_rebind','F_wrong_rebind']:assert np.array_equal(ps[name],ps['Frozen_A'])
    n+=1
    if n%128==0:print('AUDIT137',n,flush=True)
 assert n==(12 if args.engineering else 1798);o={'status':'PASS','n':n,'no_B_complete_fallback':noB,'edited_rows':editrows,'edited_queries':editqueries,'CPU_RGB_max_error':mxrgb,'CPU_late_max_error':mxlate,'CPU_probability_max_error':mxprob,'FINAL_access':False,'GT_used':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),o);print(o,flush=True)
if __name__=='__main__':main()
