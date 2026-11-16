"""Native model and calibrated-bank engineering, genuine samples, no task labels."""
from core105 import *
import argparse
def safe_extra105(A,pair,z0):
 regp=B105/'registration_full_PRIVATE'/(pair['pair_PRIVATE']+'.json');reg=jread(regp)
 if not reg['geometry_cycle_supported']:return None,{'status':reg['status']}
 B=pair['B'];assert B['eligible_for_extra_media'];assert sha(B['imagepath'])==reg['B_sensor']['image_sha256'] and sha(B['pointpath'])==reg['B_sensor']['point_sha256'];g,proof=geo.additional(A,B,z0,reg);proof.update(status='usable' if g['visible'].any() else 'no_static_visible_extra_support',registration_sha256=sha(regp));return (g if g['visible'].any() else None),proof
@torch.enable_grad()
def check_derivative105(m,f):
 a,g,s=components_solve(m,f,f['r']);J=torch.cat([(g[:,:,None]*a[:,None,:]).reshape(256,-1),g],1);J-=J.mean(0,keepdim=True);G=(g@g.T)*(a@a.T+1);G-=G.mean(0,keepdim=True)+G.mean(1,keepdim=True)-G.mean();ge=float(abs(G-J@J.T).max());assert ge<1e-7;W=m.match[-1].weight.detach().double().requires_grad_(True);b=m.match[-1].bias.detach().double().requires_grad_(True);raw=F.linear(a,W,b)+f['late'].double();ss=native_scores(F.normalize(raw,dim=-1),f['u'].double(),f['mask'].bool())+f['r'];ss-=ss.mean();error=0.
 for q in [0,128,255]:
  dw,db=torch.autograd.grad(ss[q],(W,b),retain_graph=True);error=max(error,float(abs(torch.cat([dw.flatten(),db])-J[q]).max()))
 assert error<1e-9;return {'Gram_max_error':ge,'autograd_Jacobian_max_error':error}
def main():
 assert (H105/'experiment/DESIGN_LOCK.json').is_file();assert (B105/'registration_full_PRIVATE/SEALED.json').is_file();anchors=jread(H105/'FULL_ANCHORS_PRIVATE.json');aa={x['A']['metapath']:x for x in anchors};pairs={x['pair_PRIVATE']:x for x in jread(H105/'FULL_PAIRS_PRIVATE.json')};checks=[];n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   rows=jread(R105/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');eligible=[]
   for r in rows:
    ar=aa[str(Path(r['metapath']).resolve())];pid=ar['B_pair']
    if pid and jread(B105/'registration_full_PRIVATE'/(pid+'.json'))['geometry_cycle_supported']:eligible.append(r)
   assert len(eligible)>=2
   for r in eligible[:2]:
    restore(m,S)
    with torch.no_grad():z0,g,_=capture98(base,r,src)
    ar=aa[str(Path(r['metapath']).resolve())];extra,proof=safe_extra105(ar['A'],pairs[ar['B_pair']],z0);assert extra is not None;rebuilt=geo.original_rebuild(r,z0)
    for k,v in rebuilt.items():assert np.array_equal(v,g[k]),('source_geometry_kernel_changed',k)
    z=tensor_prefix(z0);d=tensor_data(g,r)
    with torch.no_grad():
     rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=native105(m,base,z,d,route,S,[]);old=load(B98/(src+'_to_'+tgt)/'samples_PRIVATE'/(r['id']+'.npz'));pp=dict(zip(old['arm_names'],old['scores']))['prepared_frozen'];err=float(abs(A['p']-torch.as_tensor(pp,device='cuda')).max());assert err<1e-7
     for gname,eg in [('duplicate',g),('unused',dict(g))]:
      if gname=='unused':eg={k:v.copy() for k,v in g.items()};eg['p2i'][:]=-1;eg['p2i_weight'][:]=0;eg['i2p'][:]=-1;eg['i2p_weight'][:]=0;eg['visible'][:]=False
      dd=data105(g,[eg],r);f=native105(m,base,z,dd,route,S,[tensor_image105(eg)]);e=float(abs(f['p']-A['p']).max());assert e<1e-5;checks.append({'source':src,'target':tgt,'control':gname,'native_probability_error':e})
     dd=data105(g,[extra],r);images=[tensor_image105(extra)];AB=native105(m,base,z,dd,route,S,images);er=float(abs(read105(m,AB)-AB['p']).max());assert er<1e-7;deriv=check_derivative105(m,AB);q=(A['p']+AB['p'])/2
    st,trials,info=lm105(m,[A,AB],q.log()[None].repeat(2,1),head)
    with torch.no_grad():
     put94(m,st);m.route=route;m.use_interface=True
     with banks105(m,images):p,_,e=m.main(base,z,dd,'bidirectional')
     exact=float(abs(p-read105(m,AB)).max());assert exact<1e-7;re=float(abs(e['residual'].double()-AB['r']).max());assert re==0.
    checks.append({'source':src,'target':tgt,'native_A_probability_error':err,'native_AB_cached_error':er,'patched_AB_native_error':exact,'per_view_residual_invariance':re,'extra_support':proof,'fit':info,**deriv});restore(m,S);n+=1;print('ENGINEERING105',src,tgt,n,flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(H105/'ENGINEERING.json',{'state':'PASS_NATIVE_A_GRAPH_AND_UNUSED_DUPLICATE_RGB_BANK_CONTROLS_REAL_EXTRA_VIEW_AND_PATCHED_HEAD_PARITY','n':n,'checks':checks,'GT_read':False,'FINAL_access':False})
if __name__=='__main__':main()
