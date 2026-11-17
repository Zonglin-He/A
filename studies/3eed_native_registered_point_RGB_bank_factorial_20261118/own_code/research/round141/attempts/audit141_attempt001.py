"""CPU native full reader, physical correspondence and separate-image CNN audit."""
from common141 import *
import numpy as np,torch,argparse
from torch.nn import functional as F
from PIL import Image
from scipy.spatial import cKDTree
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');ap.add_argument('--attempt',type=int,default=1);args=ap.parse_args();assert args.attempt>=1 and (args.engineering or args.attempt==1);argv=sys.argv;sys.argv=[argv[0]];sys.path.insert(0,str(R/'research/round135'));from audit135 import read135,tensors;sys.argv=argv
sys.path[:0]=[str(R/'crossplatform/round7'),str(R/'crossplatform/round12')]
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
from temporal_association import raw_to_model_transform
ARMS=['F_RGBB_only','F_nativeB_only','F_nativeB_RGBB','F_nativeB_RGBB_wrong','F_extraA_duplicateA']
def raw(r):
 p=Path(r['pointpath']);a=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);return np.asarray(a[:,:3],float)
def cam(r):return CameraSpec.from_camera_fields(r['platform'],read_camera_allowlist(r['metapath']),waymo_view_suffix=int(r.get('suffix') or 0),missing_distortion_policy='assume_zero_declared')
def tx(x,T):
 t=np.asarray(T);return x@t[:3,:3].T+t[:3,3]
def sparse_graph(points,bg,wh,visibility_override=None,near=None):
 uv=points['uv'];visible=points['in_image'].copy();ids=np.flatnonzero(bg['in_image']);pix=np.floor(bg['uv'][ids]/4).astype(int);bw=(wh[0]+3)//4;buf=np.full(bw*((wh[1]+3)//4),np.inf);np.minimum.at(buf,pix[:,1]*bw+pix[:,0],bg['depth'][ids]);vi=np.flatnonzero(visible);pp=np.floor(uv[vi]/4).astype(int);depth=buf[pp[:,1]*bw+pp[:,0]];visible[vi]&=np.isfinite(depth)&(abs(points['depth'][vi]-depth)<=.3+.02*points['depth'][vi])
 if near is not None:visible&=near<=.25
 if visibility_override is not None:visible[:len(visibility_override)]=visibility_override
 q=np.where(np.isfinite(uv),uv,0)*np.array([384/wh[0],256/wh[1]])/8;ij=np.floor(q).astype(int);N=len(q);p2i=np.full((N,9),-1,int);pw=np.zeros((N,9),np.float32);offset=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
 for i in np.flatnonzero(visible):
  p=ij[i]+offset;ok=(p[:,0]>=0)&(p[:,0]<48)&(p[:,1]>=0)&(p[:,1]<32);p2i[i,ok]=p[ok,1]*48+p[ok,0];pw[i,ok]=np.exp(-((p[ok]+.5-q[i])**2).sum(1)/2)
 reverse=np.full((1536,8),-1,int);rw=np.zeros((1536,8),np.float32);vv=np.flatnonzero(visible)
 if len(vv):
  yy,xx=np.indices((32,48));centers=np.c_[xx.ravel()+.5,yy.ravel()+.5];dist,ix=cKDTree(q[vv]).query(centers,k=min(8,len(vv)));dist=np.asarray(dist).reshape(1536,-1);ix=np.asarray(ix).reshape(1536,-1);keys=vv[ix];z=points['depth'][keys];front=np.where(dist<=1.5,z,np.inf).min(1)
  for j in range(1536):
   good=(dist[j]<=1.5)&(z[j]<=front[j]+.5+.03*front[j]);kk=keys[j,good];reverse[j,:len(kk)]=kk;rw[j,:len(kk)]=np.exp(-dist[j,good]**2/2)
 return {'p2i':p2i,'p2i_weight':pw,'i2p':reverse,'i2p_weight':rw,'visible':visible}
def extra_route(rt,boxes,xyz):
 out={k:v.copy() for k,v in rt.items()};own=((np.abs(xyz[None]-boxes[:,None,:3]))<=boxes[:,None,3:]/2).all(-1);more=np.zeros(rt['supports'].shape[:2]+(len(xyz),),np.float32)
 for e in range(more.shape[0]):
  for j,q in enumerate(rt['hypotheses_PRIVATE'][e]):
   if q>=0:more[e,j]=own[int(q)]
 out['supports']=np.concatenate([rt['supports'],more],-1);out['own_support']=np.concatenate([rt['own_support'],own.astype(np.float32)],-1);return out
def merged(gs):
 return {'p2i':np.concatenate([np.where(g['p2i']>=0,g['p2i']+1536*i,-1) for i,g in enumerate(gs)],1),'p2i_weight':np.concatenate([g['p2i_weight'] for g in gs],1),'i2p':np.concatenate([g['i2p'] for g in gs]),'i2p_weight':np.concatenate([g['i2p_weight'] for g in gs]),'visible':np.logical_or.reduce([g['visible'] for g in gs])}
def cnn(image,W):
 x=torch.from_numpy(image.copy()).permute(2,0,1)[None].float()/255;x=(x-.5)*2
 for i,groups in [(0,4),(3,4),(6,8)]:
  x=F.conv2d(x,W[f'rgb.{i}.weight'],W[f'rgb.{i}.bias'],stride=2,padding=1);x=F.group_norm(x,groups,W[f'rgb.{i+1}.weight'],W[f'rgb.{i+1}.bias']);x=F.gelu(x)
 return x.numpy()
def local_reads(query,key,value,route,W,heads):
 query=query[:,0];key=key[:,0];value=value[:,0];D=query.shape[-1];hd=D//heads;ww=W['in_proj_weight'];bb=W['in_proj_bias'];q=F.linear(query,ww[:D],bb[:D]).reshape(-1,heads,hd).transpose(0,1);k=F.linear(key,ww[D:2*D],bb[D:2*D]).reshape(-1,heads,hd).transpose(0,1);v=F.linear(value,ww[2*D:],bb[2*D:]).reshape(-1,heads,hd).transpose(0,1);att=(q@k.transpose(-1,-2)/np.sqrt(hd)).softmax(-1);text=[];rel=[];reads=[];own=query.new_zeros((256,D));tm=route['token_roles']
 def pool(a,m):
  weight=a[:,None]*m[None];weight/=weight.sum(-1,keepdim=True).clamp_min(1e-12);out=(weight@v).transpose(0,1).reshape(len(m),D);out=F.linear(out,W['out_proj.weight'],W['out_proj.bias']);return out*m.any(-1)[:,None]
 for e,m in enumerate(tm):
  text.append(query[m].mean(0) if m.any() else query.sum(0)*0);rr=route['relation_tokens'][e];rel.append(query[rr].mean(0) if rr.any() else query.sum(0)*0);a=att[:,m].mean(1) if m.any() else att.mean(1)*0+1/len(key);reads.append(pool(a,route['supports'][e]))
  if e==0:own=pool(a,route['own_support'])
 return {'local':torch.stack(reads) if reads else query.new_zeros((0,3,D)),'own':own,'text':torch.stack(text) if text else query.new_zeros((0,D)),'relation_text':torch.stack(rel) if rel else query.new_zeros((0,D))}
def main():
 check();torch.set_num_threads(4);root=B/(('engineering' if args.attempt==1 else 'engineering_attempt'+str(args.attempt).zfill(3)) if args.engineering else 'formal');seal=rd(root/'SEALED.json');reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));W={s:tensors(dict(np.load(R/'crossplatform/round135'/(s+'_SOURCE_MODEL_PRIVATE.npz')))) for s in ['waymo','drone','quad']};n=0;nr=0;maxp=0.;maxgraph=0.;maxcnn=0.;maxnear=0.;maxlocal=0.;status=[];cross={}
 if args.engineering:
  for src in ['waymo','drone','quad']:
   cf=next(c for c in rd(R/'research/round97/experiment/DESIGN_LOCK.json')['sources'] if c['source']==src);assert sha(cf['checkpoint_path'])==cf['checkpoint_sha256'];sd=torch.load(cf['checkpoint_path'],map_location='cpu',weights_only=True)['model'];cw=np.load(root/(src+'_BASE_LASTCROSS_PRIVATE.npz'));prefix=str(cw['module_name'])+'.';names=['in_proj_weight','in_proj_bias','out_proj.weight','out_proj.bias']
   for name in names:assert np.array_equal(cw[name],sd[prefix+name].numpy())
   cross[src]=(tensors({name:cw[name] for name in names}),int(cw['n_heads']));del sd
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name;a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));ff=np.load(D/'features_PRIVATE'/(sid+'.npz'));banks=np.load(D/'banks_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));old=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));oldps=dict(zip(old['arm_names'],old['scores']));z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['source_boxes'],z['boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.isfinite(a['scores']).all() and np.allclose(a['scores'].sum(1),1,atol=1e-9)
  for key in ['Frozen_A','Frozen_F','R135_A','R135_F','C_B_A','direct_C_B_A']:assert np.array_equal(ps[key],oldps[key])
  assert rec['source_parameters_updated']==0 and rec['has_B']==e['has_B']
  if not e['has_B']:
   for key in ARMS:assert np.array_equal(ps[key],ps['Frozen_A'])
  else:
   nr+=1;rt0=dict(np.load(R/'crossplatform/round135/formal'/name/'route_PRIVATE'/(sid+'.npz')));gkey=hashlib.sha256(z['xyz'].tobytes()+z['point_indices'].tobytes()).hexdigest();gA=dict(np.load(R/'crossplatform/round97/shared_physical_geometry_PRIVATE'/(sid+'_'+gkey[:20]+'.npz')));A=e['A_sensor'];Bb=e['pair_PRIVATE']['B'];assert sha(A['pointpath'])==A['point_sha256'] and sha(Bb['pointpath'])==Bb['point_sha256'] and sha(A['imagepath'])==A['image_sha256'] and sha(Bb['imagepath'])==Bb['image_sha256'];regi=rd(R/e['registration_path']);ca,cb=cam(A),cam(Bb);Ma,Mb=raw_to_model_transform(ca),raw_to_model_transform(cb);rawA,rawB=raw(A),raw(Bb);fullA=tx(rawA,Ma);tree=cKDTree(fullA);near=tree.query(banks['B_all_xyz'])[0];new=cKDTree(z['xyz'][0]).query(banks['B_all_xyz'])[0]>1e-4;ids=np.flatnonzero((near<=.25)&new)[:512];assert np.array_equal(ids,banks['B_selected_indices']);assert np.array_equal(banks['B_extra_xyz'],banks['B_all_xyz'][ids]) and np.array_equal(banks['B_extra_features'],banks['B_all_features'][ids]);k=len(ids);assert k==rec['extra_native_B_tokens'];salt='R106sample|'+e['pair_PRIVATE']['pair_PRIVATE']+'|real|B';rng=np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16));indices=rng.choice(len(rawB),16384,replace=len(rawB)<16384);sample=np.ascontiguousarray(tx(tx(rawB[indices],regi['forward']['T_raw_B_to_A']),Ma),np.float32);assert hashlib.sha256(sample.tobytes()).hexdigest()==rec['B_sample_sha256'];ne=float(cKDTree(sample).query(banks['B_all_xyz'])[0].max());maxnear=max(maxnear,ne);assert ne<1e-4
   ax=banks['A_extra_xyz'];assert len(ax)==k and tree.query(ax)[0].max(initial=0)<1e-4;whA=Image.open(A['imagepath']).size;whB=Image.open(Bb['imagepath']).size
   def ga(extra):
    if not len(extra):return {key:gA[key].copy() for key in ['p2i','p2i_weight','i2p','i2p_weight','visible']}
    xyz=np.concatenate([z['xyz'][0],extra]);p=project_model_points(xyz,ca,whA);bg=project_model_points(fullA,ca,whA);out=sparse_graph(p,bg,whA,gA['visible'])
    for key in ['p2i','p2i_weight','visible']:out[key][:1024]=gA[key]
    return out
   def gb(extra):
    xyz=np.concatenate([z['xyz'][0],extra]);points_rawB=tx(tx(xyz,np.linalg.inv(Ma)),np.linalg.inv(np.asarray(regi['forward']['T_raw_B_to_A'])));modB=tx(points_rawB,Mb);p=project_model_points(modB,cb,whB);bg=project_model_points(tx(rawB,Mb),cb,whB);return sparse_graph(p,bg,whB,near=cKDTree(rawB).query(points_rawB)[0])
   orig={key:gA[key] for key in ['p2i','p2i_weight','i2p','i2p_weight','visible']};gB0=gb(np.empty((0,3)));gAb=ga(banks['B_extra_xyz']);gBb=gb(banks['B_extra_xyz']);gAa=ga(ax);bad={key:v.copy() for key,v in gBb.items()};visible=np.flatnonzero((gBb['p2i']>=0).any(1));perm=np.arange(1024+k);rr=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R141wrong_extraRGB').encode()).hexdigest()[:16],16));order=rr.permutation(visible)
   if len(order)>1:perm[order]=np.roll(order,1)
   inv=np.argsort(perm);bad['p2i']=gBb['p2i'][perm];bad['p2i_weight']=gBb['p2i_weight'][perm];bad['i2p']=np.where(gBb['i2p']>=0,inv[gBb['i2p'].clip(0)],-1);specs={'F_RGBB_only':([orig,gB0],rt0),'F_nativeB_only':([gAb],extra_route(rt0,z['boxes'],banks['B_extra_xyz'])),'F_nativeB_RGBB':([gAb,gBb],extra_route(rt0,z['boxes'],banks['B_extra_xyz'])),'F_nativeB_RGBB_wrong':([gAb,bad],extra_route(rt0,z['boxes'],banks['B_extra_xyz'])),'F_extraA_duplicateA':([gAa,gAa],extra_route(rt0,z['boxes'],ax))}
   for arm in ARMS:
    graphs,route=specs[arm];expected=merged(graphs)
    for key,v in expected.items():
     actual=banks[arm+'_'+key]
     if key.endswith('weight'):
      err=float(abs(v-actual).max(initial=0));maxgraph=max(maxgraph,err);assert err<1e-5,(name,arm,key,err)
     else:assert np.array_equal(v,actual),(name,arm,key)
    feat=tensors({key:ff[arm+'_'+key] for key in ['h','u','late','local','own','text','relation_text','mask']});p=read135(feat,tensors(route),W[src]);err=float(abs(p-ps[arm]).max());maxp=max(maxp,err);assert err<5e-6
    if args.engineering:
     rgbB=np.array(Image.open(Bb['imagepath']).convert('RGB').resize((384,256),Image.Resampling.BILINEAR));imgs=[rgbB,gA['image']] if len(graphs)==2 and arm!='F_extraA_duplicateA' else [gA['image'],gA['image']] if len(graphs)==2 else [gA['image']]
     for j,im in enumerate(imgs):
      err=float(abs(cnn(im,W[src])-ff[arm+'_CNN'+str(j)]).max());maxcnn=max(maxcnn,err);assert err<1e-4
     cw,heads=cross[src];local=local_reads(torch.from_numpy(ff[arm+'_cross_query']),torch.from_numpy(ff[arm+'_cross_key']),torch.from_numpy(ff[arm+'_cross_value']),tensors(route),cw,heads)
     for name,value in local.items():
      err=float(abs(value.numpy()-ff[arm+'_'+name]).max(initial=0));maxlocal=max(maxlocal,err);assert err<2e-4,(arm,name,err)
   status.append({k:rec[k] for k in ['extra_native_B_tokens','A_resample_padding','native_B_static_count','native_B_novel_count','interpolation_bridge_max_error','old_F_reader_bridge_error','RGBB_original_A_visible','RGBB_expanded_visible','seconds']}|{'source':src,'target':tgt,'id_PRIVATE':sid})
  n+=1
  if n%64==0:print('AUDIT141',n,nr,flush=True)
 assert n==(12 if args.engineering else 1798);out={'status':'PASS','n':n,'B_inputs':nr,'max_CPU_full_reader_probability_error':maxp,'max_graph_weight_error':maxgraph,'max_engineering_CPU_RGB_CNN_error':maxcnn,'max_engineering_CPU_expanded_local_read_error':maxlocal,'max_B_native_to_actual_sample_distance_m':maxnear,'fixed_A256_and_noB_complete_fallback':True,'source_parameters_updated':0,'independent_full_transformer_not_CPU_recomputed':True,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out)
 if not args.engineering:dump(H/'PROCESS_ROWS_PRIVATE.json',status)
 print(out,flush=True)
if __name__=='__main__':main()
