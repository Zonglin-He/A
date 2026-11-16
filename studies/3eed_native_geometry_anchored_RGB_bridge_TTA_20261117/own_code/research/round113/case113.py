"""A-only fitting followed by held-out observation replay; no task labels."""
from core113 import *
def groupnorm113(m,g):
 out={};off=0
 for n,p in m.tta_parameters():
  k=p.numel();out[n]=float(g[off:off+k].double().square().sum());off+=k
 return {k:float(np.sqrt(sum(v for n,v in out.items() if n.startswith(k)))) for k in ['point_adapter.','image_adapter.','point_to_image.q.','point_to_image.k.','point_to_image.v.','point_to_image.o.','image_to_point.','gate_I','gate_P']}

def fitcase113(base,m,S,src,r):
 z0,g,z,proof=current113(src,r);d=tensor_data(g,r);restore(m,S);rt,role=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);pm,im=maskpair79(r,d);ws=weight_sets({'id':'UNUSED_UNIFORM'},d,pm,im,np.zeros(1024),np.zeros(1536))[0]['uniform'];perm=permutation94(r,d,0);dw=bound94(d,perm);starts={k:initial113(m,S,k=='anchor') for k in ['full','anchor']};cache={};refs={}
 with torch.no_grad():
  for k,st in starts.items():restore(m,st);cache[k]=native_views113(m,base,z,d,route,st,pm,im);refs[k]=read113(m,cache[k],route,st)
 parents={};deltas={};grads={};grec={}
 for k in ['full','anchor','anchor_wrong']:
  start=starts['full' if k=='full' else 'anchor'];dd=dw if k=='anchor_wrong' else d;gg,raw,rec=physical_direction91(m,base,z,dd,start,pm,im,ws,False);parents[k]=(gg,raw,rec);grads[k]=gg.cpu().numpy();grec[k]={**rec,'group_gradient_L2':groupnorm113(m,gg)}
 fits={};records={}
 for setting in SETTINGS113:
  anchor=setting.startswith('anchor');which='anchor' if anchor else 'full';start=starts[which];wrong=setting.endswith('_wrong');kind='all' if setting.endswith('_all') else ('rgb' if setting=='anchor_rgb_nm' else 'bridge');natural=setting.endswith('_natural');parent=parents[which];gg=parents['anchor_wrong'][0] if wrong else parent[0]
  fits[setting],records[setting]=fit113(m,base,z,d,dw,route,start,kind,gg,parent[1],pm,im,ws,cache[which],wrong,natural)
 # Guard evidence: all fitting is complete before any C/B arrays or old C/F scores.
 stamp={'all_seven_update_states_selected':True,'before_B_C_arrays':True,'monotonic_seconds':time.monotonic()};restore(m,S);freeze91(m)
 return {'z0':z0,'g':g,'z':z,'d':d,'dw':dw,'rt':rt,'route':route,'pm':pm,'im':im,'ws':ws,'starts':starts,'cacheA':cache,'fits':fits,'updates':records,'gradients':grads,'gradient_records':grec,'proof':proof,'role':role,'C_boundary_stamp':stamp,'wrong_changed_seeds':int(np.count_nonzero(perm!=np.arange(1024)))}

def evaluate113(base,m,S,src,r,c,engineering=False):
 z0,z,d,route,pm,im=c['z0'],c['z'],c['d'],c['route'],c['pm'],c['im'];assert c['C_boundary_stamp']['all_seven_update_states_selected'];obsf=B111/(src+'_to_'+r['platform'])/'observations_PRIVATE'/(r['id']+'.npz');obs=load(obsf);oo={}
 for v in ['B','C']:
  if 'real_'+v+'_vis' in obs:oo[v]={k:torch.as_tensor(obs['real_'+v+'_'+k],device='cuda') for k in ['vis','valid']}
 zz={'A':z,'C':prefix106(z,[oo['C']],False) if 'C' in oo else z,'fusedABC':prefix106(z,list(oo.values()),True)};scores={};ff={};cpu_map={};invariant=[];cache={};er=[]
 def save(label,vv,rd):
  for k,v in arrays113(vv,rd).items():ff[label+'_'+k]=v
 def arm(name,label,rd,readout,no_late=False,no_r=False):
  scores[name]=rd[readout].detach().cpu().numpy();cpu_map[name]={'feature_key':label,'readout':readout,'no_late':no_late,'no_r':no_r}
 with torch.no_grad():
  for key,st in c['starts'].items():
   cache[key]={}
   for view,vz in zz.items():
    restore(m,st);freeze91(m);vv=c['cacheA'][key] if view=='A' else native_views113(m,base,vz,d,route,st,pm,im);cache[key][view]=vv;rd=read113(m,vv,route,st);label='frozen_'+key+'_'+view;save(label,vv,rd)
    for readout in ['mean3','feature3']:arm(key+'_frozen_'+readout+'_'+view,label,rd,readout)
    if key=='anchor' and view=='C':arm('anchor_frozen_C',label,rd,'single')
    if key=='anchor' and view=='fusedABC':
     for suffix,nl,nr in [('noLate',True,False),('lateOnly',False,True),('native',True,True)]:
      aa=read113(m,vv,route,st,nl,nr);arm('anchor_'+suffix+'_feature3_fusedABC',label,aa,'feature3',nl,nr)
      if suffix!='native':arm('anchor_'+suffix+'_fusedABC',label,aa,'single',nl,nr)
  for setting,states in c['fits'].items():
   anchor=setting.startswith('anchor');key='anchor' if anchor else 'full';st=c['starts'][key];fast=anchor and c['updates'][setting]['scope']!='all'
   for tag,th in states.items():
    restore(m,st);puttheta91(m,th);freeze91(m)
    for view,vz in zz.items():
     vv=fast_views113(m,vz,d,cache[key][view],pm,im) if fast else native_views113(m,base,vz,d,route,st,pm,im);rd=read113(m,vv,route,st);label=setting+'_'+tag+'_'+view;save(label,vv,rd);arm(label,label,rd,'single')
     if view=='fusedABC':arm(setting+'_'+tag+'_feature3_fusedABC',label,rd,'feature3')
     if fast:
      ref=read113(m,cache[key][view],route,st);assert all(torch.equal(a[k],b[k]) for a,b in zip(vv,cache[key][view]) for k in ['h','u','local','own','text','relation_text']);assert torch.equal(rd['rs'],ref['rs']);invariant.append({'setting':setting,'state':tag,'view':view,'h_u_local_text_r_bitexact':True,'late_difference_L2':float(torch.sqrt(sum((a['late']-b['late']).double().square().sum() for a,b in zip(vv,cache[key][view]))))})
      if engineering and view=='A':
       native=native_views113(m,base,vz,d,route,st,pm,im);nr=read113(m,native,route,st);err=max(float(abs(rd[k]-nr[k]).max()) for k in ['single','mean3','feature3']);assert err<1e-12 and all(torch.equal(a[k],b[k]) for a,b in zip(vv,native) for k in ['h','u','late']);er.append(err)
  # Original R97 currentA legacy raw/BT must remain exactly the same algorithm.
  old=load(B97/(src+'_to_'+r['platform'])/'samples_PRIVATE'/(r['id']+'.npz'));op=dict(zip(old['arm_names'],old['scores']));legacyerr=max(float(abs(scores['full_all_'+tag+'_A']-op['physical_'+tag]).max()) for tag in ['raw','bt']);assert legacyerr<1e-6 and all(int(scores['full_all_'+tag+'_A'].argmax())==int(op['physical_'+tag].argmax()) for tag in ['raw','bt'])
  oldrec=jread(B97/(src+'_to_'+r['platform'])/'records_PRIVATE'/(r['id']+'.json'))['updates']['physical_bt'];assert oldrec['alpha']==c['updates']['full_all']['accepted_alpha'];assert len(oldrec['trials'])==len(c['updates']['full_all']['trials'])
  # Before/after third-mask recovery is diagnostic only, after all fits and C reads.
  p3,i3=masks(r,d,76003);w3=weight_sets({'id':'UNUSED_UNIFORM'},d,[p3],[i3],np.zeros(1024),np.zeros(1536))[0]['uniform'];hold={}
  for key,st in c['starts'].items():restore(m,st);hold[key+'_frozen']=float(uniform_aux91(m,z,d,[p3],[i3],w3))
  for setting,states in c['fits'].items():
   st=c['starts']['anchor' if setting.startswith('anchor') else 'full']
   for tag,th in states.items():restore(m,st);puttheta91(m,th);hold[setting+'_'+tag]=float(uniform_aux91(m,z,d,[p3],[i3],w3))
 restore(m,S);freeze91(m);assert set(scores)==set(NEW113)
 return scores,ff,cpu_map,{'C_available':bool('C' in oo),'R111_observations_sha256':sha(obsf),'legacy_R97_raw_bt_max_probability_error':legacyerr,'legacy_R97_raw_bt_query_and_alpha_exact':True,'fast_full_native_max_error':max(er,default=0.),'invariance_checks':invariant,'holdout_mask76003_recovery':hold,'all_C_neural_after_all_fits':True}
