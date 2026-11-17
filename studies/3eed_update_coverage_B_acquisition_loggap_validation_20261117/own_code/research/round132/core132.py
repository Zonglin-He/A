"""Same R131 network, step and geometry; selected sensor B supplied explicitly."""
from common132 import *
import sys
sys.path.insert(0,str(R132/'research/round131'));from core131 import *
# Keep round-specific names; historical wildcard helpers export generic R/H/B.
from common132 import R132,H132,B132,rd,dump,sha,utc,basecheck
def extra132(base,A,pair,z0,regpath):
 reg=rd(R132/regpath)
 if not reg['geometry_cycle_supported']:return None,{'status':reg['status']}
 sensor=pair['B'];assert sensor['eligible_for_extra_media'];actual=sha(sensor['pointpath']);assert actual==reg['B_sensor']['point_sha256'];raw=geo.points(sensor);assert np.isfinite(raw).all();rng=np.random.default_rng(int(hashlib.sha256(('R106sample|'+pair['pair_PRIVATE']+'|real|B').encode()).hexdigest()[:16],16));ids=rng.choice(len(raw),16384,replace=len(raw)<16384);pts=np.asarray(raw[ids],float);pts=geo.transform_points(pts,np.asarray(reg['forward']['T_raw_B_to_A']));fullraw=geo.transform_points(np.asarray(raw,float),np.asarray(reg['forward']['T_raw_B_to_A']));M=geo.raw_to_model_transform(geo.camera(A));pts=np.ascontiguousarray(geo.transform_points(pts,M),np.float32);full=geo.transform_points(fullraw,M)
 with torch.no_grad():out=base.backbone_net(torch.as_tensor(pts,device='cuda')[None]);v=out['fp2_features'][0].T;xyz=out['fp2_xyz'][0].cpu().numpy();aligned,valid,proof=interpolation106(z0['xyz'][0],xyz,v,full)
 aa=torch.as_tensor(z0['vis0'][0],device='cuda');aligned=torch.where(valid[:,None],aligned,aa);proof.update(status='usable' if valid.any() else 'no_seed_alignment',sample_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),point_sha256=actual,registration_path=regpath,registration_sha256=sha(R132/regpath));return ({'vis':aligned,'valid':valid} if valid.any() else None),proof
def source132(base,m,S,e,pairs):
 entry=e['entry'];r=entry['row'];src=entry['source'];restore(m,S);z0=load(R132/entry['files']['prefix']['path']);g=load(R132/entry['files']['geometry']['path']);g['query_anchor_indices']=z0['query_points_sample_inds'];z=tensor_prefix(z0);d=tensor_data(g,r)
 with np.load(R132/f'crossplatform/round124/formal/{src}/features_PRIVATE/{r["id"]}.npz') as f:late=torch.as_tensor(f['fixed_late'],device='cuda');res=torch.as_tensor(f['fixed_r'],device='cuda')
 sel=rd(B132/'selections_PRIVATE'/('SOURCE_'+e['A']['frame_PRIVATE']+'.json')) if e['pair_PRIVATE'] else None;obs,proof=extra132(base,e['A'],pairs[e['pair_PRIVATE']],z0,sel['selected_registration']) if sel and sel['selected_rank'] else (None,{'status':'no_source_B_registration'});start123(m,S);A=z['vis0'][0];B=A if obs is None else obs['vis'];valid=np.zeros(1024,bool) if obs is None else obs['valid'].cpu().numpy();ff={'fixed_late':late.cpu().numpy(),'fixed_r':res.cpu().numpy(),'mask':z['token_mask'].cpu().numpy(),'valid_B':valid};probs=[]
 with torch.no_grad():
  for tag,P in [('A',A),('B',B)]:
   lp,h,u=body127(m,base,z,d,P,late,res,True);ff[tag+'_h']=h.cpu().numpy();ff[tag+'_u']=u.cpu().numpy();ff['p'+tag]=lp.exp().cpu().numpy();probs.append(ff['p'+tag])
 with np.load(R132/f'crossplatform/round124/formal/{src}/samples_PRIVATE/{r["id"]}.npz') as f:old=dict(zip(f['arm_names'],f['scores']))['zero'];error=float(abs(old-probs[0]).max());assert error<5e-6,error
 sg=support_geometry(z0['xyz'][0],z0['boxes'],valid);Sg=(sg['n_A']>=4)&(sg['n_B']>=4)&(sg['coverage_B']>=.5);ff.update(assignment=sg['assignment'],n_A=sg['n_A'],n_B=sg['n_B'],S=Sg);restore(m,S);return ff,{'B_available':obs is not None,'old_source_A_probability_error':error,'support':proof,'task_labels_read':False}
def fit132(base,m,S,z0,d,late,res,A,B,valid,views,sg,support,C,T,engineering=False):
 S0,radius=start123(m,S);Cw,swaps=shuffle(support,C,sg['assignment'],views['salt']);active=bool(valid.any() and support.sum()>=2 and C.any() and (support&~C).any());scope={'S':support,'C':C,'teacher_conditional':T};states={};traces={};gradients={};ff=dict(views['features']);scores={'zero':views['zero'],'direct':direct(views['zero'],sg['groups'],support,C,T,'partial')};FAB=torch.where(valid[:,None],(A+B)/2,A);checks=[]
 for kind,labels in [('partial',C),('wrong',Cw)]:
  if active:
   if kind=='wrong' and not swaps:states[kind]=states['partial'].copy();traces[kind]=traces['partial'];continue
   label='proper_base_check' if engineering and kind=='partial' else kind;delta,tr,gg,check=fit_one130(base,m,S0,tensor_prefix(z0),d,FAB,late,res,sg['groups'],scope,labels,radius,engineering,label);states[kind]=delta;traces[kind]=tr;gradients.update({k.replace('proper_base_check','partial'):v for k,v in gg.items()})
   if check:checks.append(check)
  else:states[kind]=np.zeros(4896,np.float32);traces[kind]=[]
 for kind in ['partial','wrong']:
  restore(m,S0);put123(m,S0,torch.as_tensor(states[kind],device='cuda'));freeze91(m)
  if kind=='wrong' and active and not swaps:scores[kind]=scores['partial'];ff[kind+'_h']=ff['partial_h'];ff[kind+'_u']=ff['partial_u']
  elif not states[kind].any():scores[kind]=views['zero'];ff[kind+'_h']=ff['zero_h'];ff[kind+'_u']=ff['zero_u']
  else:
   with torch.no_grad():lp,h,u=body127(m,base,tensor_prefix(z0),d,FAB,late,res,True)
   scores[kind]=lp.exp().cpu().numpy();ff[kind+'_h']=h.cpu().numpy();ff[kind+'_u']=u.cpu().numpy()
  assert {k for k,v in state(m).items() if not np.array_equal(v,S0[k])}<=set(NAMES)
 ff.update(S=support,C=C,wrong_C=Cw,assignment=sg['assignment'],n_A=sg['n_A'],n_B=sg['n_B'],valid_B=valid.cpu().numpy(),teacher_conditional=T);restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());return scores,ff,states|gradients,{'partial_eligible':active,'scope_eligible':bool(support.sum()>=2),'effective_shuffle':bool(active and swaps),'identity_control':bool(active and not swaps),'parameter_radius':radius,'traces':traces,'engineering_checks':checks,'B_available':bool(valid.any()),'only4896Up':True}
