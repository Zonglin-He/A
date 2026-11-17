"""Exact original-rho partial adaptation on locked new ordinary A/B inputs."""
from pathlib import Path
import sys
R131=Path(__file__).resolve().parents[2];H131=R131/'research/round131';B131=R131/'crossplatform/round131'
sys.path.insert(0,str(R131/'research/round130'));from core130 import *
from prepare130 import shuffle
from support129 import geometry as support_geometry,sets as partial_sets,direct
sys.path.insert(0,str(R131/'research/round98'));from core98 import capture98
sys.path.insert(0,str(R131/'research/round106'));from core106 import interpolation106,geo
ARMS131=['zero_F','direct_partial_F','partial_F','wrong_base_F']
def context131(e):
 name=e['source']+'_to_'+e['target'];sid=e['row']['id'];p=R131/'crossplatform/round98'/name/'features_PRIVATE'/(sid+'.npz');q=R131/'crossplatform/round98'/name/'evidence_PRIVATE'/(sid+'.npz')
 with np.load(p) as f:late=f['F_late'].copy()
 with np.load(q) as f:res=f['r0'].copy()
 return torch.as_tensor(late,device='cuda'),torch.as_tensor(res,device='cuda'),{'A_late_path':str(p.relative_to(R131)),'A_r_path':str(q.relative_to(R131)),'late_file_sha256':sha(p),'r_file_sha256':sha(q),'provider':'unchanged frozen source-prepared A only; exact R129 context bridged'}
def extra131(base,A,pair,z0):
 regpath=B131/'registration_PRIVATE'/(pair['pair_PRIVATE']+'.json');reg=jread(regpath)
 if not reg['geometry_cycle_supported']:return None,{'status':reg['status'],'registration_sha256':sha(regpath)}
 sensor=pair['B'];assert sensor['eligible_for_extra_media'];actual=sha(sensor['pointpath']);assert actual==reg['B_sensor']['point_sha256'];raw=geo.points(sensor);assert np.isfinite(raw).all();rng=np.random.default_rng(int(hashlib.sha256(('R106sample|'+pair['pair_PRIVATE']+'|real|B').encode()).hexdigest()[:16],16));ids=rng.choice(len(raw),16384,replace=len(raw)<16384);pts=np.asarray(raw[ids],float);pts=geo.transform_points(pts,np.asarray(reg['forward']['T_raw_B_to_A']));fullraw=geo.transform_points(np.asarray(raw,float),np.asarray(reg['forward']['T_raw_B_to_A']));M=geo.raw_to_model_transform(geo.camera(A));pts=np.ascontiguousarray(geo.transform_points(pts,M),np.float32);full=geo.transform_points(fullraw,M)
 with torch.no_grad():out=base.backbone_net(torch.as_tensor(pts,device='cuda')[None]);v=out['fp2_features'][0].T;xyz=out['fp2_xyz'][0].cpu().numpy();aligned,valid,proof=interpolation106(z0['xyz'][0],xyz,v,full)
 aa=torch.as_tensor(z0['vis0'][0],device='cuda');aligned=torch.where(valid[:,None],aligned,aa);proof.update(status='usable' if valid.any() else 'no_seed_alignment',sample_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),point_sha256=actual,registration_sha256=sha(regpath));return ({'vis':aligned,'valid':valid} if valid.any() else None),proof
def process131(base,m,S,e,anchor,pair,engineering):
 r=e['row'];restore(m,S);assert anchor['A']['eligible_for_extra_media'];deny={f['image_sha256'] for f in jread(R131/'research/round105/FRAMES_PRIVATE.json') if f['protected_image_content_alias']};assert sha(r['imagepath']) not in deny
 with torch.no_grad():z0,g,receipt=capture98(base,r,e['source'])
 old=load(R131/'crossplatform/round97'/(e['source']+'_to_'+e['target'])/'samples_PRIVATE'/(r['id']+'.npz'));assert np.array_equal(z0['boxes'],old['source_boxes']);po=dict(zip(old['arm_names'],old['scores']));original=native_scores(torch.as_tensor(z0['query'],device='cuda'),torch.as_tensor(z0['tokens'],device='cuda'),torch.as_tensor(z0['token_mask'],device='cuda').bool()).double().softmax(0);old_error=float(abs(original.cpu().numpy()-po['original_source']).max());assert old_error<1e-7,old_error
 z=tensor_prefix(z0);d=tensor_data(g,r);late,res,provider=context131(e);obs,proof=extra131(base,anchor['A'],pair,z0) if pair is not None else (None,{'status':'no_distinct_ordinary_B_candidate'});A=z['vis0'][0];B=A if obs is None else obs['vis'];valid=torch.zeros(1024,device='cuda',dtype=torch.bool) if obs is None else obs['valid'];assert torch.equal(A[~valid],B[~valid]);FAB=torch.where(valid[:,None],(A+B)/2,A);S0,radius=start123(m,S);ff={'fixed_late':late.cpu().numpy(),'fixed_r':res.cpu().numpy(),'mask':z['token_mask'].cpu().numpy()};probs={}
 with torch.no_grad():
  for tag,P in [('teacher_A',A),('teacher_B',B),('zero_F',FAB)]:
   if tag=='teacher_B' and obs is None:lp=torch.as_tensor(probs['teacher_A'],device='cuda').log();h=torch.as_tensor(ff['teacher_A_h'],device='cuda');u=torch.as_tensor(ff['teacher_A_u'],device='cuda')
   else:lp,h,u=body127(m,base,z,d,P,late,res,True)
   probs[tag]=lp.exp().cpu().numpy();ff[tag+'_h']=h.cpu().numpy();ff[tag+'_u']=u.cpu().numpy()
 cal=jread(R131/'research/round129/SOURCE_CALIBRATION.json');sg=support_geometry(z0['xyz'][0],z0['boxes'],valid.cpu().numpy());support,C,T,diag=partial_sets(probs['teacher_A'],probs['teacher_B'],sg,cal);Cw,swaps=shuffle(support,C,sg['assignment'],e['source']+'|'+r['id']);active=bool(obs is not None and diag['has_partial_information']);effective=bool(swaps) and active;scope={'S':support,'C':C,'teacher_conditional':T};states={};traces={};gradients={};checks=[]
 for kind,labels in [('partial',C),('wrong_base',Cw)]:
  if active:
   if kind=='wrong_base' and not effective:states[kind]=states['partial'].copy();traces[kind]=traces['partial'];continue
   label='proper_base_check' if engineering and kind=='partial' else kind;delta,steps,gg,check=fit_one130(base,m,S0,z,d,FAB,late,res,sg['groups'],scope,labels,radius,engineering,label);states[kind]=delta;traces[kind]=steps;gradients.update({k.replace('proper_base_check','partial'):v for k,v in gg.items()})
   if check:checks.append(check)
  else:states[kind]=np.zeros(4896,np.float32);traces[kind]=[]
 scores={'zero_F':probs['zero_F'],'direct_partial_F':direct(probs['zero_F'],sg['groups'],support,C,T,'partial')};writes=[]
 for kind in ['partial','wrong_base']:
  restore(m,S0);put123(m,S0,torch.as_tensor(states[kind],device='cuda'));freeze91(m);changed=[k for k,v in state(m).items() if not np.array_equal(v,S0[k])];assert set(changed)<=set(NAMES);writes.append({'kind':kind,'changed_tensors':len(changed),'parameter_L2':float(flat123(m).norm())})
  if kind=='wrong_base' and active and not effective:p=scores['partial_F'];h=ff['partial_F_h'];u=ff['partial_F_u']
  elif not np.any(states[kind]):p=probs['zero_F'];h=ff['zero_F_h'];u=ff['zero_F_u']
  else:
   with torch.no_grad():lp,hh,uu=body127(m,base,z,d,FAB,late,res,True)
   p=lp.exp().cpu().numpy();h=hh.cpu().numpy();u=uu.cpu().numpy()
  scores[kind+'_F']=p;ff[kind+'_F_h']=h;ff[kind+'_F_u']=u
 ff.update(pA=probs['teacher_A'],pB=probs['teacher_B'],S=support,C=C,wrong_C=Cw,assignment=sg['assignment'],n_A=sg['n_A'],n_B=sg['n_B'],coverage_B=sg['coverage_B'],valid_B=valid.cpu().numpy(),teacher_conditional=T)
 rec={'partial_eligible':active,'scope_eligible':bool(support.sum()>=2),'effective_shuffle':effective,'identity_control':bool(active and not effective),'B_available':obs is not None,'support':proof,'parameter_radius':radius,'traces':traces,'writebacks':writes,'checks':checks,'source_A_context':provider,'source_prefix':receipt,'original_source_probability_error':old_error,'source_boxes_bitexact':True,'new_C_generated':False,'no_target_GT_used':True,**diag};restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());return scores,ff,states|gradients,rec,z0
