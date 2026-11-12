"""All-case actual-state/final-objective audit for the one extra control."""
from core81 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
lock=read(H81/'experiment/SIGNAL_CONTROL_LOCK.json');pin(lock);P=B81/'signal_control2304_attempt001';ss=read(P/'SEALED.json')
for f,h in ss['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/read(H81/'experiment/DESIGN_LOCK.json')['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/read(H81/'experiment/DESIGN_LOCK.json')['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);maxp=maxstate=maxobj=0.;start=time.perf_counter();rows=[];grid=[.5,.25,.125,.0625,.03125,.015625,.0078125]
for i,r in enumerate(allrows80()[336:]):
 sid=r['id']
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as ar:z0={k:ar['A_'+k].copy() for k in EXT_KEYS}
 z=tensor_prefix(z0);d=tensor_data(load(ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz')),r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);perm=target_permutations(r,d,pm,im);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'));patch=load(P/'states_PRIVATE'/(sid+'.npz'))
 for j in range(2):
  for k,valid in [('P',pm[j]&d['visible']),('RGB',im[j]),('D',d['depth_valid'])]:
   ids=torch.nonzero(valid,as_tuple=True)[0];assert torch.equal(perm[j][k][ids].sort().values,ids)
 with torch.no_grad():p0,q0,_=m.main(base,z,d,'bidirectional');hp,hi,hw=holdout81(r,d);held0=float(sum(parts81(m,z,d,hp,hi,hw).values()))
 gg,ini=gradients81(m,z,d,pm,im,ws,perm);g=sum(gg.values());gn=float(g.norm());assert gn==rec['gradient_norm'];before=sum(ini.values());assert abs(before-rec['objective_before'])<1e-12;direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;trace=rec['attempts'];assert [v['alpha'] for v in trace]==grid[:len(trace)]
 for j,t in enumerate(trace):
  bound=before+1e-4*float(g.double()@(t['alpha']*direction).double());assert abs(t['bound']-bound)<1e-12 and t['accepted']==(t['objective']<=bound+1e-10);assert not t['accepted'] or j==len(trace)-1
 assert rec['accepted']==bool(trace and trace[-1]['accepted']);assert rec['alpha']==(trace[-1]['alpha'] if rec['accepted'] else 0.);delta=rec['alpha']*direction;off=0
 with torch.no_grad():
  for n,v in m.tta_parameters():
   nn=v.numel();wanted=torch.tensor(S[n],device='cuda')+delta[off:off+nn].reshape_as(v);maxstate=max(maxstate,float(abs(wanted.cpu().numpy()-patch[n]).max()));v.copy_(torch.tensor(patch[n],device='cuda'));off+=nn
  assert maxstate<1e-10;pp,q,_=m.main(base,z,d,'bidirectional');maxp=max(maxp,float(abs(pp.cpu().numpy()-saved['scores']).max()));assert maxp<1e-10;keep=(p0*(p0.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-q0).square().sum(-1).mean()/2;aux=sum(parts81(m,z,d,pm,im,ws,perm).values());objective=float(aux.double()+keep);maxobj=max(maxobj,abs(objective-rec['objective_after']));assert maxobj<1e-7
  if rec['accepted']:assert objective<=rec['objective_before']+1e-10
  assert int(pp.argmax())==int(saved['query']) and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['selected_original_box'],z0['boxes'][saved['query']]);actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);assert abs(float(actual.norm())-rec['parameter_step_L2'])<1e-10
  for n,v in m.state_dict().items():
   if n not in patch:assert np.array_equal(v.cpu().numpy(),S[n])
 restore(m,S);rows.append({'row':'E'+str(i+1).zfill(5),'platform':r['platform'],'arm':'all__ALL__shuffle',**{k:v for k,v in rec.items() if k!='attempts'},'heldout_true_auxiliary_before':held0})
 if(i+1)%128==0:print('AUDIT_FULL_SHUFFLE81',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H81/'independent_signal_control_audit_attempt001';O.mkdir();result={'state':'PASS_R81_FULL_SIGNAL_SHUFFLE_NATIVE_ACTUAL_STATES_FINAL_LOSS_AND_TARGET_MULTISET','n':2304,'states':2304,'max_parameter_error':maxstate,'max_probability_error':maxp,'max_final_objective_error':maxobj,'prediction_seal_sha256':sha(P/'SEALED.json'),'GT_read':False,'seconds':time.perf_counter()-start,'postscore_exploratory_control':True};dump(O/'AUDIT_RESULT.json',result);dump(H81/'SIGNAL_CONTROL_ENDPOINT_SCALAR_ROWS.json',rows);print(result)
