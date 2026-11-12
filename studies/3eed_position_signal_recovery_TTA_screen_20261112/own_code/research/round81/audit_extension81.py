"""All accepted parameter states/final losses/readouts; DEV already replayed full search."""
from core81 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H81/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H81/'experiment/EXTENSION_AUDIT_CODE_LOCK.json'));lock=read(H81/'experiment/EXTENSION_LOCK.json');pin(lock);assert read(H81/'independent_dev_native_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');P=B81/'extension2304_attempt001';seal0=read(P/'SEALED.json')
for f,h in seal0['files'].items():assert sha(P/f)==h
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxp=maxstate=maxgrad=maxobj=0.;states=0;endpoints=[];grid=[.5,.25,.125,.0625,.03125,.015625,.0078125]
for i,r in enumerate(allrows80()[336:]):
 sid=r['id']
 # Select exact A-prefix keys only; do not decompress unrelated B-view features.
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(sid+'.npz'),allow_pickle=False) as archive:z0={k:archive['A_'+k].copy() for k in EXT_KEYS}
 z=tensor_prefix(z0);d=tensor_data(load(ROOT/'crossplatform/round78/extension_geometry2304_attempt001/samples_PRIVATE'/(sid+'.npz')),r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));records=read(P/'records_PRIVATE'/(sid+'.json'));old,refs=reference80(r)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-refs['frozen_bi']))<1e-10;hp,hi,hw=holdout81(r,d);held0=parts81(m,z,d,hp,hi,hw)
 gg,initial=gradients81(m,z,d,pm,im,ws);enable79(m);nat=[m.auxiliary(z,d,'bidirectional',pp,ii)[1] for pp,ii in zip(pm,im)];ps=m.tta_parameters()
 for j,(k,key,c) in enumerate([('P','point',.1),('RGB','RGB',.1),('D','depth',.01)]):
  loss=torch.stack([a[key] for a in nat]).mean()*c;dx=torch.autograd.grad(loss,[v for _,v in ps],retain_graph=j<2,allow_unused=True);g=torch.cat([(torch.zeros_like(v) if x is None else x).detach().flatten() for (_,v),x in zip(ps,dx)]);maxgrad=max(maxgrad,float(abs(g-gg[k]).max()));assert maxgrad<1e-6
 pred=dict(refs)
 for scope,sig,shuffled in [(s,t,False) for s,t in lock['correct_configs']]+[(lock['selected_scope'],lock['selected_signal'],True)]:
  arm=scope+'__'+sig+('__shuffle' if shuffled else '');rec=records[arm];perm=target_permutations(r,d,pm,im) if shuffled else None;cg,ini=gradients81(m,z,d,pm,im,ws,perm) if shuffled else (gg,initial);g=scope_gradient(m,cg,scope,sig);gn=float(g.norm());assert gn==rec['gradient_norm'];before=signal81(ini,sig);assert abs(before-rec['objective_before'])<1e-12;direction=-(1/(g.norm()+1e-6)).clamp_max(1)*g;trace=rec['attempts'];assert [t['alpha'] for t in trace]==grid[:len(trace)]
  for j,t in enumerate(trace):
   bound=before+1e-4*float(g.double()@(t['alpha']*direction).double());assert abs(bound-t['bound'])<1e-12;assert t['accepted']==(t['objective']<=bound+1e-10);assert not t['accepted'] or j==len(trace)-1
  assert rec['accepted']==bool(trace and trace[-1]['accepted']);assert rec['alpha']==(trace[-1]['alpha'] if rec['accepted'] else 0.)
  delta=rec['alpha']*direction;patch=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));off=0
  with torch.no_grad():
   for n,v in m.tta_parameters():
    nn=v.numel();expected=torch.tensor(S[n],device='cuda')+delta[off:off+nn].reshape_as(v);maxstate=max(maxstate,float(abs(expected.cpu().numpy()-patch[n]).max()));v.copy_(torch.tensor(patch[n],device='cuda'));off+=nn
    if not paramscope(n,scope):assert np.array_equal(patch[n],S[n])
   assert maxstate<1e-10;pp,q,_=m.main(base,z,d,'bidirectional');pred[arm]=pp.cpu().numpy();keep=(refp*(refp.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()+.1*(q-refq).square().sum(-1).mean()/2;parts=parts81(m,z,d,pm,im,ws,perm);aux=signal81(parts,sig);L=float(aux.double()+keep);maxobj=max(maxobj,abs(L-rec['objective_after']));assert maxobj<1e-7
   if rec['accepted']:assert L<=rec['objective_before']+1e-10
   actual=torch.cat([(v-torch.tensor(S[n],device='cuda')).flatten() for n,v in m.tta_parameters()]);assert abs(float(actual.norm())-rec['parameter_step_L2'])<1e-10;assert rec['query_changed']==(int(pp.argmax())!=int(refp.argmax()));held=parts81(m,z,d,hp,hi,hw);assert abs(float(signal81(held,sig))-rec['heldout_true_auxiliary_after'])<1e-7
   for n,v in m.state_dict().items():
    if n not in patch:assert np.array_equal(v.cpu().numpy(),S[n])
  endpoints.append({'row':'E'+str(i+1).zfill(5),'platform':r['platform'],'arm':arm,**{k:v for k,v in rec.items() if k!='attempts'},'heldout_true_auxiliary_before':float(signal81(held0,sig).detach())});restore(m,S);states+=1
 for arm,v in pred.items():maxp=max(maxp,float(abs(v-saved['scores'][list(saved['arm_names']).index(arm)]).max()))
 assert maxp<1e-10 and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 if(i+1)%128==0:print('AUDIT_EXT81',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H81/'independent_extension_native_audit_attempt001';O.mkdir();result={'state':'PASS_R81_ALL_EXTENSION_NATIVE_GRADIENTS_ACCEPTED_STATES_FINAL_LOSSES_AND_READOUTS','n':2304,'arm_names':seal0['arm_names'],'states':states,'max_native_gradient_error':maxgrad,'max_parameter_regeneration_error':maxstate,'max_probability_error':maxp,'max_final_objective_error':maxobj,'all_rejected_trace_bounds_and_first_acceptance_metadata_checked':True,'all_rejected_forward_objectives_recomputed':False,'DEV_all_search_forward_objectives_recomputed':True,'all_final_objectives_recomputed':True,'GT_read':False,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);dump(H81/'EXTENSION_ENDPOINT_SCALAR_ROWS.json',endpoints);print(result)
