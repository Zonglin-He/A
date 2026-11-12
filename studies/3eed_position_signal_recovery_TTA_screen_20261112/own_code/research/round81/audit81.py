"""Separate native-gradient check, accepted-state regeneration and exact readout replay."""
from core81 import *
import argparse,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','extension']);a=ap.parse_args();dev=a.stage=='dev';p=read(H81/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H81/'experiment/AUDIT_CODE_LOCK.json'));P=B81/('dev336_attempt001' if dev else 'extension2304_attempt001');seal0=read(P/'SEALED.json')
for f,h in seal0['files'].items():assert sha(P/f)==h
if dev:rr=target_rows();configs=CONFIGS
else:rr=allrows80()[336:];lock=read(H81/'experiment/EXTENSION_LOCK.json');configs=[tuple(x) for x in lock['correct_configs']]
base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);start=time.perf_counter();maxp=maxstate=maxgrad=0.;states=0;endpoint=[]
for i,r in enumerate(rr):
 sid=r['id'];z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);saved=load(P/'samples_PRIVATE'/(sid+'.npz'));recs=read(P/'records_PRIVATE'/(sid+'.json'));old,refs=reference80(r)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-refs['frozen_bi']))<1e-10;hp,hi,hw=holdout81(r,d);held0=parts81(m,z,d,hp,hi,hw)
 gg,initial=gradients81(m,z,d,pm,im,ws);enable79(m);native=[m.auxiliary(z,d,'bidirectional',pp,ii)[1] for pp,ii in zip(pm,im)];nativeparts={k:torch.stack([n[key] for n in native]).mean()*c for k,key,c in [('P','point',.1),('RGB','RGB',.1),('D','depth',.01)]};ps=m.tta_parameters()
 for j,k in enumerate(['P','RGB','D']):
  grad=torch.autograd.grad(nativeparts[k],[v for _,v in ps],retain_graph=j<2,allow_unused=True);g=torch.cat([(torch.zeros_like(v) if x is None else x).detach().flatten() for (_,v),x in zip(ps,grad)]);maxgrad=max(maxgrad,float(abs(g-gg[k]).max()));assert maxgrad<1e-6 and abs(float(nativeparts[k].detach())-initial[k])<1e-7
 pred=dict(refs);todo=[(s,t,False) for s,t in configs]
 if not dev:todo.append((lock['selected_scope'],lock['selected_signal'],True))
 for scope,sig,shuffled in todo:
  arm=scope+'__'+sig+('__shuffle' if shuffled else '');perm=target_permutations(r,d,pm,im) if shuffled else None;cg,ini=gradients81(m,z,d,pm,im,ws,perm) if shuffled else (gg,initial);pp,patch,rec=step81(m,base,z,d,S,refp,refq,r,scope,sig,cg,ini,pm,im,ws,perm);stored=load(P/'states_PRIVATE'/(sid+'_'+arm+'.npz'));orig=recs[arm]
  assert rec['alpha']==orig['alpha'] and rec['accepted']==orig['accepted'] and rec['attempts']==orig['attempts'];assert rec['objective_after']<=rec['objective_before']+1e-10 if rec['accepted'] else True
  for n,v in patch.items():maxstate=max(maxstate,float(abs(v-stored[n]).max()))
  assert maxstate<1e-10
  with torch.no_grad():
   for n,v in m.tta_parameters():v.copy_(torch.tensor(stored[n],device='cuda'))
   for n,v in m.state_dict().items():
    if n not in stored:assert np.array_equal(v.cpu().numpy(),S[n])
   actual,_,_=m.main(base,z,d,'bidirectional');assert np.max(abs(actual.cpu().numpy()-pp))<1e-10
   # Scope support is exact: every unselected parameter remains source.
   for n,v in m.tta_parameters():
    if not paramscope(n,scope):assert np.array_equal(v.cpu().numpy(),S[n])
  pred[arm]=pp;restore(m,S);states+=1;endpoint.append({'row':('D' if dev else 'E')+str(i+1).zfill(5),'platform':r['platform'],'arm':arm,**{k:v for k,v in rec.items() if k!='attempts'},'heldout_true_auxiliary_before':float(signal81(held0,sig).detach())})
 for arm,pp in pred.items():maxp=max(maxp,float(abs(pp-saved['scores'][list(saved['arm_names']).index(arm)]).max()))
 assert maxp<1e-10 and np.array_equal(saved['source_boxes'],z0['boxes']) and np.array_equal(saved['queries'],saved['scores'].argmax(1)) and np.array_equal(saved['selected_original_boxes'],z0['boxes'][saved['queries']])
 if(i+1)%64==0:print('AUDIT81',a.stage,i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;O=H81/('independent_'+a.stage+'_native_audit_attempt001');O.mkdir();result={'state':'PASS_R81_NATIVE_COMPONENT_GRADIENTS_ARMIJO_AND_ACTUAL_STATES','n':len(rr),'arm_names':seal0['arm_names'],'states':states,'max_native_gradient_error':maxgrad,'max_parameter_regeneration_error':maxstate,'max_probability_error':maxp,'all_updates_read_task_labels':False,'GT_read':False,'canonical_source_256_boxes_and_readout_verified':True,'all_unknown_fallback_and_zero_update_rows_retained':True,'prediction_seal_sha256':sha(P/'SEALED.json'),'seconds':time.perf_counter()-start};dump(O/'AUDIT_RESULT.json',result);dump(H81/(a.stage.upper()+'_ENDPOINT_SCALAR_ROWS.json'),endpoint);print(result)
