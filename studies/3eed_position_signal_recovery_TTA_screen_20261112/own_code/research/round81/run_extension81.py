"""One fixed position/signal pair plus matched controls on existing larger cohorts."""
from core81 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H81/'experiment/DESIGN_LOCK.json');pin(p);lock=read(H81/'experiment/EXTENSION_LOCK.json');pin(lock);pin(read(H81/'experiment/EXTENSION_CODE_LOCK.json'));base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);O=B81/'extension2304_attempt001';assert not O.exists();O.mkdir(parents=True)
for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir()
arms=lock['arms'];configs=[tuple(c) for c in lock['correct_configs']];start=time.perf_counter();statuses=[];tries=accepted=0
for i,r in enumerate(allrows80()[336:]):
 sid=r['id'];z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);old,probs=reference80(r);assert np.array_equal(z0['boxes'],old['source_boxes'])
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-probs['frozen_bi']))<1e-10
 gg,initial=gradients81(m,z,d,pm,im,ws);rec={}
 for scope,sig,shuffled in [(s,t,False) for s,t in configs]+[(lock['selected_scope'],lock['selected_signal'],True)]:
  arm=scope+'__'+sig+('__shuffle' if shuffled else '');perm=target_permutations(r,d,pm,im) if shuffled else None;cg,ini=gradients81(m,z,d,pm,im,ws,perm) if shuffled else (gg,initial);pp,patch,record=step81(m,base,z,d,S,refp,refq,r,scope,sig,cg,ini,pm,im,ws,perm);probs[arm]=pp;rec[arm]=record;tries+=record['attempt_count'];accepted+=int(record['accepted']);np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch)
 ss=np.stack([probs[a] for a in arms]);q=ss.argmax(1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(arms),scores=ss,queries=q,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][q]);dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':not bool(d['visible'].any()) if a in rec else False,'fallback':a in rec and not rec[a]['accepted'],'no_parameter_update':not(a in rec and rec[a]['accepted']) if a!='R80_uniform_raw' else False} for a in arms}})
 if(i+1)%128==0:print('EXT81',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':2304,'correct_configurations':len(configs),'attempted_proposals':tries,'accepted_updates':accepted,'states':2304*(len(configs)+1),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'configuration_retuning_after_DEV':False});seal(O,'R81_LOCKED_SELECTED_POSITION_SIGNAL_AND_CONTROLS_SEALED_BEFORE_EXTENSION_GT',n=2304,arm_names=arms);print(read(O/'COUNTS.json'))
