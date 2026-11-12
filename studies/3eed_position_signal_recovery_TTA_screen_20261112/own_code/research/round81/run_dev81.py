"""All DEV336 cells predicted and sealed before any task-label lookup."""
from core81 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H81/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H81/'experiment/CODE_LOCK.json'));assert read(H81/'engineering_attempt001/RESULT.json')['state'].startswith('PASS_');base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);O=B81/'dev336_attempt001';assert not O.exists();O.mkdir(parents=True)
for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir()
arms=REFS81+CONFIG_NAMES;start=time.perf_counter();statuses=[];tries=0;accepted=0
for i,r in enumerate(target_rows()):
 sid=r['id'];z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);old,probs=reference80(r)
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-probs['frozen_bi']))<1e-10
 gg,initial=gradients81(m,z,d,pm,im,ws);rec={}
 for scope,sig in CONFIGS:
  arm=scope+'__'+sig;pp,patch,record=step81(m,base,z,d,S,refp,refq,r,scope,sig,gg,initial,pm,im,ws);probs[arm]=pp;rec[arm]=record;tries+=record['attempt_count'];accepted+=int(record['accepted']);np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch)
 ss=np.stack([probs[a] for a in arms]);qq=ss.argmax(1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(arms),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':not bool(d['visible'].any()) if a in CONFIG_NAMES else False,'fallback':a in rec and not rec[a]['accepted'],'no_parameter_update':not(a in rec and rec[a]['accepted']) if a!='R80_uniform_raw' else False} for a in arms}})
 if(i+1)%32==0:print('DEV81',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',{'n':336,'configs':len(CONFIGS),'attempted_proposals':tries,'accepted_updates':accepted,'states':336*len(CONFIGS),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False});seal(O,'R81_DEV_ALL_POSITION_SIGNAL_PREDICTIONS_SEALED_BEFORE_GT',n=336,arm_names=arms);print(read(O/'COUNTS.json'))
