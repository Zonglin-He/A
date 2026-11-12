"""Predict all exposed cases, no labels, no source training, no new losses."""
from core82 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
p=read(H82/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H82/'experiment/CODE_LOCK.json'));base=load_source(read(ROOT/p['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/p['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);O=B82/'prediction2640_attempt001';assert not O.exists();O.mkdir(parents=True)
for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir()
start=time.perf_counter();statuses=[];reused=trials=0;maxraw=0.
for i,r in enumerate(allrows80()):
 sid=r['id'];z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im);old=load(B80/'prediction2640_attempt001/samples_PRIVATE'/(sid+'.npz'));ref=dict(zip(old['arm_names'],old['scores']));probs={a:ref[a] for a in ['source','frozen_bi','frozen_direct3','frozen_ABC','R59']};probs.update(R80_raw_correct=ref['uniform_sgd1'],R80_raw_wrong=ref['uniform_wrong_sgd1']);rec={}
 with torch.no_grad():refp,refq,_=m.main(base,z,d,'bidirectional');assert np.max(abs(refp.cpu().numpy()-ref['frozen_bi']))<1e-10;hp,hi,hw=holdout81(r,d);held0=float(aux82(m,z,d,hp,hi,hw));truth0=float(aux82(m,z,d,pm,im,ws))
 rawold=read(B80/'prediction2640_attempt001/records_PRIVATE'/(sid+'.json'));rawheld={}
 for wrong in [False,True]:
  arm='bt_wrong' if wrong else 'bt_correct';g,delta,ini=gradient79(m,base,z,d,refp,refq,pm,im,ws,wrong);assert torch.isfinite(g).all();set_step(m,S,delta);candidate=patch82(m);rp=rawpatch82(r,wrong);err=max(float(np.max(abs(candidate[n]-rp[n]))) for n in candidate);maxraw=max(maxraw,err);assert err<=1e-7
  put_patch(m,rp)
  with torch.no_grad():rawheld[arm]=float(aux82(m,z,d,hp,hi,hw));raw_true=float(aux82(m,z,d,pm,im,ws));rawpp,_,_=m.main(base,z,d,'bidirectional');assert np.max(abs(rawpp.cpu().numpy()-ref['uniform_wrong_sgd1' if wrong else 'uniform_sgd1']))<1e-10
  restore(m,S);pp,patch,record=search82(m,base,z,d,S,refp,refq,r,g,ini['objective_before'],pm,im,ws,wrong);record.update(heldout_true_auxiliary_before=held0,true_training_auxiliary_before=truth0,raw_heldout_true_auxiliary_after=rawheld[arm],raw_true_training_auxiliary_after=raw_true,raw_gradient_state_parity_error=err,raw_record=rawold['uniform_wrong_sgd1' if wrong else 'uniform_sgd1'])
  rec[arm]=record;probs[arm]=pp;reused+=int(record['reused_completed_R81_correct_search']);trials+=sum(not t['reused_R81_trace'] for t in record['attempts']);np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+arm+'.npz'),**patch)
 ss=np.stack([probs[a] for a in ARMS82]);qq=ss.argmax(1);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS82),scores=ss,queries=qq,source_boxes=z0['boxes'],selected_original_boxes=z0['boxes'][qq]);dump(O/'records_PRIVATE'/(sid+'.json'),rec);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':not bool(d['visible'].any()) if a not in ['source','frozen_ABC','R59'] else False,'fallback':a in rec and not rec[a]['accepted'],'no_parameter_update':not(a in rec and rec[a]['accepted']) if a not in ['R80_raw_correct','R80_raw_wrong'] else False} for a in ARMS82}})
 if(i+1)%128==0:print('R82',i+1,'seconds',round(time.perf_counter()-start,1),'reused',reused,flush=True)
assert fingerprint(base)==fp;assert all(np.array_equal(v.detach().cpu().numpy(),S[n]) for n,v in m.state_dict().items());cost={'n':2640,'states':5280,'completed_correct_searches_reused':reused,'new_trial_forward_objectives':trials,'max_R80_raw_direction_state_error':maxraw,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'FINAL_access':False};dump(O/'READOUT_STATUS_PRIVATE.json',statuses);dump(O/'COUNTS.json',cost);seal(O,'R82_FIXED_DIRECTION_CORRECT_AND_ORIGINAL_WRONG_BINDING_PREDICTIONS_SEALED',n=2640,arm_names=ARMS82);print(cost)
