"""Post-score exploratory control: full-fusion combined recovery vs target shuffle."""
from core81 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
lock=read(H81/'experiment/SIGNAL_CONTROL_LOCK.json');pin(lock);base=load_source(read(ROOT/read(H81/'experiment/DESIGN_LOCK.json')['source_plan']));torch.set_num_threads(4);fp=fingerprint(base);S=load(ROOT/read(H81/'experiment/DESIGN_LOCK.json')['init_state']);m=Fusion79(base.contrastive_align_projection_image).cuda().eval();restore(m,S);O=B81/'signal_control2304_attempt001';assert not O.exists();O.mkdir(parents=True)
for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir()
start=time.perf_counter();trials=0;accepted=0
for i,r in enumerate(allrows80()[336:]):
 sid=r['id'];z0,z,d=item80(r);pm,im=maskpair79(r,d);ws=uniform_weights80(d,pm,im)
 with torch.no_grad():p0,q0,_=m.main(base,z,d,'bidirectional')
 perm=target_permutations(r,d,pm,im);gg,initial=gradients81(m,z,d,pm,im,ws,perm);pp,patch,record=step81(m,base,z,d,S,p0,q0,r,'all','ALL',gg,initial,pm,im,ws,perm);np.savez_compressed(O/'states_PRIVATE'/(sid+'.npz'),**patch);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),scores=pp,query=np.array(int(pp.argmax())),source_boxes=z0['boxes'],selected_original_box=z0['boxes'][pp.argmax()]);dump(O/'records_PRIVATE'/(sid+'.json'),record);trials+=record['attempt_count'];accepted+=int(record['accepted'])
 if(i+1)%128==0:print('FULL_SIGNAL_SHUFFLE81',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
assert fingerprint(base)==fp;dump(O/'COUNTS.json',{'n':2304,'states':2304,'accepted_updates':accepted,'attempted_proposals':trials,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'method_or_hyperparameters_changed':False,'postscore_exploratory_control':True});seal(O,'R81_FULL_COMBINED_SIGNAL_SHUFFLE_SEALED_BEFORE_NEW_PREDICTION_SCORE',n=2304);print(read(O/'COUNTS.json'))
