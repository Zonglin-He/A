"""Source VAL unlabeled updates first; task labels join only after prediction seal."""
from adapt90 import *
from views90 import frozen_reads90
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
assert read(H90/'independent_source_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');sel=read(B90/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json');S={b:load(ROOT/sel['models'][b]['state_path']) for b in BRANCHES90};base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);torch.manual_seed(0);m=Fusion90(base.contrastive_align_projection_image).cuda().eval();load_R76_90(m);O=B90/'source_aux128_attempt001'
for k in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/k).mkdir(parents=True,exist_ok=False)
start=time.perf_counter();cost={'accepted_updates':0,'trial_calls':0};support=[]
with torch.no_grad():pass
for i,r in enumerate(source_rows('VAL128')):
 z0,z,d,rt,route,pm,im,ws=data90(r,True);m.route=route;result={};record={}
 for b in BRANCHES90:
  restore(m,S[b]);configure90(m,b);m.requires_grad_(False)
  with torch.no_grad():p,mean,feature,e=frozen_reads90(m,base,z,d,pm,im);result[b+'_frozen']=p.cpu().numpy();result[b+'_mean3']=mean.cpu().numpy();result[b+'_feature3']=feature.cpu().numpy()
  ref,out,patch,tr,refs,_=update90(m,base,z,d,pm,im,ws,3 if b=='candidate_joint' else 1);assert np.array_equal(ref,result[b+'_frozen']);record[b]=tr
  for st,prob in out.items():result[b+'_tta'+str(st)]=prob;np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+b+'_tta'+str(st)+'.npz'),**patch[st])
  cost['accepted_updates']+=sum(t['accepted'] for t in tr);cost['trial_calls']+=sum(len(t['trials']) for t in tr)
  if b=='candidate_joint':
   for variant in ['no_ref','soft_pool']:
    restore(m,S[b]);configure90(m,b,'no_ref' if variant=='no_ref' else 'full');m.pool_mode='soft' if variant=='soft_pool' else 'strict';m.requires_grad_(False)
    with torch.no_grad():result[variant+'_frozen']=m.main(base,z,d,'bidirectional')[0].cpu().numpy()
 for kind in ['bypass','physical','shuffle']:
  b='candidate_joint';restore(m,S[b]);configure90(m,b);ref,out,patch,tr,refs,_=update90(m,base,z,d,pm,im,ws,1,kind,shuffle90(r) if kind=='shuffle' else None);name=b+'_'+kind;record[name]=tr;result[name+'1']=out[1];np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'1.npz'),**patch[1]);cost['accepted_updates']+=sum(t['accepted'] for t in tr);cost['trial_calls']+=sum(len(t['trials']) for t in tr)
 # Every model/step choice was already source-selected. No task labels were accessed.
 arms=list(result);a=np.stack([result[k] for k in arms]);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),scores=a,arm_names=np.array(arms),queries=a.argmax(-1),source_boxes=z0['boxes']);dump(O/'records_PRIVATE'/(r['id']+'.json'),record)
 if (i+1)%32==0:print('SOURCE_AUX90',i+1,flush=True)
assert fingerprint(base)==fp;dump(O/'COUNTS.json',{'n':128,'arm_names':arms,**cost,'seconds':time.perf_counter()-start,'task_labels_read_by_update_producer':False,'used_for_model_or_hyperparameter_selection':False});seal(O,'R90_SOURCE_AUXILIARY_UPDATE_PREDICTIONS_SEALED_WITHOUT_TASK_LABELS',n=128,arm_names=arms);print(read(O/'COUNTS.json'))
