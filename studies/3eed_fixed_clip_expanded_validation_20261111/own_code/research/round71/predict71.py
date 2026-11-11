"""Five fixed readouts, including exact projection-matched visual permutation."""
from common71 import *
from scipy.special import softmax
import time
p=read(H/'experiment/PREDICT_PLAN_LOCK.json');pin(p);assert len({r['id'] for r in rows()+devrows()})==2640;out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();stats=[];start=time.perf_counter()
for cohort,rr in [('EXTENDED',rows()),('DEV336_DIAGNOSTIC',devrows())]:
 for i,r in enumerate(rr):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';dev=cohort=='DEV336_DIAGNOSTIC';prep=ROOT/('crossplatform/round70/prepared336_attempt001' if dev else p['prepared_root']);clip=ROOT/('crossplatform/round70/clip336_attempt001' if dev else p['clip_root']);baseline=ROOT/('crossplatform/round70/predicted336_attempt001' if dev else p['baseline_root']);a=load(prep/rel);c=load(clip/rel);b=load(baseline/rel,['arm_names','scores','source_boxes','source_scores']);lookup=dict(zip(b['arm_names'],b['scores']));assert np.array_equal(a['boxes'],b['source_boxes']) and np.array_equal(a['scores'],b['source_scores']);scores=[lookup[k] for k in ARMS[:3]];base=np.log(np.maximum(a['scores'].astype(float),1e-300));scores += [softmax(base+control_evidence(a,c,shuffle)) for shuffle in [False,True]];scores=np.stack(scores);assert np.isfinite(scores).all() and scores.shape==(5,256)
  if dev:assert np.array_equal(scores[3],lookup['A_projectonly_joint'])
  q=scores.argmax(1);np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=b['source_boxes'],source_scores=b['source_scores'],selected_original_boxes=b['source_boxes'][q]);unknown=not bool((a['project_reliability']>0).any());stats.append({'id':sid,'arm_statuses':{k:{'unknown':unknown if k.startswith('CLIP') else False,'fallback':unknown if k.startswith('CLIP') else False,'no_parameter_update':True} for k in ARMS}})
  if(i+1)%128==0:print('PRED71',cohort,i+1,len(rr),flush=True)
dump(out/'READOUT_STATUS_PRIVATE.json',stats);dump(out/'COSTS.json',{'n':2640,'new_CLIP_inputs':2304,'cached_DEV_control':336,'seconds':time.perf_counter()-start,'source_training_or_parameter_updates':0});seal(out,'R71_FIVE_FIXED_READOUTS_ALL2640_SEALED_BEFORE_LABEL_LOOKUP',n=2640,arm_names=ARMS);print({'n':2640,'arms':5})
