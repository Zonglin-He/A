from common72 import *
from constraints72 import direct
import time
p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');pin(p);out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();e=ROOT/p['evidence_root'];ad=ROOT/p['adapt_root'];se=read(e/'SEALED.json');sa=read(ad/'SEALED.json');base=ROOT/'crossplatform/round71/predicted2640_attempt001';sb=read(base/'SEALED.json');stats=[];start=time.perf_counter()
for r in rows():
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz'
 for root,ss in [(e,se),(ad,sa),(base,sb)]:assert sha(root/rel)==ss['files'][rel]
 ev=load(e/rel);a=load(ad/rel);b=load(base/rel);lookup=dict(zip(b['arm_names'],b['scores']));p0=b['source_scores'].astype(float);p0=p0/p0.sum();probs={k:lookup[{'CLIP_reference':'CLIP_direct'}.get(k,k)] for k in ARMS[:4]}
 for name,key,kind in [('graph_direct_mu1','main_N','mu1'),('graph_direct_opt','main_N','opt'),('graph_direct_hard','main_N','hard'),('RGBshuffle_direct_opt','RGB_N','opt'),('Nshuffle_direct_opt','Nshuffle_N','opt'),('box_direct_opt','box_N','opt'),('color_only_direct_opt','color_N','opt'),('relation_only_direct_opt','relation_N','opt')]:probs[name]=direct(p0,ev[key][ev['query_group']],kind)
 for name in ['graph_tta1','graph_tta3','RGBshuffle_tta3','Nshuffle_tta3']:probs[name]=a[name]/a[name].sum()
 scores=np.stack([probs[k] for k in ARMS]);q=scores.argmax(1);assert np.isfinite(scores).all() and np.allclose(scores[1:].sum(1),1,atol=2e-7);np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=b['source_boxes'],source_scores=b['source_scores'],selected_original_boxes=b['source_boxes'][q]);rec=read(ad/'cases_PRIVATE'/(sid+'.json'));st={}
 for name in ARMS:
  tag='RGBshuffle' if name.startswith('RGBshuffle') else 'Nshuffle' if name.startswith('Nshuffle') else 'graph';up='_tta' in name;unknown=not bool(ev['target_known'].any());st[name]={'unknown':unknown if name not in ARMS[:4] else False,'fallback':(not rec[tag]['active']) if name not in ARMS[:4] else False,'no_parameter_update':not(up and rec[tag]['active'])}
 stats.append({'id':sid,'arm_statuses':st})
dump(out/'READOUT_STATUS_PRIVATE.json',stats);dump(out/'COSTS.json',{'n':2640,'seconds':time.perf_counter()-start,'new_full_3D_or_CLIP_forwards':0});seal(out,'R72_ALL16_READOUTS_SEALED_BEFORE_TASK_LOOKUP',n=2640,arm_names=ARMS);print({'n':2640,'arms':len(ARMS)})
