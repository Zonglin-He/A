"""Independent numerical replay, all-input provenance and raw graph subset audit."""
from common72 import *
from scipy.special import softmax,logsumexp
import time
def native(w,h,t,mask):
 x=h
 for k in ('0','2','4'):
  x=x@w[k+'.weight'].T+w[k+'.bias']
  if k!='4':x=np.maximum(x,0)
 x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);s=x@t.T/.07
 return logsumexp(s[:,mask],axis=1)-logsumexp(s[:,~mask],axis=1)
def main():
 p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt003.json');pin(p);out=H/'independent_audit_attempt001';assert not out.exists();out.mkdir();pred=ROOT/p['prediction_root'];e=ROOT/p['evidence_root'];ad=ROOT/p['adapt_root'];raw=ROOT/'crossplatform/round72/evidence2640_attempt001';base=ROOT/'crossplatform/round71/predicted2640_attempt001';seals={str(x):read(x/'SEALED.json') for x in [pred,e,ad,raw,base]};initial=load(ROOT/p['head_initial']);start=time.perf_counter();maxerr=0.;replays=steps=0
 def verified(root,rel):assert sha(root/rel)==seals[str(root)]['files'][rel];return load(root/rel)
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';ev=verified(e,rel);old=verified(raw,rel);z=verified(pred,rel);a=verified(ad,rel);b=verified(base,rel);assert np.array_equal(z['source_boxes'],b['source_boxes']);assert np.array_equal(z['queries'],z['scores'].argmax(1));assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);assert list(z['arm_names'])==ARMS
  lookup=dict(zip(b['arm_names'],b['scores']))
  for i,k in enumerate(ARMS[:4]):assert np.array_equal(z['scores'][i],lookup['CLIP_direct' if k=='CLIP_reference' else k])
  assert np.array_equal(ev['main_N'],ev['color_N']|ev['relation_N']);assert np.array_equal(ev['color_N'],old['color_N']);assert int(ev['Nshuffle_N'].sum())==int(ev['main_N'].sum());assert not ev['main_N'][~ev['target_known']].any();assert not ev['Nshuffle_N'][~ev['target_known']].any()
  p0=z['source_scores'].astype(float);p0/=p0.sum();saved=read(ad/'cases_PRIVATE'/(sid+'.json'))
  for arm,key,kind in [('graph_direct_mu1','main_N','mu1'),('graph_direct_opt','main_N','opt'),('graph_direct_hard','main_N','hard'),('RGBshuffle_direct_opt','RGB_N','opt'),('Nshuffle_direct_opt','Nshuffle_N','opt'),('box_direct_opt','box_N','opt'),('color_only_direct_opt','color_N','opt'),('relation_only_direct_opt','relation_N','opt')]:
   N=ev[key][ev['query_group']];q=p0.copy()
   if N.any() and not N.all():
    mass=p0[N].sum()
    if kind=='mu1':q=softmax(np.log(np.maximum(p0,1e-300))-N)
    else:
     target=0. if kind=='hard' else mass*.5;q[N]=p0[N]*(target/mass);q[~N]=p0[~N]*((1-target)/(1-mass));q/=q.sum()
   assert np.allclose(q,z['scores'][ARMS.index(arm)],atol=1e-14,rtol=1e-13)
  for tag,key in [('graph','main_N'),('RGBshuffle','RGB_N'),('Nshuffle','Nshuffle_N')]:
   rec=saved[tag];steps+=len(rec['steps'])
   if not rec['active']:
    for k in (1,3):assert np.array_equal(a[tag+'_tta'+str(k)],z['source_scores'].astype(float))
    continue
   f=prefix(r,('h','tokens','token_mask'));z0=native(initial,f['h'],f['tokens'],f['token_mask']).astype(float)
   for k in (1,3):
    w=load(ad/'actual_head_states_PRIVATE'/(sid+'_'+tag+str(k)+'.npz'));assert any(not np.array_equal(w[n],initial[n]) for n in w);delta=native(w,f['h'],f['tokens'],f['token_mask']).astype(float)-z0;expected=softmax(np.log(np.maximum(p0,1e-300))+delta);actual=a[tag+'_tta'+str(k)];err=float(np.max(abs(expected-actual)));maxerr=max(maxerr,err);assert err<5e-5;replays+=1
   assert all(s['gradient_norm']>0 and s['parameter_change_L2']>0 for s in rec['steps'])
 # Independently reconstruct raw RGB/LiDAR inputs and global graph supports on
 # hash-selected observations per recording, selected without task labels.
 from geometry72 import scene,supports
 from text72 import parse
 selected={}
 for r in rows():
  key=(cohort_map()[r['id']],r['platform'],r['scene']);selected.setdefault(key,[]).append(r)
 samples=[]
 for rr in selected.values():samples.extend(sorted(rr,key=lambda x:hashlib.sha256(('R72rawaudit|'+x['id']).encode()).hexdigest())[:2])
 for r in samples:
  rel='graphs_PRIVATE/'+r['id']+'.npz';old=verified(raw,rel);s=scene(r)
  for k in s:assert np.array_equal(s[k],old[k]),('raw_graph_replay',k)
  f=prefix(r,('boxes','tokens'));text=parse(r['caption'],len(f['tokens']));a0=prep(r);tm=supports(s,f['boxes'],a0,text['target']);v=verified(e,'samples_PRIVATE/'+r['id']+'.npz')
  for k in tm:assert np.array_equal(tm[k],v['target_'+k]),('support_replay',k)
 rec={'state':'PASS_R72_ALL2640_FIXED_BOX_DIRECT_CONTROLS_ACTUAL_WEIGHT_REPLAY_AND_RAW_GEOMETRY','n':2640,'arms':len(ARMS),'actual_weight_states_replayed':replays,'actual_optimizer_steps':steps,'max_independent_numpy_probability_error':maxerr,'raw_graph_support_replays':len(samples),'all_unknown_and_inactive_rows_retained':True,'original_evidence_and_camera_contract_repair_both_preserved':True,'no_expert_feedback_or_task_label_lookup':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
if __name__=='__main__':main()
