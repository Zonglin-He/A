"""Finite score-only ablations; no model updates and no task-label access."""
import numpy as np
from common26 import *
from simple_g26 import simple_fusion
def main():
 out=BASE/'scores464_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();counts={'n':0,'has_pool':0,'mapped_queries':0,'fulltext_clip_truncated':0,'crop_replay_checks':0};variants=None
 for r in rows():
  pp=paths(r);z=feature(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));a=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));bank=load_npz(pp['bank']/'samples_PRIVATE'/(r['id']+'.npz'));ev=read(pp['evidence']/'cases_PRIVATE'/(r['id']+'.json'));joint=read(BASE/'joint464_attempt001/cases_PRIVATE'/(r['id']+'.json'));clip=read(BASE/'clip464_attempt001/cases_PRIVATE'/(r['id']+'.json'));pool=prep['mask_pool_PRIVATE'];w=a['weights'];q0=int(z['scores'].argmax());mapped=w.sum(1)>0
  def relative(v,clip=True):
   out=np.zeros(256);anchor=v[q0] if mapped[q0] else float(np.median(v[mapped])) if mapped.any() else 0;out[mapped]=(v[mapped]-anchor)/2
   return np.clip(out,-2,2) if clip else out
  def mappedmean(array,mode='dual'):
   arr=np.asarray(array,float)
   if not len(pool):return np.zeros(256)
   return w@(arr.mean(1) if mode=='dual' else arr[:,0 if mode=='tight' else 1])
  # Exact original uses median INSTANCE evidence when q0 is unmapped, not query median.
  def original_relative(array,mode='dual',bounded=True):
   out=np.zeros(256)
   if not len(pool):return out
   arr=np.asarray(array,float);v=arr.mean(1) if mode=='dual' else arr[:,0 if mode=='tight' else 1];vv=w@v;anchor=vv[q0] if mapped[q0] else float(np.median(v));out[mapped]=(vv[mapped]-anchor)/2
   return np.clip(out,-2,2) if bounded else out
  full=np.array([[x['full']['log_odds_yes_no'] for x in ev['targets_PRIVATE'][str(m)]] for m in pool]);gray=np.array([[x['gray']['log_odds_yes_no'] for x in ev['targets_PRIVATE'][str(m)]] for m in pool]);delta=full-gray
  origvision=original_relative(delta);assert np.max(np.abs(origvision-bank['vision']))<1e-10
  cues={
   'legacy_complex':bank['vision']+bank['position']+bank['relation'],
   'no_position_reference':bank['vision'],
   'bounded_no_gray_dual':original_relative(full),
   'bounded_gray_tight':original_relative(delta,'tight'),
   'bounded_gray_context':original_relative(delta,'context'),
   'centered_unclipped_gray_dual':original_relative(delta,bounded=False),
   'qwen_subject_gray_dual':mappedmean(delta),
   'qwen_subject_raw_dual':mappedmean(full),
   'qwen_subject_raw_tight':mappedmean(full,'tight'),
   'qwen_subject_raw_context':mappedmean(full,'context'),
   'qwen_full_scene_raw':w@np.array([joint['targets_PRIVATE'][str(m)]['log_odds_yes_no'] for m in pool]) if pool else np.zeros(256),
  }
  for key in ['subject','full_native','full_chunk']:
   x=np.array([clip['targets_PRIVATE'][str(m)][key] for m in pool]);cues['clip_'+key+'_dual']=mappedmean(x)
  # CLIP gray subtraction is an explicit extra ablation, never hidden in simple cosine.
  x=np.array([np.array(clip['targets_PRIVATE'][str(m)]['full_native'])-np.array(clip['targets_PRIVATE'][str(m)]['gray_native']) for m in pool]);cues['clip_full_native_gray_dual']=mappedmean(x)
  for m in pool:
   for x,h in zip(ev['targets_PRIVATE'][str(m)],clip['targets_PRIVATE'][str(m)]['crop_sha256']):assert x['full']['image_sha256']==h;counts['crop_replay_checks']+=1
  if pool:
   raw_instance=full.mean(1);ss,qq=simple_fusion(z['scores'],w,raw_instance,1.);assert np.array_equal(ss,z['scores']+cues['qwen_subject_raw_dual'])
  names=list(cues)
  if variants is None:variants=names
  assert names==variants and all(np.isfinite(v).all() for v in cues.values())
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),source=z['scores'],boxes=z['boxes'],cue_names=np.array(names),cues=np.stack([cues[n] for n in names]),mapped=mapped)
  counts['n']+=1;counts['has_pool']+=bool(pool);counts['mapped_queries']+=int(mapped.sum());counts['fulltext_clip_truncated']+=clip.get('text_receipt',{}).get('original_token_lengths',[0])[0]>77
 dump(out/'COUNTS.json',counts);dump(out/'VARIANTS.json',variants);seal(out,'R26_finite_numeric_cues_GT_free',**counts);print('READOUT26_SEALED',counts,variants)
if __name__=='__main__':main()
