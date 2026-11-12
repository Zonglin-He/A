"""Saved-step intervention magnitude, no new model, task labels, or optimizer."""
from core88 import *
P=B88/'prediction336_attempt001';S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');rows=[]
for i,r in enumerate(read(H88/'experiment/DEV336_PRIVATE.json')):
 sid=r['id'];rec=read(P/'records_PRIVATE'/(sid+'.json'));a=load(P/'states_PRIVATE'/(sid+'_full_tta1.npz'));na=rec['full'][0];size=np.sqrt(sum(np.sum((a[k].astype(float)-S[k])**2) for k in a))
 for variant in ['no_correspondence','no_rank','physical_only']:
  b=load(P/'states_PRIVATE'/(sid+'_'+variant+'1.npz'));nb=rec[variant][0];diff=np.sqrt(sum(np.sum((a[k].astype(float)-b[k])**2) for k in a));rows.append({'row':'DEV'+str(i+1).zfill(4),'platform':r['platform'],'removed':variant,'same_accepted_scale':na['scale']==nb['scale'],'full_step_L2':float(size),'full_vs_ablated_parameter_L2':float(diff),'relative_parameter_difference':float(diff/max(size,1e-20)),'full_raw_gradient_norm':na['gradient_norm_raw'],'ablated_raw_gradient_norm':nb['gradient_norm_raw'],'no_GT_or_new_model_call':True})
dump(H88/'SAVED_STEP_EFFECT_ROWS.json',rows)
for v in ['no_correspondence','no_rank','physical_only']:
 rr=[r for r in rows if r['removed']==v and r['same_accepted_scale']];print(v,len(rr),float(np.median([r['relative_parameter_difference'] for r in rr])),max(r['relative_parameter_difference'] for r in rr))
