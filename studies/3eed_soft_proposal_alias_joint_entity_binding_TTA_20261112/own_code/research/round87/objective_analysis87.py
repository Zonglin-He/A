"""Separate full-readout KL from masked KL using sealed probabilities only."""
from core87 import *
from scipy.special import softmax
P=B87/'prediction336_attempt001';tasks=read(H87/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');out=[]
for t in tasks:
 sid=t['id_PRIVATE'];a=load(P/'samples_PRIVATE'/(sid+'.npz'));rec=read(P/'records_PRIVATE'/(sid+'.json'))['variants']['full'][0];pi=a['full_pi'];p0=softmax(a['s0']);after=a['scores'][ARMS87.index('full_tta1')]
 def kl(p):return float(np.sum(pi*(np.log(np.maximum(pi,1e-300))-np.log(np.maximum(p,1e-300)))))
 f0,f1=kl(p0),kl(after);active=rec['classification_before'] is not None;o={'row':t['row'],'platform':t['platform'],'active':active,'full_KL_before':f0 if active else None,'full_KL_after':f1 if active else None,'masked_mean_KL_before':rec['classification_before']-f0 if active else None,'masked_mean_KL_after':rec['classification_after']-f1 if active else None,'full_KL_increased':active and f1>f0+1e-10,'target_top1_matches_final':int(pi.argmax())==int(after.argmax()),'target_winner_changed_by_update':int(pi.argmax())!=int(after.argmax()),'task_delta_vs_frozen_pp':100*(t['IoU']['full_tta1']-t['IoU']['frozen_bi']),'task_delta_vs_direct_pp':100*(t['IoU']['full_tta1']-t['IoU']['full_direct'])};out.append(o)
dump(H87/'READOUT_OBJECTIVE_ROWS.json',out);print({'active':sum(r['active'] for r in out),'full_KL_increased':sum(r['full_KL_increased'] for r in out),'target_top1_not_matched':sum(r['target_winner_changed_by_update'] for r in out)})
