"""After prediction seal: initial source-prepared physical error on unused third mask."""
from core91 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source',action='store_true');args=ap.parse_args();rows=source_rows('VAL128') if args.source else read(H91/'experiment/DEV336_PRIVATE.json');label='source128' if args.source else 'prediction336';P=B91/(label+'_attempt001');assert (P/'SEALED.json').exists();proj=nn.Sequential(nn.Linear(288,288),nn.ReLU(),nn.Linear(288,288),nn.ReLU(),nn.Linear(288,64));m=Fusion91(proj).cuda().eval().requires_grad_(False);S=loadbase91(m,'R76');held=read(P/'HELDOUT_ROWS.json');out=[]
with torch.no_grad():
 for i,r in enumerate(rows):
  z0,z,d,rt,route,pm,im,ws=data91(r,args.source);hp,hi=masks(r,d,76003);ww=weight_sets({'id':'UNUSED_UNIFORM'},d,[hp],[hi],np.zeros(1024),np.zeros(1536))[0]['uniform'];before=float(uniform_aux91(m,z,d,[hp],[hi],ww));row=('SOURCEVAL' if args.source else 'DEV')+str(i+1).zfill(4)
  for x in held:
   if x['row']==row:out.append({**x,'heldout_true_physical_before':before,'heldout_error_change':x['heldout_true_physical_error']-before,'not_used_for_fitting_or_acceptance':True})
dump(H91/('SOURCE_HELDOUT_BEFORE_AFTER_ROWS.json' if args.source else 'HELDOUT_BEFORE_AFTER_ROWS.json'),out);print({'n':len(rows),'rows':len(out),'new_task_GT_scoring':0,'parameters_changed':False})
