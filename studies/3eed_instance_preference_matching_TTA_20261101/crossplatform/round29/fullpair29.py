"""One fixed full-scene/full-description pair feedback. No crop/gray rules."""
import sys,time,numpy as np
from common29 import *
sys.path.insert(0,str(ROOT/'crossplatform/round22'))
from scene22 import Scene,render
from observations27 import load
from association import DEFAULT_CONFIG

def main():
 import torch
 torch.set_num_threads(4);out=BASE/'fullpair208_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();plan=read(ROOT/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json');mdir=Path(plan['model_dir']);assert all(sha(mdir/f)==h for f,h in plan['model_files_sha256'].items());engine=Scene(mdir,DEFAULT_CONFIG);engine.setup_choices();calls=0;start=time.perf_counter()
 for i,r in enumerate(rows()[:208]):
  pairs=old(r,'pairs')['pairs'];payload={'alias':r['alias'],'pairs':[]}
  if pairs:
   frame,scene,binding=load(r)
   for p in pairs:
    a,b=p['mask_indices_PRIVATE'];A,B=scene['objects'][a],scene['objects'][b];v={order:engine.compare(render(frame['rgb'],A,B,reverse=reverse),r['caption'],'') for order,reverse in [('normal',False),('reversed',True)]};payload['pairs'].append({'pair_alias':p['pair_alias'],'full':v});calls+=2
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload)
  if (i+1)%16==0:print('FULLPAIR29',i+1,'calls',calls,'seconds',round(time.perf_counter()-start,1),flush=True)
 assert calls==308 and engine.verify_unchanged();seal(out,'R29_fullscene_fullcaption_pair_preferences_before_task_audit',n=208,pairs=154,calls=calls,generated_tokens=0,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('FULLPAIR29_SEALED',flush=True)
if __name__=='__main__':main()
