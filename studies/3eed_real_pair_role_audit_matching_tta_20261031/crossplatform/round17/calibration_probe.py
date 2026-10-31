"""Finite4-observation pre-label contextual-calibration engineering probe.
Not a primary G variant and never used to choose pseudo targets or B results.
"""
import sys,time
import numpy as np
from common import ROOT,EXP,read,dump,sha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round13')]
from pair_verifier import images_for,messages
from input_io import load_case
from scene_observation import build_scene_observation
from association_runner import FrozenVLM
from association import DEFAULT_CONFIG
class Probe(FrozenVLM):
 def forward_options(self,msg,images,options):
  t=self.torch;text=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);ins=self.processor(text=[text],images=images,return_tensors='pt',padding=False,truncation=False).to('cuda');ids=[self.processor.tokenizer.encode(x,add_special_tokens=False) for x in options];assert all(len(x)==1 for x in ids)
  with t.inference_mode():o=self.model(**ins,use_cache=False);z=o.logits[0,-1,t.tensor([x[0] for x in ids],device='cuda')].float();logp=z.log_softmax(0).cpu().numpy()
  return logp
def main():
 check_plan();lock=read(EXP/'CALIBRATION_PROBE_PRELABEL_LOCK.json');assert lock['program_sha256']==sha(__file__)
 inp=EXP.parent/'input';allrows=read(inp/'A64_PRIVATE.json');rows=[r for p in ('quad','drone') for r in [x for x in allrows if x['platform']==p][:2]];cases={c['id']:c for c in read(inp/'A_CASES_PRIVATE.json')};idx=read(inp/'A_FRAME_INDEX_PRIVATE.json');prep=ROOT/'crossplatform/round17/pairs64_attempt001';verify(prep);out=ROOT/'crossplatform/round17/calibration_probe4_attempt001';out.mkdir(exist_ok=False)
 old=read(ROOT/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json');engine=Probe(__import__('pathlib').Path(old['model_dir']),DEFAULT_CONFIG);results=[];start=time.perf_counter();calls=0
 for r in rows:
  c=cases[r['id']];frame,_,binding=load_case(r,c,idx);h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=binding['image_sha256']);pairs=read(prep/'cases_PRIVATE'/(r['id']+'.json'))['pairs']
  tasks=[]
  if pairs:
   ids=pairs[0]['mask_indices'];tasks.extend([('pair',ids,'ABNU'),('pair',list(reversed(ids)),'ABNU')]);tasks.extend([('role',[m],'TROU') for m in ids])
  for mode,ids,options in tasks:
   images=images_for(frame['rgb'],scene['objects'],ids);full=engine.forward_options(messages(r['caption'],scene['objects'],ids,mode),images,options);prior=engine.forward_options(messages('N/A',scene['objects'],ids,mode),images,options);calls+=2;cal=full-prior;cal=np.exp(cal-cal.max());cal/=cal.sum()
   results.append({'case_alias':c['case_alias'],'platform':r['platform'],'mode':mode,'mask_indices_PRIVATE':ids,'options':list(options),'logp_full_PRIVATE':full,'logp_contentfree_PRIVATE':prior,'calibrated_p_PRIVATE':cal,'raw_choice':options[int(full.argmax())],'contentfree_choice':options[int(prior.argmax())],'calibrated_choice':options[int(cal.argmax())]})
 assert engine.verify_unchanged();dump(out/'RESULTS_PRIVATE.json',results);dump(out/'registered_lock.json',lock);seal(out,'R17_contentfree_calibration4_engineering_only',n=4,calls=calls,seconds=time.perf_counter()-start,model_unchanged=True,not_primary_G_or_task_accuracy=True);print('CALIBRATION_PROBE_SEALED',calls,flush=True)
if __name__=='__main__':main()
