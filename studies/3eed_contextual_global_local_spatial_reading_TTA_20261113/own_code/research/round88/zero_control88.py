"""Prespecified stronger no-update control: same full plus two masked reads."""
from core88 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source
start=time.perf_counter();base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));m=Fusion88(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False);S=load(ROOT/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz');restore(m,S);O=B88/'frozen_routed3_control336_attempt001';(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False)
with torch.no_grad():
 for nr,r in enumerate(read(H88/'experiment/DEV336_PRIVATE.json')):
  z0,z,d,a,pm,im,ws=data88(r);m.route=tensor_route88(a);m.route_mode='full';p=m.main(base,z,d,'bidirectional')[0];views=[m.main(base,z,d,'bidirectional',point_mask=x,image_mask=y)[0] for x,y in zip(pm,im)];mean=torch.stack([p]+views).mean(0);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),probabilities=torch.stack([p]+views).cpu().numpy(),mean=mean.cpu().numpy());
  if (nr+1)%64==0:print('ZERO88',nr+1,flush=True)
dump(O/'COUNTS.json',{'n':336,'native_suffix_calls':1008,'seconds':time.perf_counter()-start,'no_update':True,'GT_read':False});seal(O,'R88_SAME_INFORMATION_THREE_VIEW_ZERO_UPDATE_CONTROL_SEALED',n=336);print(read(O/'COUNTS.json'))
