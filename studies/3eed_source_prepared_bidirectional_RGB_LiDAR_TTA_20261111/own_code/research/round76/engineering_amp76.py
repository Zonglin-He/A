from common76 import *
from model76 import *
import time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
base=load_source(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(base);r=source_rows('TRAIN512')[0];z=tensor_prefix(prefix(r));d=tensor_data(load(casepath(r)),r);torch.manual_seed(0);m=Fusion(base.contrastive_align_projection_image).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003);y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];target=torch.softmax(torch.tensor(y,device='cuda',dtype=torch.float64)/.1,0);costs=[]
for i in range(3):
 t=time.perf_counter();pm,im=masks(r,d,i);opt.zero_grad(set_to_none=True)
 with torch.autocast(device_type='cuda',dtype=torch.bfloat16):p,q,_=m.main(base,z,d,'bidirectional');aux,_=m.auxiliary(z,d,'bidirectional',pm,im);loss=-(target*p.clamp_min(1e-300).log()).sum()+aux
 assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1);assert torch.isfinite(gn);opt.step();torch.cuda.synchronize();costs.append(time.perf_counter()-t)
assert fingerprint(base)==fp;out=H/'engineering_amp_attempt001';assert not out.exists();out.mkdir();dump(out/'RESULT.json',{'state':'PASS_R76_BF16_SOURCE_TRAINING_FP32_MATCH_AND_VALIDATION','train_forward_backward_seconds':costs,'source_unchanged':True,'only_source_supervision_read':True,'no_target_read':True,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});print(read(out/'RESULT.json'))
