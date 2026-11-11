"""R70 exact frozen CLIP encoding; no source or adapter forwards."""
from common71 import *
import sys,time,torch,argparse
from PIL import Image
sys.path.insert(0,str(ROOT/'research/round70'));from geometry70 import crops
from clip70 import text_features
sys.path.insert(0,str(ROOT/'crossplatform/round8'));from semantic_encoder import FrozenSemanticEncoder
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=read(H/('experiment/CLIP_PLAN_LOCK_ENGINEERING.json' if args.engineering else 'experiment/CLIP_PLAN_LOCK.json'));pin(p);prep=H/'engineering_prepare4_attempt001' if args.engineering else ROOT/p['prepared_root'];ss=read(prep/'SEALED.json') if (prep/'SEALED.json').exists() else None;rr=rows();used={}
 if args.engineering:rr=[r for r in rr if 'samples_PRIVATE/'+r['id']+'.npz' in ss['files']]
 out=H/'engineering_clip4_attempt001' if args.engineering else ROOT/p['clip_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();enc=FrozenSemanticEncoder(device='cuda',batch_size=32);versions=[x._version for x in enc.model.parameters()];start=time.perf_counter();calls=0
 for i,r in enumerate(rr):
  rel='samples_PRIVATE/'+r['id']+'.npz';case=prep/'cases_PRIVATE'/(r['id']+'.json')
  while not case.exists():
   if not (H/'RUNNING_CPU.json').exists() and not (prep/'SEALED.json').exists():raise RuntimeError('Preparation stopped before completing required input')
   time.sleep(.2)
  for attempt in range(20):
   try:rec=read(case);break
   except json.JSONDecodeError:time.sleep(.05)
  used[rel]=sha(prep/rel);used['cases_PRIVATE/'+r['id']+'.json']=sha(case)
  if ss is not None:assert used[rel]==ss['files'][rel]
  a=load(prep/rel);assert sha(r['imagepath'])==rec['image_sha256'];im=Image.open(r['imagepath']).convert('RGB');idx=np.flatnonzero(a['rect_valid']);ims=[crops(im,a['rects'][k],scale) for k in idx for scale in [1.,1.25]];vf=enc.encode_crops(ims).numpy().reshape(len(idx),2,512);v=np.zeros((len(a['rects']),2,512),np.float32);v[idx]=vf;st,sr=text_features(enc,rec['subject_PRIVATE']);calls+=len(ims);np.savez_compressed(out/rel,visual=v,subject_text=st,logit_scale=np.array(float(enc.model.logit_scale.exp().item())));dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'subject_receipt':sr,'encoded_groups':len(idx),'image_crops':len(ims),'image_sha256':rec['image_sha256']})
  if(i+1)%32==0:print('CLIP71',i+1,len(rr),calls,round(time.perf_counter()-start,1),flush=True)
 ss=read(prep/'SEALED.json')
 for name,digest in used.items():assert ss['files'][name]==digest,('streamed_input_final_seal',name)
 assert versions==[x._version for x in enc.model.parameters()];enc._assert_frozen();dump(out/'COUNTS.json',{'n':len(rr),'image_crops':calls,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'model_unchanged':True,'source_forwards':0,'input_preparation_seal_sha256':sha(prep/'SEALED.json'),'all_streamed_input_bytes_checked_against_final_seal':True,'adapter_training_steps':0,'resource':enc.resource});seal(out,'R71_FIXED_FROZEN_CLIP_REAL_REGIONS_ALL_PREDICTIONS_UNSCORED',n=len(rr));print(read(out/'COUNTS.json'))
