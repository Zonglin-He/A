"""Frozen CLIP only: all projected groups, two crops, literal text embeddings."""
from common70 import *
from geometry70 import crops
import sys,time,torch,argparse
from PIL import Image
sys.path.insert(0,str(ROOT/'crossplatform/round8'));from semantic_encoder import FrozenSemanticEncoder

def text_features(enc,text):
 ids=enc.tokenizer.encode(text,add_special_tokens=False);parts=[ids[i:i+75] for i in range(0,len(ids),75)];vv=[]
 for part in parts:
  inp=torch.tensor([enc.tokenizer.build_inputs_with_special_tokens(part)],device=enc.device)
  with torch.no_grad():v=enc.model.get_text_features(input_ids=inp,attention_mask=torch.ones_like(inp));vv.append(torch.nn.functional.normalize(v,dim=-1)[0].cpu().numpy())
 result=np.average(vv,axis=0,weights=[len(x) for x in parts]);result/=np.linalg.norm(result);return result.astype(np.float32),{'tokens':len(ids),'chunks':len(parts),'all_tokens_covered':True,'interchunk_composition_not_preserved':len(parts)>1}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=read(H/'experiment/CLIP_PLAN_LOCK.json');pin(p);prep=ROOT/p['prepared_root'];ss=read(prep/'SEALED.json');rr=rows();rr=[next(r for r in rr if r['platform']==pl) for pl in ['drone','quad']] if args.engineering else rr;out=H/'engineering_clip2_attempt001' if args.engineering else ROOT/p['clip_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();enc=FrozenSemanticEncoder(device='cuda');versions=[x._version for x in enc.model.parameters()];start=time.perf_counter();calls=0
 for i,r in enumerate(rr):
  rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(prep/rel)==ss['files'][rel];z=load(prep/rel);rec=read(prep/'cases_PRIVATE'/(r['id']+'.json'));assert sha(r['imagepath'])==rec['image_sha256'];im=Image.open(r['imagepath']).convert('RGB');idx=np.flatnonzero(z['rect_valid']);ims=[crops(im,z['rects'][k],scale) for k in idx for scale in [1.,1.25]];vf=enc.encode_crops(ims).numpy().reshape(len(idx),2,512);full=np.zeros((len(z['rects']),2,512),np.float32);full[idx]=vf;st,sr=text_features(enc,rec['subject_PRIVATE']);ft,fr=text_features(enc,r['caption']);calls+=len(ims);np.savez_compressed(out/rel,visual=full,subject_text=st,full_text=ft,logit_scale=np.array(float(enc.model.logit_scale.exp().item())));dump(out/'cases_PRIVATE'/(r['id']+'.json'),{'subject_receipt':sr,'full_receipt':fr,'encoded_groups':len(idx),'image_crops':len(ims),'image_sha256':rec['image_sha256']})
  if(i+1)%32==0:print('CLIP70',i+1,336,calls,flush=True)
 assert versions==[x._version for x in enc.model.parameters()];enc._assert_frozen();dump(out/'COUNTS.json',{'n':len(rr),'image_crops':calls,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'model_unchanged':True,'resource':enc.resource});seal(out,'R70_FROZEN_CLIP_REAL_PROJECTED_CROPS_NO_OTHER_VISION_MODEL',n=336);print({'n':len(rr),'image_crops':calls})
