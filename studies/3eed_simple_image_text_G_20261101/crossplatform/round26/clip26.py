"""Literal full-text CLIP cosine; shared pool and identical two crop functions."""
import sys,time,hashlib,numpy as np
from common26 import *
from observations26 import load
from visualZ import crops
sys.path.insert(0,str(ROOT/'crossplatform/round8'))
from semantic_encoder import FrozenSemanticEncoder
def chunk_features(enc,text):
 import torch
 ids=enc.tokenizer.encode(text,add_special_tokens=False);chunks=[ids[i:i+75] for i in range(0,len(ids),75)];feats=[]
 for chunk in chunks:
  seq=enc.tokenizer.build_inputs_with_special_tokens(chunk);assert len(seq)<=77;inp=torch.tensor([seq],device=enc.device)
  with torch.inference_mode():f=enc.model.get_text_features(input_ids=inp,attention_mask=torch.ones_like(inp));f=f/f.norm(dim=-1,keepdim=True)
  feats.append(f[0].float().cpu().numpy())
 f=np.average(feats,axis=0,weights=[len(c) for c in chunks]);f/=np.linalg.norm(f)
 return f,{'original_nonspecial_tokens':len(ids),'chunk_count':len(chunks),'covered_tokens':sum(map(len,chunks)),'composition_across_chunks_not_preserved':len(chunks)>1}
def main():
 from PIL import Image
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert sha(Path(__file__))==read(EXP/'JOINT_CODE_LOCK.json')['clip_program_sha256'];out=BASE/'clip464_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();enc=FrozenSemanticEncoder(device='cuda');versions=[(n,p._version) for n,p in enc.model.named_parameters()];start=time.perf_counter();count=0
 for i,r in enumerate(rows()):
  pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));g=read(pp['text']/'cases_PRIVATE'/(r['id']+'.json'));res={'alias':r['alias'],'targets_PRIVATE':{}}
  if prep['mask_pool_PRIVATE']:
   frame,scene,binding=load(r);assert binding==prep['bindings'];tf,receipt=enc.encode_texts([r['caption'],g['subject_query']]);cf,cr=chunk_features(enc,r['caption']);res.update(text_receipt=receipt,chunk_receipt=cr,literal_caption_sha256=hashlib.sha256(r['caption'].encode()).hexdigest())
   for m in prep['mask_pool_PRIVATE']:
    images=crops(frame['rgb'],scene['objects'][m]);allimages=images+[Image.new('RGB',x.size,(127,127,127)) for x in images];vf=enc.encode_crops(allimages).numpy();count+=len(allimages);normal=vf@tf.numpy().T;chunk=vf@cf
    res['targets_PRIVATE'][m]={'full_native':[float(x) for x in normal[:2,0]],'subject':[float(x) for x in normal[:2,1]],'full_chunk':[float(x) for x in chunk[:2]],'gray_native':[float(x) for x in normal[2:,0]],'gray_subject':[float(x) for x in normal[2:,1]],'crop_sha256':[arrsha(np.asarray(x)) for x in images]}
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),res)
  if (i+1)%32==0:print('CLIP26',i+1,count,round(time.perf_counter()-start,1),flush=True)
 assert versions==[(n,p._version) for n,p in enc.model.named_parameters()];seal(out,'R26_simple_CLIP_cosine_no_G_logic_no_GT',n=len(rows()),image_forwards=count,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated())
if __name__=='__main__':main()
