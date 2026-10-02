"""Frozen external CLIP, explicit phrases, same fixed nine candidates; no GT."""
import os,sys,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_transfer_aligned_common_v1 import *
def run():
 import torch,numpy as np
 from PIL import Image
 from transformers import CLIPModel,CLIPTokenizerFast
 from scripts.run_final_simplification_v1 import lease
 from scripts.run_tastvg_full_b1_experts_v1 import observation
 from vg_tta import exact_frame_decode_audit_v2 as binding
 from vg_tta.tastvg_paper48_hc2_decode_v1 import decode as hc_decode
 from vg_tta.exact_frame_decode_audit_v2 import decode as vid_decode
 from vg_tta.tastvg_aligned_token_binding_v1 import score
 from methods.decota_final_simplified_v1.tensors import state_hash
 lock=verify('TOKEN');assert (BASE/'transfer/PREDICTION_BARRIER.json').exists();sys.addaudithook(guard);fh=lease();tick=time.monotonic();done=frames_seen=0
 torch.set_num_threads(4);torch.manual_seed(20261002);np.random.seed(20261002);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
 asset=ROOT/'checkpoints/clip_vit_b16_token_v1';model,info=CLIPModel.from_pretrained(str(asset),local_files_only=True,output_loading_info=True,attn_implementation='eager')
 assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model=model.cuda().float().eval().requires_grad_(False);tokenizer=CLIPTokenizerFast.from_pretrained(str(asset),local_files_only=True);mh=state_hash(model.state_dict());text_cache={};checks=[]
 def text_tokens(phrases):
  seq=[];parity=[]
  for phrase in phrases:
   if phrase not in text_cache:
    tok=tokenizer([phrase],return_tensors='pt',padding=False,truncation=False).to('cuda');assert tok['input_ids'].shape[1]<=77
    with torch.inference_mode():
     z=model.text_model(**tok);v=model.text_projection(z.last_hidden_state);keep=tok['attention_mask'].bool()&(tok['input_ids']!=tokenizer.bos_token_id)&(tok['input_ids']!=tokenizer.eos_token_id);q=v[keep].cpu();native=model.get_text_features(**tok);eot=model.text_projection(z.pooler_output);assert torch.equal(native,eot)
    text_cache[phrase]=(q.numpy(),dict(EOT_projection_bitwise=True,lexical_tokens=len(q)))
   q,receipt0=text_cache[phrase];seq.append(q);parity.append(receipt0)
  return np.concatenate(seq) if seq else np.empty((0,512),dtype=np.float32),parity
 mean=np.array([.48145466,.4578275,.40821073],dtype=np.float32);std=np.array([.26862954,.26130258,.27577711],dtype=np.float32)
 for ds in ['hc2','vidstg']:
  binding.decode=hc_decode if ds=='hc2' else vid_decode;plan=read(QUAL/ds/'PLAN.json');p=read(OLD/ds/'PLAN.json');parses={r['parent']:r['phrases'] for r in read(BASE/'token'/ds/'PHRASES_PRIVATE.json')}
  for c in plan['cells']:
   f=BASE/'token'/ds/'scores'/f'{c["cell"]:03}.pt'
   if f.with_suffix('.json').exists():receipt(f);done+=1;continue
   budget();row=p['rows'][c['parent']];images,ids=binding.decode(row['input']);assert ids==row['frame_ids'];pixels,pixel,_=observation(row,c['condition'],images);assert pixel==c['pixel_sha256'];old=load(ROOT/c['payload']);assert old['pre_sha']==c['pre_state_sha256']
   parse=parses[c['parent']];qo,op=text_tokens(parse['object']);qe,ep=text_tokens(parse['event']);patches=[];vision_parity=False
   with torch.inference_mode():
    for start in range(0,len(ids),8):
     arr=np.stack([(np.asarray(Image.fromarray(a).resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)/np.float32(255.)-mean)/std for a in pixels[start:start+8]]);inp=torch.from_numpy(arr).permute(0,3,1,2).contiguous().cuda();out=model.vision_model(pixel_values=inp);v=model.visual_projection(model.vision_model.post_layernorm(out.last_hidden_state[:,1:]));assert v.shape[1:]==(196,512);patches.append(v.cpu().numpy())
     if start==0:
      original=model.get_image_features(pixel_values=inp);assert torch.equal(original,model.visual_projection(out.pooler_output));vision_parity=True
   patches=np.concatenate(patches);tubes=np.stack([z['prediction']['boxes'].numpy() for z in old['update_steps'][0]['candidates']]);interval=old['slow']['indices'];sc=score(patches,qo,qe,tubes,interval)
   commit(f,dict(cell=c['cell'],source_id=c['source_id'],parent=c['parent'],condition=c['condition'],order=c['order'],arrival=c['arrival'],pixel_sha256=pixel,pre_state_sha256=c['pre_state_sha256'],patches=patches,q_obj=qo,q_evt=qe,scores=sc,interval=interval,phrase_counts=dict(object=len(parse['object']),event=len(parse['event'])),referent_rule=parse['referent_rule'],text_projection_parity=op+ep,vision_projection_bitwise=vision_parity,GT_read=False,parameter_updates=0,new_experts=0))
   done+=1;frames_seen+=len(ids);checks.append(dict(dataset=ds,cell=c['cell'],vision_projection_bitwise=vision_parity,text_phrases=len(op)+len(ep)));status(BASE/'token/STATUS.json',dict(status='running',dataset=ds,done=done,total=60,frames=frames_seen,worker_pid=os.getpid(),time=time.time()));print('ALIGNED TOKEN',ds,c['cell'],done,60,'frames',len(ids),flush=True)
   del patches,pixels,images,old,tubes,sc;gc.collect();torch.cuda.empty_cache()
 assert done==60 and state_hash(model.state_dict())==mh;verify('TOKEN')
 write(BASE/'token/PREDICTION_BARRIER.json',dict(cells=60,candidates=540,GT_read=False,parameter_updates=0,new_experts=0,files={str(f.relative_to(BASE)):sha(f) for f in (BASE/'token').glob('*/scores/*.json')},time=time.time()))
 write(BASE/'token/RESOURCES.json',dict(worker_wall_seconds=time.monotonic()-tick,wall_includes_loading_decode_IO_CPU_scoring=True,frames=frames_seen,projection_parity_checks=checks,new_experts=0,parameter_updates=0,peak_vram_bytes=torch.cuda.max_memory_allocated()))
 status(BASE/'token/STATUS.json',dict(status='sealed_pending_root_score',done=60,total=60,time=time.time()));fh.close()
if __name__=='__main__':
 try:run()
 except BaseException as e:
  write(BASE/'token'/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));status(BASE/'token/STATUS.json',dict(status='failed',error=repr(e),time=time.time()));raise
