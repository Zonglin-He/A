"""Source-only bounded 8B PTD training-interface pilot; never a teacher qualification claim."""
import ast,copy,gc,hashlib,importlib.util,json,math,os,shutil,sys,time,traceback,zipfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.decota_matrix_common_v1 import read,write,status,sha
OUT=ROOT/'artifacts/ptd_8b_teacher_feasibility_v1';PTD=ROOT/'external/ParallelTubeDecoding';BASE=ROOT/'checkpoints/Qwen3-VL-8B-Instruct'
sys.path.insert(0,str(PTD/'src'))

def load_file(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def register():
 pins=[Path(__file__),ROOT/'protocols/ptd_8b_teacher_feasibility_v1.md',PTD/'src/model/ptd_tokens.py',PTD/'src/model/ptd_mask_utils.py',PTD/'src/dataset/sft_dataset.py',PTD/'src/train/monkey_patch_forward.py',PTD/'data/prepare_vidstg.py']
 parents=[ROOT/'artifacts/ptd_learned_credit_v1/SOURCE_INPUTS.json',ROOT/'artifacts/ptd_learned_credit_v1/INTAKE.json',BASE/'DOWNLOAD_MANIFEST.json']
 write(OUT/'REGISTRATION.json',dict(time=time.time(),pins={str(p):sha(p) for p in pins},parents={str(p):sha(p) for p in parents},protected={str(ROOT/'methods/CURRENT_METHOD.json'):sha(ROOT/'methods/CURRENT_METHOD.json')},gpu_seconds=1800,output_bytes=4*2**30,pilot_optimizer_steps=4,source_GT=True,target_GT=False,teacher_qualified=False,student_updates=0))

def verify():
 r=read(OUT/'REGISTRATION.json');pins=dict(r['pins'])
 for p in sorted((OUT/'amendments').glob('*.json')):pins.update(read(p).get('pins',{}))
 for p,h in {**pins,**r['parents'],**r['protected']}.items():assert sha(p)==h,p
 return r

def intake():
 verify();rr=read(ROOT/'artifacts/ptd_learned_credit_v1/SOURCE_INPUTS.json');old=read(ROOT/'artifacts/ptd_learned_credit_v1/INTAKE.json')
 excluded=set(old['excluded_parents']);assert not {r['source'] for r in rr}&excluded
 prep=load_file('prepare_vid_p0',PTD/'data/prepare_vidstg.py');ann=read(ROOT/'external/VidSTG-Dataset/annotations/train_annotations.json');out=[];rejected=[]
 with zipfile.ZipFile(ROOT/'downloads/vidor/training-annotation.zip') as z:
  members={Path(n).stem:n for n in z.namelist() if n.endswith('.json')}
  for r in rr:
   q=r['input'];a=ann[r['official_annotation_index']];assert str(a['vid'])==r['source'];assert q['caption']==a['captions'][0]['description'];assert sha(q['video_path'])==q['video_sha256']
   raw=json.loads(z.read(members[r['source']]));target=a['captions'][0]['target_id'];s,e=a['used_segment']['begin_fid'],a['used_segment']['end_fid']
   ids=prep.sample_frames(s,min(e,q['frame_count']-1),q['fps']);boxes=[]
   for i,fid in enumerate(ids,1):
    if not a['temporal_gt']['begin_fid']<=fid<a['temporal_gt']['end_fid']:continue
    obs=[o for o in raw['trajectories'][fid] if o['tid']==target]
    if not obs:continue
    assert len(obs)==1;b=obs[0]['bbox'];xy=[b[k] for k in ['xmin','ymin','xmax','ymax']];n=prep.normalize_box(xy,q['width'],q['height'])
    if n is not None:boxes.append((i,n))
   rec=prep.build_record(q['video_path'],q['caption'].strip(),boxes)
   if rec is None:rejected.append(dict(source=r['source'],reason='official_contiguous_box_builder_rejected',nframes=len(ids),observed_positions=[b[0] for b in boxes]));continue
   out.append(dict(**r,training_frame_ids=ids,record=rec,box_count=len(boxes)))
 train=sorted([r for r in out if r['split']=='train'],key=lambda r:(len(r['training_frame_ids']),r['source']))
 pick=[train[i] for i in sorted(set([0,len(train)//3,2*len(train)//3,len(train)-1]))]
 write(OUT/'SOURCE_RECORDS.json',out);write(OUT/'INTAKE.json',dict(time=time.time(),source_parents=len(rr),usable=len(out),train=len(train),validation=sum(r['split']=='validation' for r in out),rejected=rejected,pilot_sources=[r['source'] for r in pick],pilot_frames=[len(r['training_frame_ids']) for r in pick],overlap=0,exclusions_sha=sha(ROOT/'artifacts/ptd_learned_credit_v1/INTAKE.json'),official_train_sha=sha(ROOT/'external/VidSTG-Dataset/annotations/train_annotations.json'),records_sha=sha(OUT/'SOURCE_RECORDS.json')))
 print(read(OUT/'INTAKE.json'),flush=True)

class LoRALinear(nn.Module):
 def __init__(self,base,rank=32,alpha=64,dropout=.05):
  super().__init__();self.base=base;self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32));self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5));self.scale=alpha/rank;self.drop=nn.Dropout(dropout)
 @property
 def weight(self):return self.base.weight
 def forward(self,x):return self.base(x)+(F.linear(F.linear(self.drop(x).float(),self.A),self.B)*self.scale).to(x.dtype)

class RowEmbedding(nn.Module):
 def __init__(self,base,ids):
  super().__init__();self.base=base;self.num_embeddings=base.num_embeddings;self.embedding_dim=base.embedding_dim;self.register_buffer('ids',torch.tensor(ids,device=base.weight.device));self.rows=nn.Parameter(base.weight[self.ids].float().clone());mapping=torch.full((base.num_embeddings,),-1,device=base.weight.device,dtype=torch.long);mapping[self.ids]=torch.arange(len(ids),device=mapping.device);self.register_buffer('mapping',mapping)
 @property
 def weight(self):return self.base.weight
 def forward(self,x):
  base=self.base(x);ix=self.mapping[x];return torch.where((ix>=0).unsqueeze(-1),F.embedding(ix.clamp(min=0),self.rows).to(base.dtype),base)

class RowHead(nn.Module):
 def __init__(self,base,ids):
  super().__init__();self.base=base;self.register_buffer('ids',torch.tensor(ids,device=base.weight.device));self.rows=nn.Parameter(base.weight[self.ids].float().clone())
 @property
 def weight(self):return self.base.weight
 def forward(self,x):return self.base(x).index_copy(-1,self.ids,F.linear(x.float(),self.rows).to(x.dtype))

def processor():
 from transformers import AutoProcessor,AddedToken
 from model.ptd_tokens import ptd_token_strings,STRUCTURAL_TOKENS
 pr=AutoProcessor.from_pretrained(BASE,local_files_only=True);t=pr.tokenizer
 t.add_tokens([AddedToken(s,normalized=False,special=True) for s in STRUCTURAL_TOKENS],special_tokens=True)
 t.add_tokens([AddedToken(s,normalized=False,special=False) for s in ptd_token_strings()[len(STRUCTURAL_TOKENS):]],special_tokens=False)
 du=load_file('p0_data_utils',PTD/'src/dataset/data_utils.py');du.patch_qwen3_video_processor(pr);du.patch_processor_with_time_tokens(pr)
 return pr

def append_targets(pr,data,start):
 # Execute the unmodified official pure target-builder method, without importing optional trainer packages.
 tree=ast.parse((PTD/'src/dataset/sft_dataset.py').read_text());cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SupervisedDataset');fn=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='_append_ptd_targets')
 ns=dict(torch=torch,PTD_BLOCK_SIZE=6,IGNORE_INDEX=-100,DEFAULT_IM_END_TOKEN='<|im_end|>');exec(compile(ast.Module(body=[fn],type_ignores=[]),str(PTD/'src/dataset/sft_dataset.py'),'exec'),ns)
 from model.ptd_tokens import ptd_token_strings
 toks=ptd_token_strings()+['<|object_ref_start|>','<|object_ref_end|>','<|box_start|>','<|box_end|>','<|im_end|>']
 dummy=SimpleNamespace(processor=pr,ptd_token_ids={t:pr.tokenizer.convert_tokens_to_ids(t) for t in toks},newline_id=pr.tokenizer.encode('\n',add_special_tokens=False)[0],time_id_to_index={pr.tokenizer.convert_tokens_to_ids(f'<t{i}>'):i for i in range(1,101)},time_index_to_id={i:pr.tokenizer.convert_tokens_to_ids(f'<t{i}>') for i in range(1,101)},coordinate_id_to_value={pr.tokenizer.convert_tokens_to_ids(f'<{i}>'):i for i in range(1001)})
 return ns['_append_ptd_targets'](dummy,data,start)

def example(pr,r):
 from vg_tta.exact_frame_decode_audit_v2 import decode
 from qwen_vl_utils.vision_process import smart_resize
 from transformers.video_utils import VideoMetadata
 q=dict(r['input'],frame_ids=r['training_frame_ids']);frames,ids=decode(q);h,w=frames.shape[1:3];hh,ww=smart_resize(h,w,factor=32,min_pixels=100352,max_pixels=151200)
 x=torch.from_numpy(frames).permute(0,3,1,2).float();x=F.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False,antialias=True)
 text=r['record']['conversations'][0]['value'].removeprefix('<video>\n');msg=[dict(role='user',content=[dict(type='video'),dict(type='text',text=text)])];prompt=pr.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 meta=VideoMetadata(total_num_frames=q['frame_count'],fps=q['fps'],frames_indices=ids,width=w,height=h,video_backend='exact_ffmpeg_frame_ids')
 p=pr(text=[prompt],videos=[x],video_metadata=[meta],do_resize=False,do_sample_frames=False,return_tensors='pt');assert int(p['video_grid_thw'][0,0])==len(ids)
 response=pr.tokenizer.encode(r['record']['conversations'][1]['value']+'<|im_end|>\n',add_special_tokens=False);start=p['input_ids'].numel();tokens=torch.cat([p['input_ids'][0],torch.tensor(response)]);mm=p.get('mm_token_type_ids',torch.zeros_like(p['input_ids']))[0];mm=torch.cat([mm,torch.zeros(len(response),dtype=torch.long)])
 d=dict(input_ids=tokens,labels=torch.cat([torch.full((start,),-100),torch.tensor(response)]),mm_token_type_ids=mm,video_grid_thw=p['video_grid_thw'],pixel_values_videos=p['pixel_values_videos']);d=append_targets(pr,d,start)
 d['ptd_prefix_lengths']=d.pop('ptd_prefix_length').unsqueeze(0)
 for k in ['input_ids','labels','mm_token_type_ids','attention_mask','ptd_position_ids','ptd_context_limits']:d[k]=d[k].unsqueeze(0)
 return d,dict(source=r['source'],frames=len(ids),frame_ids=ids,grid=d['video_grid_thw'].tolist(),size=[hh,ww],tokens=d['input_ids'].numel(),supervised=int(d['labels'].ne(-100).sum()),ntp=int(d['labels'][0,:int(d['ptd_prefix_lengths'][0])].ne(-100).sum()))

def model_init():
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from train.monkey_patch_forward import replace_qwen3_with_ptd_forward
 from model.ptd_tokens import add_and_initialize_ptd_tokens
 replace_qwen3_with_ptd_forward();pr=AutoProcessor.from_pretrained(BASE,local_files_only=True)
 model,info=AutoModelForImageTextToText.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',output_loading_info=True);assert not info['missing_keys'] and not info['unexpected_keys']
 ids=add_and_initialize_ptd_tokens(model,pr);model.config.ptd_num_time_tokens=100;model.config.ptd_block_size=6;model.requires_grad_(False)
 replaced=[]
 for name,m in list(model.named_modules()):
  if isinstance(m,nn.Linear) and 'language_model' in name:
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,LoRALinear(m));replaced.append(name)
 oldin,oldout=model.get_input_embeddings(),model.get_output_embeddings();assert oldin.weight.data_ptr()!=oldout.weight.data_ptr(),'8B untied embedding expected'
 model.set_input_embeddings(RowEmbedding(oldin,ids));model.set_output_embeddings(RowHead(oldout,ids));model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 write(OUT/f'model_init_{time.time_ns()}.json',dict(lora_modules=replaced,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),vocab=len(pr.tokenizer),token_ids=ids,hidden=model.config.text_config.hidden_size,base_missing_keys=sorted(info['missing_keys']),base_unexpected_keys=sorted(info['unexpected_keys'])))
 return model

def joint_loss(model,d):
 labels=d['labels'];keep=torch.nonzero(labels[0,1:].ne(-100),as_tuple=False).flatten();h=model.model(**{k:v for k,v in d.items() if k!='labels'},use_cache=False).last_hidden_state[0,keep]
 targets=labels[0,keep+1];ntp=(keep+1)<d['ptd_prefix_lengths'][0];sums=[0.,0.];counts=[0,0];loss=0
 # Checkpoint only the output projection so no full sequence x vocabulary activations are retained.
 from torch.utils.checkpoint import checkpoint
 for i in range(0,len(keep),32):
  def ce(hh,tt):return F.cross_entropy(model.lm_head(hh).float(),tt,reduction='none')
  l=checkpoint(ce,h[i:i+32],targets[i:i+32],use_reentrant=False);loss=loss+l.sum()/len(keep)
  for j,mask in enumerate([ntp[i:i+32],~ntp[i:i+32]]):sums[j]+=float(l.detach()[mask].sum());counts[j]+=int(mask.sum())
 return loss,dict(ntp_loss=sums[0]/counts[0],mtp_loss=sums[1]/counts[1],ntp_count=counts[0],mtp_count=counts[1])

def pilot():
 reg=verify();from scripts.ptd_opd_information_v1 import Budget
 b=Budget(OUT,'source_joint_sft_pilot',reg['gpu_seconds'])
 try:
  torch.manual_seed(20260927);pr=processor();rmap={r['source']:r for r in read(OUT/'SOURCE_RECORDS.json')};src=read(OUT/'INTAKE.json')['pilot_sources'];model=model_init();params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0);history=[]
  basehash=hashlib.sha256(model.get_input_embeddings().base.weight[:100].detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
  for source in src:
   tick=time.monotonic();d,meta=example(pr,rmap[source]);write(OUT/f'example_{source}.json',meta);d={k:v.to('cuda') for k,v in d.items()};model.train();model.model.visual.eval();opt.zero_grad(set_to_none=True);loss,stats=joint_loss(model,d);assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,.3));assert math.isfinite(grad) and grad>0
   rowgrad=float(model.get_input_embeddings().rows.grad.norm());outgrad=float(model.lm_head.rows.grad.norm());assert rowgrad>0 and outgrad>0
   opt.step();torch.cuda.synchronize();b.calls+=1;b.check();history.append(dict(**meta,**stats,loss=float(loss.detach()),gradient_norm=grad,input_row_grad=rowgrad,output_row_grad=outgrad,seconds=time.monotonic()-tick,peak_bytes=torch.cuda.max_memory_allocated()));write(OUT/f'step_{source}.json',history[-1]);print('STEP',history[-1],flush=True);del d,loss;gc.collect();torch.cuda.empty_cache()
  assert basehash==hashlib.sha256(model.get_input_embeddings().base.weight[:100].detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
  state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad};torch.save(state,OUT/'PILOT_ADAPTER.pt');reload=torch.load(OUT/'PILOT_ADAPTER.pt',weights_only=True);assert all(torch.equal(v,reload[k]) for k,v in state.items())
  save_pr=__import__('transformers').AutoProcessor.from_pretrained(BASE,local_files_only=True);save_pr.tokenizer=pr.tokenizer;save_pr.save_pretrained(OUT/'processor_complete');write(OUT/'PILOT_COMPLETE.json',dict(time=time.time(),history=history,adapter_sha=sha(OUT/'PILOT_ADAPTER.pt'),save_readback_exact=True,old_embedding_sample_unchanged=True,task_teacher=False,GRPO_done=False,teacher_quality_unmeasured=True));status(OUT/'STATUS.json',dict(state='pilot_complete_not_task_teacher',time=time.time()))
 except BaseException as e:
  f=dict(time=time.time(),error=repr(e),traceback=traceback.format_exc());write(OUT/'failures'/f'{time.time_ns()}.json',f);status(OUT/'STATUS.json',dict(state='failed',**f));raise
 finally:b.close()

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['register','intake','pilot']);a=p.parse_args();globals()[a.action]()
