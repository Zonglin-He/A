"""Finite registered diagnostic. Workers never load annotation files."""
import argparse, copy, gc, hashlib, json, os, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
from scripts.ptd_spatial_adapter_ab_v1 import path as parent_path,frames_for,inputs_for,processor_load,model_load,infer
from scripts.ptd_dense_teacher_audit_v1 import serialize,marked,digest
OUT=ROOT/'artifacts/ptd_visual_evidence_decomposition_v1'
DENSE=ROOT/'artifacts/ptd_dense_teacher_audit_v1'
TEACHER=ROOT/'checkpoints/Qwen3-VL-8B-Instruct'

def verify():
    r=read(OUT/'REGISTRATION.json')
    for f,h in r['pins'].items():assert sha(ROOT/f)==h,f
    if (OUT/'STAGE0_LOCK.json').exists():
        lock=read(OUT/'STAGE0_LOCK.json')
        expected=lock['code_sha']
        for p in sorted((OUT/'amendments').glob('*.json')):
            expected=read(p).get('code_sha',expected)
        assert sha(__file__)==expected
    return r

def spent():
    return sum(read(p)['seconds'] for p in (OUT/'worker_receipts').glob('*.json'))

class Budget:
    def __init__(self,stage):
        from scripts.run_final_simplification_v1 import lease
        verify();self.guard=lease();self.before=spent();self.t=time.monotonic();self.stage=stage;self.calls=0;self.tokens=0
        self.limit=1800 if stage=='stage0' else 7200
        self.r=OUT/'worker_receipts'/f'{time.time_ns()}.json'
        assert not (OUT/'ACTIVE_WORKER.json').exists(),'Unclosed worker: account time before resuming'
        write(OUT/'ACTIVE_WORKER.json',dict(pid=os.getpid(),started=time.time(),stage=stage,prior_seconds=self.before))
        prior_stage=sum(read(p)['seconds'] for p in (OUT/'worker_receipts').glob('*.json') if read(p)['stage']==self.stage)
        def expired(*args):raise TimeoutError('CUMULATIVE_GPU_BUDGET_WATCHDOG')
        signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(min(7200-self.before,self.limit-prior_stage))-2))
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    def check(self):
        seconds=time.monotonic()-self.t
        assert self.before+seconds<7200,'TOTAL_GPU_WALL_BUDGET'
        prior_stage=sum(read(p)['seconds'] for p in (OUT/'worker_receipts').glob('*.json') if read(p)['stage']==self.stage)
        assert prior_stage+seconds<self.limit,'STAGE_GPU_WALL_BUDGET'
        assert torch.cuda.max_memory_allocated()<=28*2**30,'PEAK_MEMORY_BUDGET'
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<4*2**30,'ARTIFACT_BUDGET'
    def close(self):
        signal.alarm(0)
        write(self.r,dict(stage=self.stage,seconds=time.monotonic()-self.t,prior_seconds=self.before,calls=self.calls,tokens=self.tokens,peak_bytes=torch.cuda.max_memory_allocated()))
        (OUT/'ACTIVE_WORKER.json').unlink();self.guard.close()

def prepare():
    verify();assert not (OUT/'STAGE0_LOCK.json').exists()
    rows=read(DENSE/'INPUTS.json');jobs=read(DENSE/'JOBS.json');donors={}
    for co in sorted({r['cohort'] for r in rows}):
        ordered=sorted([r for r in rows if r['cohort']==co],key=lambda r:digest(r['source']))
        for a,b in zip(ordered,ordered[1:]+ordered[:1]):donors[a['key']]=b['key']
    lock=dict(created=time.time(),code_sha=sha(__file__),pilot_keys=[r['key'] for r in rows if r['smoke']],donors=donors,
      pilot_anchor='first existing K4 anchor',pilot_coordinate=0,coordinate_events='canonical decimal digits 0..1000 followed by comma for x1/y1/x2 or closing bracket for y2',
      normalization='float64 logsumexp over all 1001 complete canonical token paths, unconstrained model probabilities; not all possible text encodings',
      prefix='unchanged K4 View-A prompt; teacher-forced native REVISE objects for preceding frames and native preceding coordinates; current action fixed REVISE',
      conditioning='teacher gets explicit native boxes, forced preceding native output and action=REVISE; PTD spatial probes get fixed semantic/time and no preceding coordinates. Compare coordinate-label distributions, not identical-conditioning policies or a joint chain-rule KL.',
      action_boundary='Because KEEP/ABSTAIN have null boxes, q is conditional on forced REVISE, not unconditional corrective policy. Save actual action evidence from prior sealed run; no new generated answers.',
      max_event_logp_error_nat=.1,normalization_abs_tol=1e-12,teacher_forest_block_paths=True,
      reference_values=[0,9,10,417,999,1000],stage0_seconds=1800,total_seconds=7200,GT_worker=False)
    write(OUT/'STAGE0_LOCK.json',lock)

def suffix_for(job,anchor,coord):
    prev=[dict(action=1,box=b) for b in job['native_tokens'][:anchor]]
    text='['+','.join(json.dumps(x,separators=(',',':')) for x in prev)
    if prev:text+=','
    text+='{"action":1,"box":['
    for v in job['native_tokens'][anchor][:coord]:text+=str(v)+','
    return text

def images_for(row,job,branch,rows,lock):
    if branch=='minus':return []
    if branch=='plus':return [Image.open(p).convert('RGB') for p in job['images']]
    donor=rows[lock['donors'][row['key']]];frames,_=frames_for(donor,'clean');ims=[]
    for pos,box,p in zip(job['positions'],job['native_tokens'],job['images']):
        size=Image.open(p).size;im=Image.fromarray(frames[pos]).resize(size,Image.Resampling.LANCZOS)
        ims.append(marked(im,np.asarray(box)/1000.))
    return ims

def teacher_inputs(pr,job,images,branch,suffix):
    text=serialize(pr,job)
    if branch=='minus':text=text.replace('<|vision_start|><|image_pad|><|vision_end|>','')
    text+=suffix
    x=pr(text=[text],images=images or None,do_resize=False,return_tensors='pt')
    if branch=='minus':
        assert 'pixel_values' not in x and 'image_grid_thw' not in x
        for t in ['<|image_pad|>','<|video_pad|>','<|vision_start|>','<|vision_end|>']:
            assert pr.tokenizer.convert_tokens_to_ids(t) not in x['input_ids']
    return x.to('cuda'),text

@torch.inference_mode()
def log_next(model,h):
    return model.lm_head(h).float().log_softmax(-1).double()

@torch.inference_mode()
def forest_events(model,x,events,budget):
    """One causal suffix per event; independent trees share immutable prefix KV."""
    budget.check();model.model.rope_deltas=None
    prefix=model(**x,use_cache=True,logits_to_keep=1,return_dict=True);budget.calls+=1;budget.tokens+=x['input_ids'].numel()
    cache=prefix.past_key_values;n=cache.get_seq_length();first=prefix.logits[0,-1].float().log_softmax(-1).double()
    delta=model.model.rope_deltas.detach().clone() if model.model.rope_deltas is not None else torch.zeros((1,1),dtype=torch.long,device='cuda')
    groups=[];offset=[];ids=[];starts=[]
    for i,e in enumerate(events):
        starts.append(len(ids));ids+=e[:-1];groups += [i]*(len(e)-1);offset+=list(range(len(e)-1))
    ids=torch.tensor([ids],device='cuda');g=torch.tensor(groups,device='cuda');o=torch.tensor(offset,device='cuda');m=ids.shape[1]
    visible=(g[:,None]==g[None,:])&(o[:,None]>=o[None,:]);mask=torch.zeros((m,n+m),dtype=torch.bfloat16,device='cuda');mask[:,n:].masked_fill_(~visible,-torch.inf)
    pos=(n+o).view(1,1,-1).expand(3,1,-1)+delta.view(1,1,1)
    out=model.model.language_model(input_ids=ids,position_ids=pos,attention_mask=mask[None,None],past_key_values=cache,use_cache=True,visual_pos_masks=None,deepstack_visual_embeds=None,return_dict=True)
    budget.calls+=1;budget.tokens+=m
    # Chunk the vocabulary projection; all head vocabulary rows enter the denominator.
    target=torch.tensor([t for e in events for t in e[1:]],device='cuda');scores=[]
    for start in range(0,m,32):
        lp=log_next(model,out.last_hidden_state[0,start:start+32]);scores.append(lp.gather(-1,target[start:start+32,None]).squeeze(-1).cpu())
    tail=torch.cat(scores);raw=torch.stack([first[e[0]].cpu()+tail[s:s+len(e)-1].sum() for e,s in zip(events,starts)])
    cache.crop(n);mass=torch.logsumexp(raw,0);lp=raw-mass
    assert torch.isfinite(raw).all() and abs(float(lp.exp().sum())-1)<1e-12
    result=dict(raw=raw,logp=lp,log_event_mass=float(mass),prefix_tokens=n,forest_tokens=m,rope_delta=delta.cpu(),event_count=len(events))
    del prefix,cache,out,mask,visible,lp;gc.collect();torch.cuda.empty_cache();budget.check();return result

@torch.inference_mode()
def complete_event(model,x,event,budget):
    n=x['input_ids'].shape[1];xx={k:v for k,v in x.items()};xx['input_ids']=torch.cat((x['input_ids'],torch.tensor([event],device='cuda')),dim=1)
    xx['attention_mask']=torch.ones_like(xx['input_ids'])
    if 'mm_token_type_ids' in xx:xx['mm_token_type_ids']=torch.cat((x['mm_token_type_ids'],torch.zeros((1,len(event)),device='cuda',dtype=x['mm_token_type_ids'].dtype)),1)
    model.model.rope_deltas=None
    out=model(**xx,use_cache=False,logits_to_keep=len(event)+1,return_dict=True);budget.calls+=1;budget.tokens+=xx['input_ids'].numel()
    logits=out.logits[0,:-1].float().log_softmax(-1).double();val=logits.gather(-1,torch.tensor(event,device='cuda')[:,None]).sum().item();del out,xx;return val

@torch.inference_mode()
def single_cached_event(model,x,event,budget):
    model.model.rope_deltas=None
    p=model(**x,use_cache=True,logits_to_keep=1,return_dict=True);budget.calls+=1;budget.tokens+=x['input_ids'].numel()
    n=p.past_key_values.get_seq_length();ids=torch.tensor([event[:-1]],device='cuda');delta=model.model.rope_deltas if model.model.rope_deltas is not None else torch.zeros((1,1),dtype=torch.long,device='cuda');pos=torch.arange(n,n+ids.shape[1],device='cuda').view(1,1,-1).expand(3,1,-1)+delta.view(1,1,1)
    o=model.model.language_model(input_ids=ids,position_ids=pos,past_key_values=p.past_key_values,use_cache=True,visual_pos_masks=None,deepstack_visual_embeds=None,return_dict=True)
    lp=log_next(model,o.last_hidden_state[0]);v=p.logits[0,-1].float().log_softmax(-1).double()[event[0]]+lp.gather(-1,torch.tensor(event[1:],device='cuda')[:,None]).sum()
    budget.calls+=1;budget.tokens+=ids.numel();return float(v)

def teacher(stage,fp32_head=False):
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    lock=read(OUT/'STAGE0_LOCK.json');rows={r['key']:r for r in read(DENSE/'INPUTS.json')};jobs={j['key']:j for j in read(DENSE/'JOBS.json') if j['K']==4 and j['view']=='A'}
    keys=lock['pilot_keys'] if stage=='stage0' else list(rows);budget=Budget(stage);t=time.monotonic();done=[]
    target_dir=OUT/('teacher_fp32head' if fp32_head else 'teacher');tag=stage+('_FP32HEAD' if fp32_head else '')
    try:
        pr=AutoProcessor.from_pretrained(TEACHER,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(TEACHER,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        if fp32_head:
            # Preserve exact checkpoint weight values; only increase precision of the
            # probability readout. BF16 transformer and all scientific conditions stay fixed.
            head_weight=model.lm_head.weight.detach().float()
            model.lm_head.forward=lambda h:torch.nn.functional.linear(h.float(),head_weight)
        for key in keys:
            job=jobs[key];row=rows[key]
            for branch in ['plus','minus','wrong']:
                images=images_for(row,job,branch,rows,lock)
                pairs=[(0,0)] if stage=='stage0' else [(a,c) for a in range(len(job['positions'])) for c in range(4)]
                for a,c in pairs:
                    dest=target_dir/f'{digest(key)}_{branch}_{a}_{c}.pt'
                    if dest.exists():
                        assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
                    suffix=suffix_for(job,a,c);x,text=teacher_inputs(pr,job,images,branch,suffix);delimiter=',' if c<3 else ']'
                    events=[pr.tokenizer.encode(str(v)+delimiter,add_special_tokens=False) for v in range(1001)]
                    # Event tokenization at this exact numeric prefix, including delimiter.
                    base=pr.tokenizer.encode(text,add_special_tokens=False)
                    for v in lock['reference_values']:
                        assert pr.tokenizer.encode(text+str(v)+delimiter,add_special_tokens=False)==base+events[v]
                    ts=time.monotonic();result=forest_events(model,x,events,budget);errors=[]
                    if stage=='stage0':
                        for v in lock['reference_values']:
                            value=complete_event(model,x,events[v],budget);cached=single_cached_event(model,x,events[v],budget)
                            errors.append(dict(value=v,forest=float(result['raw'][v]),complete=value,cached_single=cached,error=abs(value-float(result['raw'][v])),forest_vs_single=abs(cached-float(result['raw'][v])),single_vs_complete=abs(cached-value)))
                    result.update(key=key,branch=branch,anchor=a,coordinate=c,local_index=job['local_indices'][a],event_definition=lock['coordinate_events'],prefix_sha=digest(text),input_ids=x['input_ids'].cpu(),no_image_keys=list(x) if branch=='minus' else None,image_pixel_shas=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images],reference_errors=errors,seconds=time.monotonic()-ts,FP32_head=fp32_head,GT_used=False)
                    save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest),max_error=max([e['error'] for e in errors],default=0),seconds=result['seconds']))
                    print('TEACHER',key,branch,a,c,'seconds',round(result['seconds'],2),'mass',result['log_event_mass'],'maxerr',max([e['error'] for e in errors],default=0),flush=True)
                    done.append(str(dest));del x,result;gc.collect();budget.check()
                    # Stage0 is an interface measurement, not permission to train. A failed
                    # point remains failed while the original four-source diagnostic finishes.
                    if stage!='stage0':assert max([e['error'] for e in errors],default=0)<=lock['max_event_logp_error_nat']
        pilot_files=sorted(target_dir.glob('*.pt'))
        errors=[e['error'] for p in pilot_files for e in load(p)['reference_errors']]
        write(OUT/(tag+'_TEACHER_COMPLETE.json'),dict(status='completed_interface_measurement',interface_pass=all(e<=lock['max_event_logp_error_nat'] for e in errors),max_error=max(errors,default=None),files=done,all_saved_files=[str(p) for p in pilot_files],seconds=time.monotonic()-t,calls=budget.calls,tokens=budget.tokens,GT_used=False))
    except BaseException as e:
        write(OUT/'failures'/f'{stage}_teacher_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=done,seconds=time.monotonic()-t));raise
    finally:budget.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','teacher']);a.add_argument('--stage',default='stage0');a.add_argument('--fp32-head',action='store_true');x=a.parse_args()
    prepare() if x.action=='prepare' else teacher(x.stage,x.fp32_head)
