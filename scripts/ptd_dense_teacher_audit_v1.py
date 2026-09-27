"""Bounded PTD dense teacher preparation and inference. No label reads."""
import argparse, collections, gc, hashlib, json, math, os, re, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.decota_matrix_common_v1 import read,write,status,load,sha
from scripts.ptd_spatial_adapter_ab_v1 import path as parent_path,frames_for
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter,anchor_indices,boxes_from_tokens
from scripts.decota_corrective_structured_v2 import prefix_ok as item_prefix,finished as item_finished,Grammar
PARENT=ROOT/'artifacts/ptd_spatial_adapter_ab_v1'
OUT=ROOT/'artifacts/ptd_dense_teacher_audit_v1'
MODEL=ROOT/'checkpoints/Qwen3-VL-8B-Instruct'
PROTOCOL=ROOT/'protocols/ptd_dense_teacher_audit_v1.md'

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def gaussian(tokens,sigma=5):
    axis=torch.arange(1001,dtype=torch.float32,device=tokens.device)
    return (-.5*((axis-tokens.float()[...,None])/sigma)**2).softmax(-1)
def nested(z,row,k):
    n=len(z['positions']);selected=set(anchor_indices(n))
    if k==4:return sorted(selected)
    times=[row['input']['frame_ids'][p] for p in z['positions']]
    while len(selected)<min(n,k):
        choices=[j for j in range(n) if j not in selected]
        selected.add(min(choices,key=lambda j:(-min(abs(times[j]-times[a]) for a in selected),times[j])))
    return sorted(selected)
def prefix_ok(s,n):
    if not s:return True
    if s[0]!='[':return False
    rest=s[1:];done=0
    while True:
        end=rest.find('}')
        if end<0:return done<n and item_prefix(rest,True)
        one=rest[:end+1]
        if not item_finished(one,True):return False
        done+=1;rest=rest[end+1:]
        if done==n:return ']'.startswith(rest)
        if not rest:return True
        if not rest.startswith(','):return False
        rest=rest[1:]
def finished(s,n):
    if not prefix_ok(s,n):return False
    try:x=json.loads(s)
    except ValueError:return False
    return isinstance(x,list) and len(x)==n
class PanelGrammar(Grammar):
    def constraint(self,start,n):
        def allowed(batch,tokens):
            s=self.tokenizer.decode(tokens[start:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False);key=(s,n)
            if key not in self.cache:
                out=self.eos if finished(s,n) else [i for i,v in self.pieces if prefix_ok(s+v,n)]
                assert out,('panel grammar dead end',s);self.cache[key]=out
            return self.cache[key]
        return allowed
def resize(im,edge):
    w,h=im.size;scale=min(1.,edge/max(h,w));ww=max(32,round(w*scale/32)*32);hh=max(32,round(h*scale/32)*32)
    return im.resize((ww,hh),Image.Resampling.LANCZOS)
def marked(im,xy):
    im=im.copy();w,h=im.size;box=[float(xy[0])*w,float(xy[1])*h,float(xy[2])*w,float(xy[3])*h]
    if box[2]>box[0] and box[3]>box[1]:ImageDraw.Draw(im).rectangle(box,outline=(255,0,0),width=2)
    return im
def prompt(row,z,indices,view):
    lines=["Locate the object described by the query in every numbered frame, using the complete chronological trajectory to resolve its identity.",
           'Query: '+row['input']['caption'],
           'A red rectangle is the current student prediction; it may be wrong. For each frame choose 0=KEEP if correct, 1=REVISE if you can give a better box, 2=ABSTAIN if the target or its boundaries cannot be determined.',
           'Return exactly one JSON array, in frame order, with one object per numbered frame. Each object must be {"action":0,"box":null}, {"action":1,"box":[x1,y1,x2,y2]}, or {"action":2,"box":null}. Coordinates are integer 0..1000 relative to the FULL frame, never to a crop. Do not explain.',
           'Images: '+('one full frame per numbered frame.' if view=='A' else 'for each numbered frame, FULL frame first, then its enlarged student-box context crop. The crop does not change the FULL-frame coordinate system.')]
    for ordinal,j in enumerate(indices):
        pos=z['positions'][j];fid=row['input']['frame_ids'][pos]
        lines.append(f'Frame {ordinal+1}: physical_frame={fid}, time_seconds={fid/row["input"]["fps"]:.6f}, student_xyxy='+json.dumps(z['base_tokens'][j].tolist(),separators=(',',':')))
    return '\n'.join(lines)
def prepare():
    start=time.perf_counter();OUT.mkdir(exist_ok=True);assert not (OUT/'LOCK.json').exists(),'Do not overwrite lock'
    pp=read(PARENT/'LOCK.json');rows=[r for r in read(PARENT/'INPUTS.json') if r['split']=='development'];assert len(rows)==16
    torch.set_num_threads(4);files={};audits=[];jobs=[];packing=[];seen={}
    for r in rows:
        f=parent_path(r,'clean');assert sha(f)==read(f.with_suffix('.json'))['sha'];files[str(f)]=sha(f);z=load(f)
        assert z['format_ok'] and z['logits'].shape==(len(z['positions']),4,1001)
        assert torch.isfinite(z['logits']).all() and z['coordinate_valid'].all()
        assert torch.equal(z['logits'].argmax(-1),z['base_tokens'])
        assert torch.equal(z['coord_ids'][z['base_tokens']],z['raw_blocks'][:,1:5])
        b,v=boxes_from_tokens(z['base_tokens']);assert torch.equal(b,z['boxes']) and torch.equal(v,z['geometry_valid'])
        probability=z['logits'].softmax(-1);assert torch.allclose(probability.sum(-1),torch.ones_like(probability[...,0]),atol=1e-6)
        audit=dict(key=r['key'],positions=len(z['positions']),argmax_exact=True,boxes_exact=True)
        if r['smoke']:
            smoke=PARENT/'smoke'/(r['key'].replace(':','_')+'.pt');files[str(smoke)]=sha(smoke);old=load(smoke)
            assert old['zero_exact'] and old['repeat_exact'] and old['nonzero_adapter_full_replay_exact']
            assert torch.equal(old['expected'],old['actual'])
            a=Adapter(z['h'].shape[-1]);assert sum(p.numel() for p in a.parameters())==56976
            init={k:v.clone() for k,v in a.state_dict().items()};assert torch.equal(z['logits']+a(z['h']),z['logits'])
            target=gaussian((z['base_tokens']+20).clamp(0,1000));opt=torch.optim.AdamW(a.parameters(),lr=0.,weight_decay=0.)
            def loss():return -(target*(z['logits']+a(z['h'])).log_softmax(-1)).sum(-1).mean()
            before=float(loss().detach());loss().backward();grad=float(a.up.weight.grad.norm());assert grad>0;opt.step()
            assert all(torch.equal(v,init[k]) for k,v in a.state_dict().items())
            opt=torch.optim.AdamW(a.parameters(),lr=.002,weight_decay=0.);opt.zero_grad();loss().backward();opt.step()
            after=float(loss().detach());updated=z['logits']+a(z['h']);assert not torch.equal(updated,z['logits'])
            clone=Adapter(z['h'].shape[-1]);clone.load_state_dict(a.state_dict());assert torch.equal(clone(z['h']),a(z['h']))
            audit.update(saved_full_model_replay_exact=True,zero_lr_exact=True,reload_exact=True,up_gradient=grad,synthetic_ce_before=before,synthetic_ce_after=after,maximum_logit_change=float((updated-z['logits']).abs().max().detach()),new_backbone_forwards=0,diagnostic_states_exported=False)
        audits.append(audit)
        assert sha(r['input']['video_path'])==r['input']['video_sha256'];frames,ids=frames_for(r,'clean');assert ids==r['input']['frame_ids']
        ims={}
        for j,pos in enumerate(z['positions']):
            native=Image.fromarray(frames[pos]);xy=z['base_tokens'][j].numpy()/1000.;full=marked(resize(native,448),xy)
            w,h=native.size;center=(xy[:2]+xy[2:])/2;half=np.maximum(xy[2:]-xy[:2],.032)*.75
            lo=np.maximum(0,center-half);hi=np.minimum(1,center+half);rect=[int(math.floor(lo[0]*w)),int(math.floor(lo[1]*h)),int(math.ceil(hi[0]*w)),int(math.ceil(hi[1]*h))]
            rect[2]=max(rect[0]+1,rect[2]);rect[3]=max(rect[1]+1,rect[3]);crop=native.crop(rect)
            origin=np.array(rect[:2]);span=np.array(rect[2:])-origin;local=np.concatenate(((xy[:2]*[w,h]-origin)/span,(xy[2:]*[w,h]-origin)/span));crop=marked(resize(crop,224),local)
            base=OUT/'images'/r['key'].replace(':','_');base.mkdir(parents=True,exist_ok=True)
            ff=base/f'{j:02d}_full.png';cc=base/f'{j:02d}_crop.png';full.save(ff);crop.save(cc);ims[j]=[str(ff),str(cc)]
        for k in [4,16]:
            indices=nested(z,r,k)
            for view in ['A','B']:
                images=[f for j in indices for f in (ims[j][:1] if view=='A' else ims[j])];text=prompt(r,z,indices,view)
                ident=r['key']+f'|K{k}|{view}';signature=digest(json.dumps([text,[sha(f) for f in images]],ensure_ascii=False))
                job=dict(id=ident,key=r['key'],cohort=r['cohort'],source=r['source'],K=k,view=view,local_indices=indices,positions=[z['positions'][j] for j in indices],native_tokens=z['base_tokens'][indices].tolist(),images=images,image_shas=[sha(f) for f in images],prompt=text,signature=signature,reuse=seen.get(signature))
                seen.setdefault(signature,ident);jobs.append(job)
        if len(z['positions'])==32:
            packing.append(dict(key=r['key'],K=32,chunks=[list(range(16)),list(range(16,32))],actual_unique_positions=32,teacher_calls=0))
    examples=[]
    for n in [1,2,4,16]:
        x=json.dumps([{'action':i%3,'box':[0,1,999,1000] if i%3==1 else None} for i in range(n)],separators=(',',':'))
        assert finished(x,n) and all(prefix_ok(x[:i],n) for i in range(len(x)+1));assert not prefix_ok(x+'x',n);examples.append(x)
    assert not prefix_ok('[{"action":1,"box":[1001',1)
    write(OUT/'INPUTS.json',rows);write(OUT/'JOBS.json',jobs)
    write(OUT/'INTERFACE_AUDIT.json',dict(status='pass',rows=audits,smoke_count=4,pack32=packing,grammar_examples=examples,scope='All 16 cache readback plus four new CPU synthetic-gradient diagnostics and saved exact full-model replay audit; no new student/teacher inference or GT',seconds=time.perf_counter()-start))
    protected={f:sha(ROOT/f) for f in pp['protected']}
    lock=dict(created=time.time(),authority='./private_authorization_notes/336da276-35db-45cb-a870-00b1012265b6/已粘贴的文本.txt',protocol_sha=sha(PROTOCOL),inputs_sha=sha(OUT/'INPUTS.json'),jobs_sha=sha(OUT/'JOBS.json'),parent_files=files,parent_lock_sha=sha(PARENT/'LOCK.json'),checkpoint=pp['checkpoint'],labels=pp['labels'],labels_sha=pp['labels_sha'],protected=protected,teacher_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',sigma=5,gamma_R=.5,gamma_D=1.,substantial=.02,max_tokens=1536,max_seconds=7200,peak_bytes=28*2**30,logical_requests=len(jobs),generation_requests=sum(j['reuse'] is None for j in jobs),GT_worker=False,student_updates=0)
    write(OUT/'LOCK.json',lock);status(OUT/'STATUS.json',dict(phase='prepared_waiting_8B',interface='pass',logical_requests=len(jobs),generation_requests=lock['generation_requests'],GT_used=False))
    print('PREPARED',len(rows),len(jobs),lock['generation_requests'],flush=True)
def verify():
    p=read(OUT/'LOCK.json');assert sha(PROTOCOL)==p['protocol_sha'];assert sha(OUT/'INPUTS.json')==p['inputs_sha'];assert sha(OUT/'JOBS.json')==p['jobs_sha']
    for f,h in p['parent_files'].items():assert sha(f)==h,f
    for f,h in p['protected'].items():assert sha(ROOT/f)==h,f
    manifest=OUT/'PRE_EXECUTION_CODE_MANIFEST.json'
    if manifest.exists():
        for f,h in read(manifest)['files'].items():assert sha(ROOT/f)==h,('execution dependency changed',f)
    return p
def serialize(pr,job):
    content=[]
    stride=1 if job['view']=='A' else 2
    for i in range(len(job['local_indices'])):
        content.append(dict(type='text',text=f'Frame {i+1}, full image:'))
        content.append(dict(type='image'))
        if stride==2:content.extend([dict(type='text',text=f'Frame {i+1}, context crop:'),dict(type='image')])
    content.append(dict(type='text',text=job['prompt']))
    return pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
def pack_audit():
    from transformers import AutoProcessor
    start=time.perf_counter();p=verify();pr=AutoProcessor.from_pretrained(MODEL,local_files_only=True);rows=[];chunks32=[]
    for j in read(OUT/'JOBS.json'):
        ims=[Image.open(f).convert('RGB') for f in j['images']];text=serialize(pr,j);x=pr(text=[text],images=ims,do_resize=False,return_tensors='pt')
        assert len(x['image_grid_thw'])==len(ims);assert torch.isfinite(x['pixel_values']).all()
        rows.append(dict(id=j['id'],text_sha=digest(text),input_ids_sha=hashlib.sha256(x['input_ids'].numpy().tobytes()).hexdigest(),prompt_tokens=x['input_ids'].shape[1],image_count=len(ims),grid=x['image_grid_thw'].tolist(),pixel_sha=hashlib.sha256(x['pixel_values'].numpy().tobytes()).hexdigest()))
    for r in read(OUT/'INPUTS.json'):
        z=load(parent_path(r,'clean'))
        if len(z['positions'])!=32:continue
        for begin in [0,16]:
            for view in ['A','B']:
                indices=list(range(begin,begin+16));base=OUT/'images'/r['key'].replace(':','_')
                ims=[Image.open(base/f'{j:02d}_{name}.png').convert('RGB') for j in indices for name in (['full'] if view=='A' else ['full','crop'])]
                job=dict(view=view,local_indices=indices,prompt=prompt(r,z,indices,view));text=serialize(pr,job);x=pr(text=[text],images=ims,do_resize=False,return_tensors='pt')
                assert len(x['image_grid_thw'])==len(ims) and torch.isfinite(x['pixel_values']).all()
                chunks32.append(dict(key=r['key'],view=view,local_indices=indices,image_count=len(ims),prompt_tokens=x['input_ids'].shape[1],generation_calls=0))
    write(OUT/'PACK_AUDIT.json',dict(status='pass',jobs=rows,K32_chunks=chunks32,max_prompt_tokens=max(r['prompt_tokens'] for r in rows),generation_calls=0,seconds=time.perf_counter()-start));print('PACK_PASS',max(r['prompt_tokens'] for r in rows),flush=True)
def run():
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from scripts.run_final_simplification_v1 import lease
    p=verify();pack={x['id']:x for x in read(OUT/'PACK_AUDIT.json')['jobs']};jj=read(OUT/'JOBS.json');guard=lease();start=time.perf_counter();count=0;receipts=[];torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    try:
        manifest=read(MODEL/'DOWNLOAD_MANIFEST.json');assert manifest['revision']==p['teacher_revision']
        for name,spec in manifest['files'].items():
            assert (MODEL/name).stat().st_size==spec['size']
            if spec.get('lfs'):assert sha(MODEL/name)==spec['lfs']['sha256']
        model,info=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not info['missing_keys'] and not info['unexpected_keys'];model.eval().requires_grad_(False);versions=[a._version for a in model.parameters()];pr=AutoProcessor.from_pretrained(MODEL,local_files_only=True);grammar=PanelGrammar(pr.tokenizer,model.generation_config.eos_token_id)
        write(OUT/'EXECUTION_LOCK.json',dict(code_sha=sha(__file__),lock_sha=sha(OUT/'LOCK.json'),pack_sha=sha(OUT/'PACK_AUDIT.json'),started=time.time(),model_sha_manifest=sha(MODEL/'DOWNLOAD_MANIFEST.json'),GT_used=False))
        for job in jj:
            dest=OUT/'teacher_results'/(digest(job['id'])+'.json');assert not dest.exists(),'No automatic redispatch'
            if job['reuse']:
                origin=OUT/'teacher_results'/(digest(job['reuse'])+'.json');old=read(origin)
                write(dest,{**old,'id':job['id'],'reused_from':job['reuse'],'generation_call':False,'seconds':0.,'origin_sha':sha(origin)})
            else:
                assert time.perf_counter()-start<p['max_seconds']
                for f,h in zip(job['images'],job['image_shas']):assert sha(f)==h
                images=[Image.open(f).convert('RGB') for f in job['images']];text=serialize(pr,job);x=pr(text=[text],images=images,do_resize=False,return_tensors='pt')
                assert digest(text)==pack[job['id']]['text_sha'];assert hashlib.sha256(x['input_ids'].numpy().tobytes()).hexdigest()==pack[job['id']]['input_ids_sha']
                x=x.to('cuda');n=x['input_ids'].shape[1];t=time.perf_counter()
                status(OUT/'STATUS.json',dict(phase='teacher_readout',id=job['id'],completed=count,seconds=time.perf_counter()-start))
                dispatched=OUT/'dispatch'/(digest(job['id'])+'.json');assert not dispatched.exists(),'No redispatch of uncertain previous call'
                assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<4*2**30,'OUTPUT_BUDGET'
                write(dispatched,dict(id=job['id'],started=time.time(),GT_used=False))
                with torch.inference_mode():res=model.generate(**x,max_new_tokens=p['max_tokens'],do_sample=False,use_cache=True,prefix_allowed_tokens_fn=grammar.constraint(n,len(job['local_indices'])))
                tokens=res[0,n:].cpu().tolist();answer=pr.tokenizer.decode(tokens,skip_special_tokens=True);valid=finished(answer,len(job['local_indices']))
                write(dest,dict(id=job['id'],text=answer,tokens=tokens,format_valid=valid,parsed=json.loads(answer) if valid else None,seconds=time.perf_counter()-t,prompt_tokens=n,generated_tokens=len(tokens),generation_call=True,GT_used=False))
                count+=1;assert valid,('format_failure',job['id']);assert torch.cuda.max_memory_allocated()<=p['peak_bytes'],'GPU_PEAK_BUDGET'
                del x,res;gc.collect();grammar.cache.clear()
            receipts.append(dict(path=str(dest),sha=sha(dest)));print('DENSE',len(receipts),len(jj),job['id'],'calls',count,'seconds',round(time.perf_counter()-start,1),flush=True)
        assert count==p['generation_requests'];assert versions==[a._version for a in model.parameters()];assert time.perf_counter()-start<=p['max_seconds']
        write(OUT/'TEACHER_BARRIER.json',dict(files=receipts,logical_count=len(receipts),generation_calls=count,model_unchanged=True,GT_used=False,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated()))
        status(OUT/'STATUS.json',dict(phase='teacher_sealed_pending_offline_score',generation_calls=count,student_updates=0))
    finally:
        write(OUT/'worker_receipts'/f'{time.time_ns()}.json',dict(seconds=time.perf_counter()-start,generation_calls=count,peak_bytes=torch.cuda.max_memory_allocated()));guard.close()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','pack','run']);x=a.parse_args()
    {'prepare':prepare,'pack':pack_audit,'run':run}[x.action]()
