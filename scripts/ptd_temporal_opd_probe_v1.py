"""Separate eight-parent temporal distribution OPD; fresh conditional spatial decode."""
import argparse,gc,os,re,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
from scripts.ptd_opd_information_v1 import Budget
from scripts.ptd_dense_opd_v1 import rows,digest,parent_path
from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for,infer
from vg_tta.ptd_opd_information_v1 import fit
OUT=ROOT/'artifacts/ptd_temporal_opd_probe_v1'

def chosen():return [r for r in rows() if r['p0']]
def file(row,folder):return OUT/folder/(digest(row['key'])+'.pt')

class TimeAdapter(torch.nn.Module):
    def __init__(self,dim,rank=16):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(20260924)
            self.down=torch.nn.Linear(dim,rank,bias=False,dtype=torch.float32)
            self.up=torch.nn.Linear(rank,32,bias=False,dtype=torch.float32)
            torch.nn.init.zeros_(self.up.weight)
    def forward(self,h):return self.up(torch.nn.functional.gelu(self.down(torch.nn.functional.layer_norm(h.float(),(h.shape[-1],)))))

@torch.inference_mode()
def temporal_infer(model,pr,inp,native,adapter=None):
    import model.ptd_generation as pg
    original_probe=pg._run_cached_ptd_probe;context={};capture={};sem_index=0
    tids=torch.tensor([pg.get_token_id(pr.tokenizer,f'<t{i}>') for i in range(1,33)],device='cuda')
    ref_end=pg.get_token_id(pr.tokenizer,'<|object_ref_end|>')
    def probe(*args,**kwargs):
        context['time']=torch.as_tensor(kwargs['query_token_ids']).flatten().tolist()==[ref_end]
        try:return original_probe(*args,**kwargs)
        finally:context['time']=False
    def semantic(*args,**kwargs):
        nonlocal sem_index
        val=native['semantic'][sem_index];sem_index+=1;return val
    def hook(module,args,output):
        if not context.get('time'):return output
        assert output.shape[1]==6
        h=args[0][0,1:3].float();base=output[0,1:3][:,tids].float();raw=output[0].argmax(-1)
        valid=bool(torch.isin(raw[1:3],tids).all())
        capture.update(h=h.cpu(),logits=base.cpu(),time_ids=tids.cpu(),raw=raw.cpu(),native_time_slots_valid=valid)
        if adapter is None:return output
        updated=base+adapter(h);capture['adapted_logits']=updated.cpu()
        if not valid:return output
        result=output.float().clone();result[0,1:3,:]=-torch.inf;result[0,1:3,tids]=updated
        return result
    handle=model.lm_head.register_forward_hook(hook)
    try:
        with patch.object(pg,'_run_cached_ptd_probe',probe),patch.object(pg,'_parse_semantic_block',semantic):
            result=infer(model,pr,inp)
        assert sem_index==len(native['semantic']) and result['semantic']==native['semantic']
        assert capture
        result['time_probe']=capture
        result['fresh_spatial_under_current_temporal_context']=True
        return result
    finally:handle.remove()

def register():
    manifest=read(ROOT/'artifacts/ptd_opd_information_v1/REGISTRATION.json')
    assert [r['key'] for r in chosen()]==manifest['temporal_keys'] and len(chosen())==8
    pins=[Path(__file__),ROOT/'vg_tta/ptd_opd_information_v1.py',ROOT/'scripts/ptd_dense_opd_teacher_v1.py',ROOT/'protocols/ptd_opd_information_v1.md']
    write(OUT/'REGISTRATION.json',dict(time=time.time(),keys=[r['key'] for r in chosen()],pins={str(p):sha(p) for p in pins},
        protected={str(ROOT/'methods/CURRENT_METHOD.json'):sha(ROOT/'methods/CURRENT_METHOD.json')},steps=3,LR=.002,rank=16,
        classes=32,parameter_count=41472,primary_step=3,gpu_seconds=3600,GT_training=False,spatial_adapter='zero/frozen',
        time_events='canonical frame index 1..32 plus newline; maps to actual PTD <t1>..<t32> and physical frame IDs',
        invalid_policy='native parser rejects reversed/malformed spans; zero metrics, no correction',
        boundary_selectors=['Start frame: ','End frame: '],numerical_tolerance_nat=.1))

def verify():
    reg=read(OUT/'REGISTRATION.json');pins=dict(reg['pins'])
    for f in sorted((OUT/'amendments').glob('*.json')):pins.update(read(f).get('pins',{}))
    for p,h in {**pins,**reg['protected']}.items():assert sha(p)==h,p

def capture(budget):
    pr=processor_load();model=model_load()
    for row in chosen():
        dest=file(row,'native')
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        z=load(parent_path(row,'clean'));frames,_=frames_for(row,'clean');inp,_=inputs_for(row,pr,frames)
        baseline=temporal_infer(model,pr,inp,z)
        assert baseline['completion']==z['completion'] and torch.equal(baseline['logits'],z['logits'])
        probe=baseline['time_probe'];zero=TimeAdapter(probe['h'].shape[-1]).cuda()
        repeated=temporal_infer(model,pr,inp,z,zero)
        assert repeated['completion']==baseline['completion'] and torch.equal(repeated['time_probe']['logits'],probe['logits'])
        mapping=[dict(index=i+1,token_id=int(probe['time_ids'][i]),frame_id=row['input']['frame_ids'][i],seconds=row['input']['frame_ids'][i]/row['input']['fps']) for i in range(32)]
        baseline.update(key=row['key'],mapping=mapping,zero_adapter_exact=True,GT_used=False)
        save(dest,baseline);write(dest.with_suffix('.json'),dict(sha=sha(dest),mapping=mapping))
        budget.calls+=2;budget.check();print('TIME_CAPTURE',row['key'],baseline['interval'],flush=True)
        del frames,inp,baseline,repeated;gc.collect();torch.cuda.empty_cache()
    write(OUT/'CAPTURE_COMPLETE.json',dict(count=8,calls=16,GT_read=False))

def teacher(budget):
    from scripts.ptd_dense_opd_teacher_v1 import Teacher
    t=Teacher(budget)
    t.events=[t.pr.tokenizer.encode(str(i)+'\n',add_special_tokens=False) for i in range(1,33)]
    prefixes={tuple(e[:k]) for e in t.events for k in range(1,len(e))}
    t.nodes=sorted(prefixes,key=lambda p:(len(p),p));t.node_index={p:i for i,p in enumerate(t.nodes)}
    t.keep=sorted({v for e in t.events for v in e});t.columns={v:i for i,v in enumerate(t.keep)}
    for row in chosen():
        dest=file(row,'teacher')
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        z=load(parent_path(row,'clean'));frames,_=frames_for(row,'clean');h,w=frames.shape[1:3];scale=min(1.,448/max(h,w))
        hh=max(32,min(448,round(h*scale/32)*32));ww=max(32,min(448,round(w*scale/32)*32))
        x=torch.from_numpy(frames).permute(0,3,1,2).float()
        x=torch.nn.functional.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False,antialias=True)
        content=[]
        for i,fid in enumerate(row['input']['frame_ids']):
            content.extend([dict(type='text',text=f'Frame {i+1}; time {fid/row["input"]["fps"]:.6f} seconds.'),dict(type='image')])
        ref=re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>',z['completion'],re.S).group(1)
        content.append(dict(type='text',text=f'Query: {row["input"]["caption"]}\nObject reference: {ref}\n'
            'Locate the time interval in which the queried event occurs. The start is the first sampled frame of that event; the end is the last sampled frame of that event. '
            'Use frame indices from 1 to 32, inclusive. Return only the requested boundary as one canonical decimal frame index followed by a newline. Do not output the other boundary or explanations.'))
        text=t.pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        inp=t.pr(text=[text],images=list(x),do_resize=False,return_tensors='pt');p=t.prepare(inp);items=[];checks=[]
        for selector in ['Start frame: ','End frame: ']:
            base=t.pr.tokenizer.encode(text+selector,add_special_tokens=False)
            assert base==t.pr.tokenizer.encode(text,add_special_tokens=False)+t.pr.tokenizer.encode(selector,add_special_tokens=False)
            for i in range(32):assert t.pr.tokenizer.encode(text+selector+str(i+1)+'\n',add_special_tokens=False)==base+t.events[i]
            cache,_=t.prefill(p);out=t.distribution(p,cache,selector);del cache;gc.collect();torch.cuda.empty_cache()
            errors=[]
            for i in [0,15,31]:
                direct=t.independent(p,selector,i);errors.append(dict(frame_index=i+1,error=abs(direct-float(out['raw'][i])),direct=direct,trie=float(out['raw'][i])))
            checks.append(dict(selector=selector,errors=errors));assert max(v['error'] for v in errors)<=.1
            items.append(out);print('TIME_TEACHER',row['key'],selector,[v['error'] for v in errors],flush=True)
        result=dict(key=row['key'],logp=torch.stack([v['logp'] for v in items]),raw=torch.stack([v['raw'] for v in items]),checks=checks,
            prompt=text,input_ids=inp['input_ids'],event_ids=t.events,mapping=load(file(row,'native'))['mapping'],GT_used=False)
        save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
        del inp,p,frames,x;gc.collect();torch.cuda.empty_cache();budget.check()
    write(OUT/'TEACHER_COMPLETE.json',dict(distributions=16,sources=8,GT_read=False))

def train():
    torch.set_num_threads(4)
    for row in chosen():
        dest=file(row,'fits')
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        native=load(file(row,'native'))['time_probe'];target=load(file(row,'teacher'))['logp']
        result=fit(native['h'],native['logits'],target,[0,1],seed=int(digest(row['key'])[:8],16),model=TimeAdapter(native['h'].shape[-1]))
        result.update(key=row['key'],primary_step=3)
        save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
        print('TIME_FIT',row['key'],result['history'][0]['tokens'].tolist(),result['history'][-1]['tokens'].tolist(),flush=True)
    write(OUT/'TRAIN_COMPLETE.json',dict(fits=8,optimizer_steps=24,GT_read=False))

def replay(budget):
    pr=processor_load();model=model_load()
    for row in chosen():
        dest=file(row,'predictions')
        if dest.exists():assert sha(dest)==read(dest.with_suffix('.json'))['sha'];continue
        z=load(parent_path(row,'clean'));native=load(file(row,'native'));fitted=load(file(row,'fits'))
        frames,_=frames_for(row,'clean');inp,_=inputs_for(row,pr,frames)
        adapter=TimeAdapter(native['time_probe']['h'].shape[-1]).cuda();adapter.load_state_dict(fitted['history'][-1]['state'])
        result=temporal_infer(model,pr,inp,z,adapter)
        assert torch.equal(result['time_probe']['h'],native['time_probe']['h'])
        assert torch.equal(result['time_probe']['adapted_logits'].argmax(-1),fitted['history'][-1]['tokens'])
        result.update(key=row['key'],GT_used=False)
        save(dest,result);write(dest.with_suffix('.json'),dict(sha=sha(dest)))
        budget.calls+=1;budget.check();print('TIME_REPLAY',row['key'],native['interval'],result['interval'],result['format_ok'],flush=True)
        del inp,frames;gc.collect();torch.cuda.empty_cache()
    files={str(p):sha(p) for folder in ['native','teacher','fits','predictions'] for p in (OUT/folder).glob('*.pt')}
    write(OUT/'PREDICTION_BARRIER.json',dict(files=files,time=time.time(),count=8,GT_read=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['register','capture','teacher','train','replay']);args=ap.parse_args()
    if args.action=='register':register();return
    verify()
    if args.action=='train':train();return
    b=Budget(OUT,args.action,3600)
    try:
        globals()[args.action](b);status(OUT/'STATUS.json',dict(state='completed',stage=args.action,time=time.time()))
    except BaseException as e:
        failure=dict(stage=args.action,error=repr(e),traceback=traceback.format_exc(),time=time.time())
        write(OUT/'failures'/f'{time.time_ns()}.json',failure);status(OUT/'STATUS.json',dict(state='failed',**failure));raise
    finally:b.close()
if __name__=='__main__':main()
