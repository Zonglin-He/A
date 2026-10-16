"""Exact private feature/token reuse; identical native WHO selection and decode."""
from pathlib import Path
import gc,hashlib,inspect,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_spatial_controls256_native_v1 as original
from scripts import stvg_pipeline_speed_v1 as speed
BASE,read,put,sha=original.BASE,original.read,original.put,original.sha

def verify():
    from scripts.continue_stvg_spatial_controls256_native003 import verify as old_verify
    old_verify()
    for p,h in read(BASE/'NATIVE_DISPATCH_RUNTIME004.json')['pins'].items():assert sha(ROOT/p)==h,p

def cache_load(processor,job,device,model):
    import torch,numpy as np
    from scripts.stvg_referent_abc_native002 import processor_batch
    dest=speed.BASE/'teacher_features'/job['job_id'];dest.mkdir(parents=True,exist_ok=True)
    prior=BASE/'native_cache'/job['job_id'];policy=read(prior/'POLICIES.json')
    bind=dict(job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),native_policy_sha256=sha(prior/'POLICIES.json'),native_sha256=policy['native_sha256'],checkpoint_config_sha256=sha(ROOT/'checkpoints/STVG-R1-7B/config.json'))
    assert bind['job_sha256']==policy['job_sha256'] and model.dtype==torch.bfloat16
    fresh=not(dest/'RECEIPT.json').exists()
    if fresh:
        assert not(dest/'tokens.pt').exists() and not(dest/'features.npy').exists()
        batch,ps=processor_batch(processor,job,ROOT,'cpu')
        metadata={k:v for k,v in batch.items() if k!='pixel_values_videos'}
        assert set(metadata)=={'input_ids','attention_mask','video_grid_thw','second_per_grid_ts'},set(metadata)
        with np.load(prior/'native.npz',allow_pickle=False) as z:features=z['features'].copy()
        assert features.dtype==np.float32 and features.shape[0]==1280
        bits=features.view(np.uint32);assert np.all((bits & 65535)==0)
        stored=(bits>>16).astype(np.uint16)
        assert np.array_equal((stored.astype(np.uint32)<<16).view(np.float32),features)
        torch.save(dict(metadata=metadata,prompt_sha256=ps),dest/'tokens.pt');np.save(dest/'features.npy',stored,allow_pickle=False)
        put(dest/'RECEIPT.json',dict(status='sealed',binding=bind,tokens_sha256=sha(dest/'tokens.pt'),features_sha256=sha(dest/'features.npy'),prompt_sha256=ps,lossless_original_bfloat16_storage=True,GT_read=False))
        del batch,features,bits,stored,metadata
    receipt=read(dest/'RECEIPT.json');assert receipt['binding']==bind
    assert sha(dest/'tokens.pt')==receipt['tokens_sha256'] and sha(dest/'features.npy')==receipt['features_sha256']
    # The first materialization checks every input file/tensor in the original
    # processor. Subsequent reads also bind the immutable underlying video files.
    for v in job['videos']:assert sha(ROOT/v['path'])==v['sha256']
    obj=torch.load(dest/'tokens.pt',map_location='cpu',weights_only=False)
    batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in obj['metadata'].items()}
    z=np.load(dest/'features.npy',allow_pickle=False,mmap_mode='r');features=torch.from_numpy(np.array(z,copy=True)).view(torch.bfloat16).to(device)
    return batch,obj['prompt_sha256'],features,fresh

class Cache:
    def __init__(self,arm):self.arm=arm;self.rows=[];self.qualified=set()
    def prepare(self,m,p,device,j,proof):
        import torch,numpy as np
        from scripts.stvg_referent_abc_native002 import processor_batch,pack,sequence_keep,greedy
        prior=BASE/'native_cache'/j['job_id'];s=read(prior/'POLICIES.json');assert s['model_clock_proof']==proof
        t=time.perf_counter();b,ps,f,fresh=cache_load(p,j,device,m)
        with torch.inference_mode():e,pos,_,f,runs=pack(m,b,f)
        qualification=False
        if j['dataset'] not in self.qualified and (self.arm=='A' or s['caption_available']):
            original_batch,original_ps=processor_batch(p,j,ROOT,device)
            for k in b:speed.compare_tree(b[k],original_batch[k],'cached_token_metadata.'+k)
            assert ps==original_ps
            with np.load(prior/'native.npz',allow_pickle=False) as a,torch.inference_mode():oe,op,_,of,orr=pack(m,original_batch,torch.from_numpy(a['features']).to(device=device,dtype=m.dtype))
            assert torch.equal(e,oe) and torch.equal(pos,op) and torch.equal(f,of) and all(torch.equal(x,y) for x,y in zip(runs,orr))
            ix=[r['indices'][self.arm] for r in s['records']];keep=sequence_keep(b,m,runs,ix)
            with torch.inference_mode():a,_=greedy(m,oe,op,keep,original_batch['input_ids'],8);c,_=greedy(m,e,pos,keep,b['input_ids'],8)
            assert a==c;self.qualified.add(j['dataset']);qualification=True
            put(speed.BASE/'teacher_qualification'/f'{self.arm}_{j["dataset"]}.json',dict(status='pass',dataset=j['dataset'],count=1,token_metadata_exact=True,visual_features_embeddings_mRoPE_exact=True,original_first8_tokens_exact=True,GT_read=False))
            del original_batch,oe,op,of,orr
        self.rows.append(dict(job_id=j['job_id'],metadata_cache_new=fresh,prepare_seconds=time.perf_counter()-t,qualification=qualification,GT_read=False))
        return s,s['caption_available'],s['records'],b,ps,e,pos,f,runs
    def release(self):
        import torch
        if len(self.rows)%16==0:gc.collect()
        free,_=torch.cuda.mem_get_info()
        if free<4*2**30 or torch.cuda.memory_reserved()>24*2**30:gc.collect();torch.cuda.empty_cache()
    def receipt(self):put(speed.BASE/'teacher_profiles'/(self.arm+'.json'),dict(status='complete',arm=self.arm,count=len(self.rows),records=self.rows,GT_read=False))

def source():
    text=inspect.getsource(original.teacher)
    before="s=read(cache/'POLICIES.json');assert s['job_sha256']==hashlib.sha256(json.dumps(orig,sort_keys=True).encode()).hexdigest()and s['model_clock_proof']==proof and sha(cache/'native.npz')==s['native_sha256'];available=s['caption_available'];pol=s['records'];b,ps=processor_batch(p,j,ROOT,device)\n            with np.load(cache/'native.npz',allow_pickle=False)as a,torch.inference_mode():e,pos,_,f,runs=pack(m,b,torch.from_numpy(a['features']).to(device=device,dtype=m.dtype))"
    assert text.count(before)==1;text=text.replace(before,"s,available,pol,b,ps,e,pos,f,runs=speed_cache.prepare(m,p,device,j,proof)")
    before='del b,e,pos,f,runs,pol;gc.collect();torch.cuda.empty_cache()';assert text.count(before)==1
    return text.replace(before,'del b,e,pos,f,runs,pol;speed_cache.release()')
def teacher(arm):
    assert arm in ('A','B','C');cache=Cache(arm);ns=dict(original.__dict__);ns.update(verify=verify,speed_cache=cache)
    exec(compile(source(),str(Path(__file__)),'exec'),ns);ns['teacher'](arm);cache.receipt()
def contracts():
    import ast,numpy as np
    ast.parse(source());x=np.array([0.,1.,-2.,.5],np.float32);z=(x.view(np.uint32)>>16).astype(np.uint16);assert np.array_equal(x,(z.astype(np.uint32)<<16).view(np.float32))
    bad=np.array([.1234567],np.float32);assert np.any((bad.view(np.uint32)&65535)!=0)
    assert '96' in source() and 'greedy' in source() and 'parse(raw,j)' in source()
    print('SPEED_TEACHER_CPU_LOSSLESS_STORAGE_AND_ORIGINAL_PROTOCOL_CONTRACTS_PASS_NO_MODEL_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else teacher(sys.argv[2])
