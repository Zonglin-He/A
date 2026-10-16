"""Native projected-token diagnostic. No pruning, generation or Student change.

Feature edges are a deliberately limited proxy; neither attention CS nor
semantic identity is inferred from cosine similarity or rectangle overlap.
"""
from pathlib import Path
import gc,hashlib,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_referent_evidence_p0_v1 import BASE,PRIOR,read,put,sha,verify
def ranks(x):
    import numpy as np
    x=np.asarray(x,dtype=float);order=np.argsort(x,kind='stable');s=x[order];out=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x)and s[j]==s[i]:j+=1
        out[order[i:j]]=(i+j-1)/2/max(1,len(x)-1);i=j
    return out
def top(x,k,exclude=()):
    import numpy as np
    a=np.asarray(x,dtype=float).copy();a[list(exclude)]=-np.inf
    return np.lexsort((np.arange(len(a)),-a))[:k]
def analyze(z,q,c,swapped,gh,gw):
    import numpy as np
    T,P,D=z.shape;assert P==gh*gw and q.shape[1]==c.shape[1]==swapped.shape[1]==D
    norm=lambda a:a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-6)
    v=norm(np.asarray(z,dtype=np.float32));qq,cc,ss=[norm(np.asarray(a,dtype=np.float32))for a in (q,c,swapped)]
    def response(text):
        m=v@text.T;k=min(2,len(text));return np.sort(m,axis=-1)[...,-k:].mean(-1)
    aq,ac,aswap=map(response,(qq,cc,ss));rq,rc,rs=[np.stack([ranks(a)for a in x])for x in (aq,ac,aswap)]
    u=np.minimum(rq,rc);d=np.abs(rq-rc);K=max(1,int(math.ceil(.125*P)))
    selections={n:np.stack([top(a,K)for a in x])for n,x in [('query',rq),('caption',rc),('consensus',u),('disagreement',d),('query_swap',rs)]}
    xy=np.stack(np.meshgrid((np.arange(gw)+.5)/gw,(np.arange(gh)+.5)/gh),-1).reshape(P,2)
    near=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=-1)<=.25;edges=[];edge_score=np.zeros((T,P));cuts=[]
    for t in range(T-1):
        # Reciprocal correspondence plus a geometric admissibility test. No
        # forced edge when global appearance changes sharply or match differs.
        scene_cos=float(norm(z[t].mean(0))@norm(z[t+1].mean(0)));cut=scene_cos<.5;cuts.append(cut)
        sim=v[t]@v[t+1].T;forward=sim.argmax(1);backward=sim.argmax(0)
        for i,j in enumerate(forward):
            j=int(j)
            if cut or backward[j]!=i or not near[i,j]:continue
            val=float(max(sim[i,j],0)*(1-np.linalg.norm(xy[i]-xy[j])));edges.append([t,i,j,val])
            edge_score[t,i]=max(edge_score[t,i],val);edge_score[t+1,j]=max(edge_score[t+1,j],val)
    selected=[]
    for t in range(T):
        fixed=top(u[t],(K+1)//2);rest=top(edge_score[t],K-len(fixed),fixed);selected.append(np.r_[fixed,rest])
    selections['connection']=np.stack(selected)
    return dict(selections=selections,query_scores=aq,caption_scores=ac,consensus=u,disagreement=d,edges=np.asarray(edges,dtype=float).reshape(-1,4),cuts=np.asarray(cuts),budget_per_cell=K)
def contracts():
    import numpy as np
    assert np.array_equal(ranks([1,1,1]),[.5,.5,.5]);assert np.array_equal(top([0,1,1],2),[1,2])
    z=np.eye(16,dtype=np.float32)[None].repeat(3,0);q=z[0][:2];c=z[0][2:4];a=analyze(z,q,c,q,4,4)
    assert all(s.shape==(3,2)and all(len(set(x))==2 for x in s)for s in a['selections'].values())
    assert all(int(i)==int(j)for t,i,j,v in a['edges']);assert len(a['edges'])==32
    assert not a['cuts'].any();bad=analyze(np.stack([z[0],-z[0],z[0]]),q,c,q,4,4);assert bad['cuts'].all()and len(bad['edges'])==0
    assert np.array_equal(a['selections']['query'],a['selections']['query_swap'])
    assert not np.array_equal(a['selections']['query'],a['selections']['caption'])
    return dict(status='pass',count=8,CPU_only=True,GT_read=False,scope='ranks/ties/budgets/reciprocity/shot break/query-swap identity; synthetic only')
def main():
    import numpy as np,torch
    from transformers import AutoProcessor
    from scripts import run_stvg_native_instance_teacher_storage002 as loader
    from scripts import stvg_table1_temporal_adapter_v1 as adapter
    from scripts.stvg_qwen_uniform_clock002 import apply
    from scripts.stvg_reference_clock_control_v1 import qualify
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts.stvg_instance_evidence32_v1 import guard
    verify();sys.addaudithook(guard);torch.set_num_threads(2)
    lock=read(BASE/'RESPONSE_RUNTIME.json')
    for rel,h in lock['pins'].items():assert sha(ROOT/rel)==h,rel
    loader.verify_local_model_receipt();loader.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
    _,model,_,device,_,load_seconds,_=loader._load_qwen(ROOT/loader.CHECKPOINT_REL,BASE/'offload');proof=apply(model)
    processor=AutoProcessor.from_pretrained(ROOT/loader.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False);adapter.configure_original_processor(processor)
    rr=read(BASE/'ROSTER_LOCK.json')['records'];requests={r['probe_alias']:read(PRIOR/'temporal/requests'/(r['probe_alias']+'.json'))for r in rr}
    swap={}
    for ds in ('hc2','vidstg'):
        ids=[r['probe_alias']for r in rr if r['dataset']==ds]
        for i,cid in enumerate(ids):swap[cid]=ids[(i+1)%len(ids)]
    records=[];start=time.perf_counter()
    def text_embedding(text):
        ids=processor.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
        assert 0<ids.numel()<=512
        return model.get_input_embeddings()(ids)[0].float().cpu().numpy()
    for job in read(BASE/'INPUT_BARRIER.json')['jobs']:
        assert sha(ROOT/job['metadata_path'])==job['metadata_sha256']and sha(ROOT/job['caption_path'])==job['caption_sha256']
        meta=read(ROOT/job['metadata_path']);caption=read(ROOT/job['caption_path'])['raw_caption'];cid=job['case_id'];key=meta['candidate']['canonical_id'];video=np.load(ROOT/meta['input_path'],allow_pickle=False)
        assert sha(PRIOR/'temporal/requests'/(cid+'.json'))==job['request_sha256']
        assert sha(ROOT/meta['input_path'])==meta['input_sha256'];tick=time.perf_counter()
        batch,_=adapter._processor_batch(processor,adapter.caption_prompt(),video=video,fps=meta['metadata']['clip_fps'],device=device)
        actual_clock=qualify(model,batch,dict(job_id=cid+'_'+key))
        grid=batch['video_grid_thw'][0].cpu().tolist();T,H,W=map(int,grid);merge=int(model.config.vision_config.spatial_merge_size)
        assert merge==2 and T==4 and H%2==W%2==0
        with torch.inference_mode():
            feat=model.model.get_video_features(batch['pixel_values_videos'],batch['video_grid_thw'])
            z=feat.reshape(T,H//2*W//2,-1).float().cpu().numpy()
            q=text_embedding(requests[cid]['query_text']);c=text_embedding(caption)if caption.strip()else np.zeros((1,z.shape[-1]),np.float32)
            s=text_embedding(requests[swap[cid]]['query_text'])
        assert np.isfinite(z).all()and z.shape[0]*z.shape[1]==T*H*W//4
        a=analyze(z,q,c,s,H//2,W//2)
        p=BASE/'responses'/cid/(key+'.npz');p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
        np.savez_compressed(p,features=z,query_embedding=q,caption_embedding=c,swapped_embedding=s,**{k:v for k,v in a.items()if k!='selections'},**{'selection_'+k:v for k,v in a['selections'].items()})
        info=dict(case_id=cid,candidate_id=key,path=str(p.relative_to(ROOT)),sha256=sha(p),grid=grid,merged_grid=[T,H//2,W//2],metadata_sha256=job['metadata_sha256'],caption_response_sha256=job['caption_sha256'],caption_empty=not bool(caption.strip()),query_swap_case=swap[cid],actual_time_encoding=actual_clock,wall_seconds=time.perf_counter()-tick,GT_read=False)
        put(p.with_suffix('.json'),info);records.append(info)
        from scripts.stvg_native_instance_common_v1 import write
        write(BASE/'STATUS.json',dict(status='running',stage='native_response_only',done=len(records),total=len(read(BASE/'INPUT_BARRIER.json')['jobs']),GT_read=False),mutable=True)
        del batch,feat,z,q,c,s,a;gc.collect();torch.cuda.empty_cache()
    put(BASE/'RESPONSE_GLOBAL_BARRIER.json',dict(status='complete',count=16,records=records,GT_read=False,model_load_seconds=load_seconds,wall_seconds=time.perf_counter()-start,clock_proof=proof,new_decoder_generation_calls=0,visual_encoder_passes=len(records)))
if __name__=='__main__':
    if sys.argv[1:] == ['contracts']:print(json.dumps(contracts()))
    else:main()
