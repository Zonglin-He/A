"""Finite source-reset episodes and fixed-write functional targets, no GT."""
import os, sys, time, functools, traceback, hashlib, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from scripts.tastvg_correction_scope_common_v1 import *

def run(ds, stage):
    import torch, numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction, source_capture
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict, central_state
    from vg_tta.tastvg_negative_evidence_v1 import block_only_state
    from vg_tta.tastvg_selected_rollout_v1 import rollout_states
    verify();budget();install_clean_loader();sys.addaudithook(guard);lh=lease();tick=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    cb=read(POOL/ds/'CAPTURE_BARRIER.json');mh=state_hash(model.state_dict())
    assert mh==cb['checkpoint_state_sha256']
    center=central_state(model);states,spec,basis=rollout_states(center,.05,4)
    deltas=[{n:v-center[n] for n,v in z.items()} for z in states]
    p=plan(ds);cohort=read(BASE/'COHORT.json');cells=[c for c in cohort['cells'] if c['dataset']==ds]
    counts=dict(suffix_replays=0,backward_calls=0,new_backbone_calls=0,new_expert_calls=0,text_only_forwards=0)
    @functools.lru_cache(maxsize=8)
    def data_cpu(parent,cond): return capture(ds,parent,cond)[0]
    def data_at(parent,cond): return device_tree(data_cpu(parent,cond),'cuda')
    def pred(data,state):
        counts['suffix_replays']+=1
        return compact_prediction(predict(model,data,device_tree(state,'cuda'))[2])
    def progress(done,total):
        status(BASE/ds/(stage+'_STATUS.json'),dict(status='running',done=done,total=total,
            worker_pid=os.getpid(),GT_read=False,time=time.time()))
        print(stage.upper(),ds,done,total,flush=True)
    smoke={}
    if stage=='prepare':
        # Text-only contextual vectors; no labels and no image forward.
        vectors=[]
        with torch.no_grad():
            for row in p['rows']:
                enc=model.text_encoder;tok=enc.tokenizer([row['input']['caption']],return_tensors='pt').to('cuda')
                raw=enc.body(**tok).last_hidden_state[0];keep=tok.attention_mask[0].bool()
                keep[0]=False;keep[int(tok.attention_mask[0].sum())-1]=False
                z=raw[keep].double().mean(0);vectors.append((z/z.norm()).cpu());counts['text_only_forwards']+=1
        z=torch.stack(vectors);sim=(z@z.T).numpy()
        commit(BASE/ds/'TEXT_CONTEXT.pt',dict(vectors=z,cosine=sim,GT_read=False,checkpoint_state_sha256=mh))
        target_rows=[];cycle=p['conditions'][1:]
        for c in cells:
            if not c['scheduled']: continue
            seq=p['splits'][c['split']]['orders'][c['order']]
            future=[j for j in range(c['arrival']+1,len(seq)) if j%4!=0
                    and p['rows'][seq[j]]['input']['video_sha256']!=p['rows'][c['parent']]['input']['video_sha256']]
            assert future
            near=min(future,key=lambda j:(-sim[c['parent'],seq[j]],j))
            far=min(future,key=lambda j:(sim[c['parent'],seq[j]],j))
            chosen=min(future,key=lambda j:hashlib.sha256((key(c)+'|'+p['rows'][seq[j]]['key']).encode()).hexdigest())
            cross=cycle[0] if c['condition']=='clean' else cycle[(cycle.index(c['condition'])+1)%len(cycle)]
            targets=[dict(parent=c['parent'],condition=c['condition'],arrival=c['arrival'],roles=['self'],alternative=False)]
            for j in future:
                roles=['different_video_same_corruption']
                if j==near:roles.append('semantic_near')
                if j==far:roles.append('semantic_far')
                if j==chosen:roles.append('matched_cross_baseline')
                targets.append(dict(parent=seq[j],condition=c['condition'],arrival=j,roles=roles,
                    alternative=False,cosine=float(sim[c['parent'],seq[j]])))
            targets.append(dict(parent=seq[chosen],condition=cross,arrival=chosen,roles=['different_corruption'],
                alternative=False,cosine=float(sim[c['parent'],seq[chosen]])))
            if ds=='vidstg': targets.append(dict(parent=c['parent'],condition=c['condition'],arrival=None,
                roles=['same_video_other_query'],alternative=True))
            target_rows.append(dict(cell=c,targets=targets))
        write(BASE/ds/'TARGET_LOCK.json',dict(rows=target_rows,GT_read=False,time=time.time(),
            context_sha256=sha(BASE/ds/'TEXT_CONTEXT.pt'),rules='all eligible future nonexpert; near/far; paired hash cross corruption'))
        if ds=='vidstg':
            from vg_tta.exact_frame_decode_audit_v2 import decode
            from scripts.run_tastvg_full_b1_experts_v1 import observation
            jobs=[(int(parent),cond) for parent in cohort['alternatives'] for cond in p['conditions']]
            for done,(parent,cond) in enumerate(jobs,1):
                budget();a=cohort['alternatives'][str(parent)];row={**a['row'],'parses':dict(subject=a['subject'])}
                frames,ids=decode(row['input']);assert ids==row['frame_ids']
                shifted,pixel,_=observation(row,cond,frames)
                d=source_capture(model,shifted,row,a['subject'])
                d.update(pixel_sha256=pixel,checkpoint_state_sha256=mh)
                if cond=='clean' and done<8:
                    out=pred(device_tree(d,'cuda'),center)
                    assert torch.equal(out['boxes'],d['prediction']['boxes']) and out['indices']==d['prediction']['indices']
                    smoke[str(parent)]=dict(source_suffix_exact=True)
                commit(alt_path(parent,cond),dict(data=d,row=row,original_parent=parent,GT_read=False))
                counts['new_backbone_calls']+=2;progress(done,len(jobs));del frames,shifted,d;gc.collect();torch.cuda.empty_cache()
        else:progress(1,1)
        total=len(cohort['alternatives'])*6 if ds=='vidstg' else 1
    elif stage=='episodic':
        from vg_tta.tastvg_event_support_v1 import OnlineMethod
        from scripts.tastvg_correction_views_common_v1 import OLD,expert as ep
        method=OnlineMethod(model,deltas,**BUNDLES[ds],student_temperature=1.,basis=basis,
            **read(OLD/ds/'A/REQUEST.json')['method'])
        source=detached(method.actor.initial,'cpu');source_sha=state_hash(source)
        commit(BASE/ds/'SOURCE_STATE.pt',dict(state=source,GT_read=False))
        jobs=sorted({(c['parent'],c['condition']) for c in cells if c['scheduled']})
        for done,(parent,cond) in enumerate(jobs,1):
            budget();method.reset();assert state_hash(method.actor.state())==source_sha
            data=data_at(parent,cond);pixel=data['pixel_sha256']
            def provider(stage):return dict(**ep(ds,stage,parent,cond,pixel)[0],pixel_sha256=pixel)
            calls=[]
            def fetched(stage):calls.append(stage);return provider(stage)
            result,_=method.arrive(data,True,lambda:fetched('temporal'),lambda:fetched('spatial'))
            assert calls==['temporal','spatial'] and result['pre_state_sha256']==source_sha
            assert torch.equal(result['prediction']['boxes'],data['prediction']['boxes'].cpu())
            assert result['prediction']['indices']==data['prediction']['indices']
            assert result['compute']['spatial_provider_calls']==1
            counts['suffix_replays']+=result['compute']['native_replays'];counts['backward_calls']+=result['compute']['backward_calls']
            # Full SGD and candidate trace retained privately; current Fast time remains pre-update.
            result['after_fast_prediction']=dict(compact_prediction(result['post_prediction']),
                indices=result['output_prediction']['indices'],physical_interval=result['output_prediction']['physical_interval'])
            commit(episode_path(ds,parent,cond),dict(parent=parent,condition=cond,result=result,
                 source_state_sha256=source_sha,full_reset=True,GT_read=False))
            method.reset();assert state_hash(method.actor.state())==source_sha
            progress(done,len(jobs));del data,result;torch.cuda.empty_cache()
        method.close();total=len(jobs)
        smoke.update(full_reset_all=True,source_prediction_all_exact=True)
    else:
        assert stage=='matrix';table=read(BASE/ds/'TARGET_LOCK.json')['rows']
        for done,item in enumerate(table,1):
            budget();c=item['cell'];x=oldchecked(local_payload_path(c));arm=x['arms']['rank_native']
            pre=x['pre_state'];post=arm['post_state'];outs=[]
            assert state_hash(pre)==c['pre_sha'] and state_hash(post)==c['post_sha']
            states={b:post if b=='full' else block_only_state(pre,post,b) for b in BLOCKS}
            for t in item['targets']:
                data=device_tree(checked(alt_path(t['parent'],t['condition']))['data'],'cuda') if t['alternative'] else data_at(t['parent'],t['condition'])
                before=pred(data,pre);after={b:pred(data,z) for b,z in states.items()}
                if 'self' in t['roles']:
                    assert torch.equal(before['boxes'],x['pre_prediction']['boxes'])
                    assert torch.equal(after['full']['boxes'],arm['post_prediction']['boxes'])
                    for b in BLOCKS[:-1]:assert torch.equal(after[b]['boxes'],arm['block_counterfactual_predictions'][b]['boxes'])
                outs.append(dict(target=t,before=before,after=after,source_interval=data['prediction']['physical_interval'],
                    pixel_sha256=data['pixel_sha256'],state_hashes={b:state_hash(z) for b,z in states.items()}))
                del data
            commit(matrix_path(c),dict(cell=c,targets=outs,donor_pre_sha=state_hash(pre),donor_post_sha=state_hash(post),
                donor_receipt_sha256=sha(Path(str(local_payload_path(c))+'.pt')),GT_read=False,parameter_updates=0))
            progress(done,len(table));del x,outs;torch.cuda.empty_cache()
        total=len(table);smoke['self_all_blocks_exact']=True
    assert state_hash(model.state_dict())==mh;verify()
    write(BASE/ds/(stage+'_RESOURCES.json'),dict(**counts,worker_wall_seconds=time.monotonic()-tick,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),wall_includes_loading_IO=True,GT_read=False))
    write(BASE/ds/(stage+'_SMOKE.json'),dict(status='pass',checks=smoke,GT_read=False,time=time.time()))
    status(BASE/ds/(stage+'_STATUS.json'),dict(status='completed',done=total,total=total,GT_read=False,time=time.time()));lh.close()

if __name__=='__main__':
    try:run(*sys.argv[1:])
    except BaseException as e:
        write(BASE/f'FAILURE_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
