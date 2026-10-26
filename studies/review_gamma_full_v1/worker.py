"""One original native raw acquisition per missing query, shared over five gamma."""
from gamma_full_common import *
import argparse,traceback,gc
def predict(ds):
    import torch,numpy as np
    from stvg.runtime import setup,lease,load_source
    from stvg.native import frozen_native_forward
    from stvg.evidence import extract_visual_grids
    from stvg.representation import observe_native_grid,extract
    from stvg.visual_prefix import native_appearance_grid
    from stvg.visual_cache import VisualCache
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.teacher_bridge import evidence_key
    from gamma_cached_core_v1 import build
    p=validate();assert read(D/'preflight_receipt.json')['status']=='passed_four_actual_cached_inputs'
    assert not(D/f'{ds}_worker.json').exists();review.gt_guard();setup();lock=lease();who=identity();atomic(D/f'{ds}_worker.json',who)
    retained=read(RETAINED/ds/'sealed_retained_subset.json')['files'];legacy=read(RUN/'reuse_index.json')[ds]
    cfg=read(R/'configs/default.json')['datasets'][ds];runtime=read(RUN/'plan.json')['provider_runtime_fingerprint']
    teacher_roots=[PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache']
    model=None;parser=None;visual=VisualCache(2*2**30);files={};started=time.perf_counter();model_load=None;bridge=[]
    counts=dict(queries=0,retained_precompute=0,legacy_both_raw=0,full_Source=0,appearance_prefix=0,new_expert=0)
    try:
        for i,row in enumerate(records(ds),1):
            resources(started)
            if i%16==1:assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['limits']['private_bytes']
            key=row['id'];tick=time.perf_counter();c,parent=selected.suite().load_parent(ds,row);current,ce=checked_current(ds,key)
            dest=D/ds;out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json')
            assert not out.exists()and not cp.exists()and not rp.exists(),'No duplicate or blind recovery'
            scratch=D/'active'/ds/key;rawfile=scratch/'raw.pt';rawmeta=scratch/'raw.json';attempt=scratch/'attempt.json'
            assert not rawfile.exists()and not attempt.exists(),'Unresolved paid attempt must be preserved'
            source_seconds=0.;teacher=None;native_proof={};raw=None
            if key in retained:
                e=retained[key]
                for field,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(RETAINED/e[field])==e[h]
                compact=torch.load(RETAINED/e['compact_path'],map_location='cpu',weights_only=False,mmap=True)
                arrays=complete_arrays(compact,current)
                with np.load(RETAINED/e['path'])as z:assert all(np.array_equal(arrays[k],z[k])for k in z.files)
                cp=RETAINED/e['compact_path'];mode='retained_precompute';origin=dict(path=str(cp),sha256=sha(cp),retained_receipt_sha256=e['receipt_sha256']);counts[mode]+=1
            else:
                prior=legacy.get(key)
                if prior:
                    assert prior['input_binding']==binding(row,runtime)and sha(prior['path'])==prior['sha256']
                    old=torch.load(prior['path'],map_location='cpu',weights_only=False,mmap=True)
                    if 'spatial_inputs'in old and 'records'in old and ('native_logits'in old or 'logits'in old):
                        logs=old.get('native_logits',[z[-1]for z in old.get('logits',[])])
                        assert np.array_equal(old['Source'],c['Source'])and old['frame_ids']==c['frame_ids']
                        assert len(logs)==2 and all(torch.equal(a,b)for a,b in zip(logs,c['native_logits']))
                        spatial,valid,ids,shape,_=extract_visual_grids(old['spatial_inputs'],c['frame_ids'])
                        raw=dict(frame_ids=ids,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        if 'native_local_grid'in old:raw['native_local_grid']=old['native_local_grid']
                    del old
                if raw is not None and 'native_local_grid'in raw:
                    mode='legacy_both_raw';origin=dict(path=prior['path'],sha256=prior['sha256']);counts[mode]+=1
                else:
                    if model is None:
                        lt=time.perf_counter();atomic(D/f'{ds}_model_load_attempt.json',dict(worker=who,before_model=True))
                        try:model=load_source(cfg)
                        except BaseException:atomic(D/f'{ds}_model_load_failure.json',dict(worker=who,seconds=time.perf_counter()-lt,error=traceback.format_exc()));raise
                        model_load=dict(worker=who,seconds=time.perf_counter()-lt,loads=1,nested_in_first_query=True);atomic(D/f'{ds}_model_load.json',model_load)
                    frames,ids=decode(row['metadata'],ds);assert ids==c['frame_ids']
                    mode='appearance_prefix'if raw is not None else'full_Source'
                    atomic(attempt,dict(worker=who,before_Source_or_prefix=True,mode=mode,input_binding=binding(row,runtime),RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),GT_read=False))
                    st=time.perf_counter();torch.manual_seed(20260920);np.random.seed(20260920)
                    if raw is not None:
                        with visual.context(model,frames,ids):grid=native_appearance_grid(model,frames,ids,row['metadata'])
                        raw['native_local_grid']=grid;native_proof=dict(native_heads_inherited_input_bound=True,legacy_spatial_raw_sha256=prior['sha256'])
                    else:
                        if parser is None:parser=QuerySubjectParser(R/'checkpoints/stanza')
                        model.cuda()
                        with observe_native_grid(model)as grid,visual.context(model,frames,ids):
                            batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses={'subject':parser(row['metadata']['caption'])['subject']}))
                        assert np.array_equal(base.zero['boxes'].cpu().numpy(),c['Source'])
                        assert len(base.zero['logits'])==2 and all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],c['native_logits']))
                        assert [z['frame_ids']for z in recs]==[z['frame_ids']for z in c['records']]
                        spatial,valid,_,shape,_=extract_visual_grids(base.spatial_inputs,ids)
                        raw=dict(frame_ids=ids,native_local_grid=grid,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        native_proof=dict(actual_native_boxes_and_both_final_logits_and_records_exact=True)
                        del batch,base;model.cpu()
                    source_seconds=time.perf_counter()-st;raw['GT_read']=False;save(rawfile,raw)
                    origin=dict(path=str(rawfile),sha256=sha(rawfile),parent_capture_sha256=parent['capture_sha256'],worker=who,mode=mode,Source_seconds=source_seconds,before_teacher_or_ROI_pool=True,GT_read=False)
                    atomic(rawmeta,origin);counts[mode]+=1
                    tkey,context=evidence_key(frames[c['positions8']],row['metadata']['caption'],positions=c['positions8'],frame_ids=ids,fingerprint=runtime,dataset=ds)
                    hits=[root for root in teacher_roots if(root/(tkey+'.json')).exists()];assert hits,'Exact original teacher required; no expert fallback'
                    tp=hits[0]/(tkey+'.pt');tr=tp.with_suffix('.json');tm=read(tr)
                    assert tm['key']==tkey and tm['context']==context and tm['sha256']==sha(tp)
                    native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)
                    assert np.array_equal(native['boxes'].numpy(),np.asarray(c['geometry']['native_boxes']))and native['native_same_track_all8']
                    teacher=dict(key=tkey,payload_path=str(tp),payload_sha256=sha(tp),receipt_path=str(tr),receipt_sha256=sha(tr),actual_eight_RGB_context_exact=True)
                    del native,frames;gc.collect();torch.cuda.empty_cache()
                geo=SpatialGeometry.from_cached_geometry(c['geometry']);aux=evidence_dp(geo.for_interval([c['frame_ids'][0],c['frame_ids'][-1]+1]))['final'].numpy()
                pool,_=extract(raw,aux);assert np.array_equal(pool,c['content_features'])
                # This compatibility cost includes the historical auxiliary DP;
                # none of the five selected gamma paths uses that auxiliary DP.
                before,compact,default=build(c,raw,ds,current);arrays=complete_arrays(compact,current)
                assert all(np.array_equal(before[k],arrays[k])for k in before)
                save(cp,compact)
                native_proof.update(original_aux_ROI_pool_exact=True,default_current_bitwise=True,actual_five_E_ROI_pools=True)
            cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=complete_arrays(cold,current)
            assert set(arrays)==set(again)and all(np.array_equal(arrays[k],again[k])for k in arrays)
            da=default_arm(ds);assert np.array_equal(arrays[da],current['FullMAP'])and np.array_equal(arrays[da+'_interval'],current['FullMAP_interval'])
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
            os.replace(out.with_suffix('.tmp'),out)
            with np.load(out)as z:assert set(arrays)==set(z.files)and all(np.array_equal(arrays[k],z[k])for k in arrays)
            rec=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp)if mode=='retained_precompute'else str(cp.relative_to(D)),compact_sha256=sha(cp),parent=parent,current_prediction_sha256=ce['sha256'],input_binding=binding(row,runtime),worker=who,mode=mode,raw_origin=origin,teacher=teacher,native_proof=native_proof,Source_seconds=source_seconds,query_elapsed_wall_seconds=time.perf_counter()-tick,new_expert=0,GT_read=False,all_arrays_cold_exact=True,fingerprint=p['fingerprint'])
            atomic(rp,rec);files[key]=committed(ds,key);counts['queries']+=1
            if rawfile.exists():
                if len(bridge)<2:bridge.append(dict(query=key,raw_path=str(rawfile),raw_sha256=sha(rawfile),receipt_path=str(rp),receipt_sha256=sha(rp)))
                else:
                    assert sha(rawfile)==read(rawmeta)['sha256'];rawfile.unlink();atomic(scratch/'disposed.json',dict(raw_sha256=origin['sha256'],receipt_sha256=sha(rp),after_commit_cold_array_readback=True))
                if len(bridge)==2 and not(D/f'{ds}_root_actual_bridge.json').exists():
                    atomic(D/f'{ds}_bridge_waiting.json',dict(status='two_actual_paid_raw_commits_waiting_root_repool_check',files=bridge,worker=who,GT_read=False))
                    while not(D/f'{ds}_root_actual_bridge.json').exists():resources(started);time.sleep(10)
                    ack=read(D/f'{ds}_root_actual_bridge.json');assert ack['status']=='passed_actual_new_raw_bridge'and ack['files']==bridge
                    for b in bridge:
                        q=Path(b['raw_path']);assert q.resolve().is_relative_to((D/'active').resolve())and sha(q)==b['raw_sha256'];q.unlink();atomic(q.parent/'disposed.json',dict(raw_sha256=b['raw_sha256'],receipt_sha256=b['receipt_sha256'],after_root_actual_repool_bridge=True))
            atomic(D/f'{ds}_progress.json',dict(completed=i,total=COUNTS[ds],counts=counts,worker=who,GT_read=False,seconds=time.perf_counter()-started));print(ds,i,mode,flush=True)
            del c,compact,cold,again,arrays,raw;gc.collect()
        assert len(files)==COUNTS[ds]and counts['new_expert']==0 and counts['retained_precompute']==192
        atomic(D/f'{ds}_sealed.json',dict(queries=len(files),files=files,counts=counts,worker=who,model_load=model_load,seconds=time.perf_counter()-started,GT_read=False,fingerprint=p['fingerprint']))
        atomic(D/f'{ds}_GPU_state.json',dict(status='complete_sealed'))
    except BaseException:atomic(D/f'{ds}_GPU_state.json',dict(status='failed_preserved',worker=who,counts=counts,error=traceback.format_exc()));raise
    finally:lock.close()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);predict(a.parse_args().dataset)
