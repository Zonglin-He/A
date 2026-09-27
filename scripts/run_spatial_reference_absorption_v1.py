"""Two sequential sealed phases: one-direction probes, then O0–O3 fits."""
import argparse,gc,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
OUT=ROOT/'artifacts/spatial_reference_absorption_v1'
PARENT=ROOT/'artifacts/spatial_anchor_transfer_v1'
CODE=['scripts/run_spatial_reference_absorption_v1.py','vg_tta/spatial_reference_absorption_v1.py',
      'tests/test_spatial_reference_absorption_v1.py','protocols/spatial_reference_absorption_v1.md']


def prepare():
    if (OUT/'LOCK.json').exists():return verify()
    from scripts.run_spatial_anchor_transfer_v1 import verify as parent_verify
    p=parent_verify();chosen=read(PARENT/'INTERVENTION_LOCK.json')
    write(OUT/'LOCK.json',dict(rows=p['rows'],configs=p['configs'],labels=p['labels'],labels_sha256=p['labels_sha256'],
        selected=chosen['rows'],selection_sha256=sha(PARENT/'INTERVENTION_LOCK.json'),production_pins=p['production_pins'],
        code={f:sha(ROOT/f) for f in CODE},loss_tol=1e-9,output_tol=1e-7,created=time.time(),max_seconds=3600,
        source_exposure='historical development, single panel GT-informed selection',GT_online=False))
    return verify()


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in {**p['production_pins'],**p['code']}.items():assert sha(ROOT/f)==h,f
    return p


def run(phase):
    import torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import config
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from methods.decota_final_simplified_v1.backbone import full_prediction,make_batch,query_subject
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_reference_absorption_v1 import fit,probe
    from vg_tta.spatial_three_rounds_v1 import temporal_state
    import scripts.run_spatial_query_negative_oracle_v1 as pauser
    import scripts.run_actionness_spatial_attribution_v1 as resumer
    p=verify();selected={r['key']:r for r in p['selected']};receipts={};guard=None;snapshot=None;model=None
    if phase=='main':assert (OUT/'PROBE_BARRIER.json').exists() and (OUT/'PROBE_RESULTS.json').exists()
    begin=time.time();torch.set_num_threads(4);torch.manual_seed(20260919)
    try:
        pauser.OUT=OUT;snapshot=pauser.pause();guard=lease()
        for cohort in p['configs']:
            model=model_load(cohort);mh=state_hash(model.state_dict());full_done=False;lr0done=False
            rr=[r for r in p['rows'] if r['cohort']==cohort and (phase=='main' or r['key'] in selected)]
            for r in rr:
                dest=OUT/phase/f'{r["key"].replace(":","_")}.pt'
                if dest.with_suffix('.json').exists():
                    assert sha(dest)==read(dest.with_suffix('.json'))['sha256'];receipts[str(dest)]=sha(dest);continue
                assert time.time()-begin<p['max_seconds']
                for f,h in [('path','sha256'),('single','single_sha256'),('cache','cache_sha256')]:assert sha(r[f])==r[h]
                old=load(r['path']);single=load(r['single']);c=load(r['cache']);ids=c['frame_ids'];torch.cuda.synchronize();rs=time.perf_counter()
                replay=SpatialReplay(model,detached(c['views'],'cuda'),len(ids));torch.cuda.synchronize();setup=time.perf_counter()-rs
                assert torch.equal(replay.zero['boxes'].cpu(),old['native_boxes'])
                assert all(torch.equal(v.cpu(),old['spatial']['initial_state'][k]) for k,v in replay.initial.items())
                cfg=config(cohort);audit=dict(native_exact=True,reset_exact=False,full_reinsertion=False,lr0=False)
                result=dict(key=r['key'],cohort=cohort,source=r['source'],frame_ids=ids,
                    indices=old['predictions']['Full_DeCoTA']['indices'],actual_observations=old['actual_observation_positions'],
                    replay_setup_seconds=setup,native_boxes=replay.zero['boxes'].cpu(),GT_online=False,audit=audit)
                if phase=='probe':
                    ai=selected[r['key']]['anchor_index'];anchor=single['fits'][ai]['anchor']
                    result.update(anchor=anchor,anchor_index=ai,probe=probe(replay,anchor,cfg.spatial_lr))
                else:
                    result.update(K4={},single={})
                    anchors=old['spatial']['anchors']
                    assert old['spatial']['planned']==4
                    for arm in ['O0','O1','O2','O3']:
                        fit0=fit(replay,anchors,cfg.spatial_lr,arm,planned=4)
                        if arm=='O0':
                            assert torch.equal(fit0['boxes'],old['predictions']['Full_DeCoTA']['boxes'])
                            assert all(torch.equal(v,old['spatial']['state'][k]) for k,v in fit0['state'].items())
                            audit['O0_exact']=True
                        result['K4'][arm]=fit0
                    if r['key'] in selected:
                        ai=selected[r['key']]['anchor_index'];anchor=single['fits'][ai]['anchor']
                        for arm in ['O0','O1','O2','O3']:
                            fit0=fit(replay,[anchor],cfg.spatial_lr,arm,planned=1)
                            if arm=='O0':
                                assert torch.equal(fit0['boxes'],single['fits'][ai]['prediction']['boxes'])
                                assert fit0['selected_step']==single['fits'][ai]['selected_step']
                            result['single'][arm]=fit0
                    if not lr0done and anchors:
                        z=fit(replay,anchors,0,'O3',steps=1)
                        assert z['accepted_updates']==0 and torch.equal(z['boxes'],replay.zero['boxes'].cpu())
                        assert not z['final_optimizer']['state'];audit['lr0']=True;lr0done=True
                    if not full_done and anchors:
                        frames,fullids=decode(old['input']);batch=make_batch(frames,fullids,old['input'],model)
                        state={**result['K4']['O3']['state'],**temporal_state(old,cfg.eta)}
                        expected=dict(boxes=result['K4']['O3']['boxes'],logits=old['predictions']['Full_DeCoTA']['logits'])
                        with query_subject(model,batch,old['parses']['subject']):
                            full=full_prediction(model,batch,fullids,c['records'],state,expected)
                        assert full['indices']==result['indices'];audit['full_reinsertion']=True;full_done=True
                        del frames,batch
                replay.restore(replay.initial)
                with torch.no_grad():v=replay.values()
                assert torch.equal(v['boxes'],replay.zero['boxes']);audit['reset_exact']=True
                assert all(v.grad is None and not v.requires_grad for v in model.parameters())
                save(dest,result);write(dest.with_suffix('.json'),dict(sha256=sha(dest),lock_sha256=sha(OUT/'LOCK.json')))
                receipts[str(dest)]=sha(dest);status(OUT/f'{phase.upper()}_STATUS.json',dict(status='running',done=len(receipts),seconds=time.time()-begin))
                print(phase,len(receipts),r['key'],flush=True);del replay,result,old,single,c,v
            assert state_hash(model.state_dict())==mh;del model;model=None;gc.collect();torch.cuda.empty_cache()
        assert len(receipts)==(8 if phase=='probe' else 64);verify()
        write(OUT/f'{phase.upper()}_BARRIER.json',dict(files=receipts,created=time.time(),GT_online=False))
        status(OUT/f'{phase.upper()}_STATUS.json',dict(status='completed',done=len(receipts),seconds=time.time()-begin))
    except BaseException as e:
        write(OUT/f'{phase.upper()}_FAILURE_{int(time.time())}.json',dict(type=type(e).__name__,error=str(e)));raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
        if guard:guard.close()
        resumer.OUT=OUT;resumer.resume(snapshot)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','probe','main']);x=a.parse_args()
    prepare() if x.command=='prepare' else run(x.command)
