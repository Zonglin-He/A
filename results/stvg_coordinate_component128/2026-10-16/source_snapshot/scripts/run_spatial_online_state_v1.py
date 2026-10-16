"""Finite online streams: seal calibration, lock LR, then continue held-out IDs."""
import argparse, gc, hashlib, sys, time
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, load, save, sha
OUT = ROOT / 'artifacts/spatial_online_state_v1'
CODE = ['scripts/run_spatial_online_state_v1.py', 'scripts/analyze_spatial_online_state_v1.py',
        'vg_tta/spatial_online_state_v1.py', 'tests/test_spatial_online_state_v1.py',
        'protocols/spatial_online_state_v1.md']


def hashed(s): return hashlib.sha256(s.encode()).hexdigest()


def prepare():
    if (OUT/'LOCK.json').exists(): return verify()
    p = read(ROOT/'artifacts/spatial_reference_absorption_v1/LOCK.json')
    streams = {}
    for c in p['configs']:
        rr = sorted([r for r in p['rows'] if r['cohort'] == c], key=lambda r: hashed('cal|' + r['source']))
        assert len(rr) == len({r['source'] for r in rr}) == 32
        streams[c] = {str(seed): [r['key'] for subset in [rr[:8], rr[8:]]
            for r in sorted(subset, key=lambda r: hashed(str(seed)+'|'+r['source']))]
            for seed in [20260920, 20260921, 20260922]}
    write(OUT/'LOCK.json', dict(rows=p['rows'], streams=streams, calibration_count=8,
        multipliers=[1., .25, .0625], arms=['E','O-all','O-split'],
        labels=p['labels'], labels_sha256=p['labels_sha256'], production_pins=p['production_pins'],
        code={f:sha(ROOT/f) for f in CODE}, source_exposure='historical, not untouched',
        temporal='fixed formal episodic C+D output', created=time.time(), GT_online=False))
    return verify()


def verify():
    p = read(OUT/'LOCK.json')
    code = dict(p['code'])
    if (OUT/'DEVICE_FIX.json').exists():
        amendment=read(OUT/'DEVICE_FIX.json')
        assert amendment['original_lock_sha256']==sha(OUT/'LOCK.json')
        code.update(amendment['code'])
    for f,h in {**p['production_pins'], **code}.items(): assert sha(ROOT/f)==h, f
    return p


def dest(cohort, seed, arm, mult, pos):
    return OUT/'streams'/cohort/str(seed)/arm/str(float(mult))/f'{pos:03d}.pt'


def run(phase):
    import torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import config
    from methods.decota_final_simplified_v1.replay import SpatialReplay
    from methods.decota_final_simplified_v1.optim import fit_spatial
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from methods.decota_final_simplified_v1.backbone import full_prediction, make_batch, query_subject
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.spatial_three_rounds_v1 import temporal_state
    from vg_tta.spatial_online_state_v1 import arrival
    p=verify(); rows={r['key']:r for r in p['rows']}
    if phase=='evaluation':
        chosen=read(OUT/'CALIBRATION.json')['selected']
        assert read(OUT/'CALIBRATION_BARRIER.json')['GT_online'] is False
    receipts={}; begin=time.time(); torch.set_num_threads(4); torch.manual_seed(20260920)
    guard=lease(); model=None
    try:
        for cohort, streams in p['streams'].items():
            model=model_load(cohort); modelhash=state_hash(model.state_dict()); cfg=config(cohort)
            for seed, keys in streams.items():
                active=[(a,m) for a in p['arms'] for m in (p['multipliers'] if phase=='calibration'
                    else sorted(set([1.,chosen[cohort][a]]), reverse=True))]
                for pos in (range(8) if phase=='calibration' else range(8,32)):
                    r=rows[keys[pos]]
                    for f,h in [('path','sha256'),('cache','cache_sha256')]: assert sha(r[f])==r[h]
                    old=load(r['path']); cache=load(r['cache']); ids=cache['frame_ids']
                    replay=SpatialReplay(model,detached(cache['views'],'cuda'),len(ids)); source=detached(replay.initial)
                    assert torch.equal(replay.zero['boxes'].cpu(),old['native_boxes'])
                    assert all(torch.equal(v.cpu(),old['spatial']['initial_state'][k]) for k,v in source.items())
                    anchors=old['spatial']['anchors']; assert old['spatial']['planned']==4
                    first_outputs={}
                    for arm,mult in active:
                        path=dest(cohort,seed,arm,mult,pos)
                        if path.with_suffix('.json').exists():
                            assert sha(path)==read(path.with_suffix('.json'))['sha256']
                            receipts[str(path)]=sha(path)
                            if pos==0: first_outputs.setdefault(mult,load(path)['after'])
                            continue
                        previous=None; previous_hash=None
                        if pos:
                            prev=dest(cohort,seed,arm,mult,pos-1)
                            previous_hash=sha(prev); assert previous_hash==read(prev.with_suffix('.json'))['sha256']
                            previous=load(prev)['state']
                        initial=arrival(source,previous,arm); replay.initial=detached(initial)
                        replay.restore(initial)
                        with torch.no_grad(): before=detached(replay.values())
                        t=time.perf_counter()
                        fitted=fit_spatial(replay,before,anchors,replace(cfg,spatial_lr=cfg.spatial_lr*mult))
                        torch.cuda.synchronize(); elapsed=time.perf_counter()-t
                        after=fitted['final']['boxes'].cpu(); selected=detached(fitted['state'],'cpu')
                        audits={}
                        if arm=='E' and mult==1:
                            assert torch.equal(after,old['predictions']['Full_DeCoTA']['boxes'])
                            assert all(torch.equal(v,old['spatial']['state'][k]) for k,v in selected.items())
                            audits['episodic_exact']=True
                        if pos==0:
                            if mult in first_outputs: assert torch.equal(after,first_outputs[mult])
                            first_outputs[mult]=after; audits['first_item_agreement']=True
                        if not anchors:
                            assert torch.equal(after,before['boxes'].cpu())
                            assert all(torch.equal(v,initial[k].cpu()) for k,v in selected.items())
                            audits['empty_preserves_history']=True
                        # Real inherited-state reference and whole-backbone verification.
                        if phase=='calibration' and seed=='20260920' and pos==1 and mult==1 and arm!='E':
                            for tag,st,expectedbox in [('before',initial,before['boxes'].cpu()),('after',selected,after)]:
                                replay.restore(st)
                                with torch.no_grad(): actual=replay.reference_values()
                                assert torch.equal(actual['boxes'].cpu(),expectedbox), tag
                                assert all(torch.equal(a,b) for a,b in zip(actual['logits'],replay.zero['logits']))
                                audits[tag+'_reference_exact']=True
                            frames,fullids=decode(old['input']); batch=make_batch(frames,fullids,old['input'],model)
                            state={**selected,**temporal_state(old,cfg.eta)}
                            expected=dict(boxes=after,logits=old['predictions']['Full_DeCoTA']['logits'])
                            with query_subject(model,batch,old['parses']['subject']):
                                full=full_prediction(model,batch,fullids,cache['records'],state,expected)
                            assert full['indices']==old['predictions']['Full_DeCoTA']['indices']
                            audits['full_model_exact']=True; del frames,batch
                        replay.restore(source)
                        with torch.no_grad(): reset=replay.values()
                        assert torch.equal(reset['boxes'],replay.zero['boxes'])
                        x=dict(key=r['key'],cohort=cohort,source=r['source'],seed=int(seed),position=pos,
                            phase=phase,arm=arm,multiplier=mult,lr=cfg.spatial_lr*mult,frame_ids=ids,
                            indices=old['predictions']['Full_DeCoTA']['indices'],anchors=anchors,
                            observed=old['actual_observation_positions'],native=old['native_boxes'],
                            before=before['boxes'].cpu(),after=after,state=selected,initial=detached(initial,'cpu'),
                            initial_hash=state_hash(initial),state_hash=state_hash(selected),previous_file_sha256=previous_hash,
                            selected_step=fitted['selected_step'],losses=fitted['losses'],failure=fitted['failure'],
                            parameter_changed=fitted['parameter_changed'],skipped=fitted['skipped'],
                            backwards=fitted['backwards'],seconds=elapsed,audit=audits,GT_online=False,
                            optimizer_reset=True,source_restored=True)
                        save(path,x);write(path.with_suffix('.json'),dict(sha256=sha(path),lock_sha256=sha(OUT/'LOCK.json')))
                        receipts[str(path)]=sha(path)
                        status(OUT/'STATUS.json',dict(status='running',phase=phase,done=len(receipts),
                            cohort=cohort,seed=seed,position=pos,seconds=time.time()-begin))
                        del fitted,before,after,selected,x
                    print(phase,cohort,seed,pos,len(receipts),round(time.time()-begin,1),flush=True)
                    del replay,cache,old; gc.collect()
            assert state_hash(model.state_dict())==modelhash
            del model;model=None;gc.collect();torch.cuda.empty_cache()
        verify();write(OUT/(phase.upper()+'_BARRIER.json'),dict(files=receipts,created=time.time(),GT_online=False))
        status(OUT/'STATUS.json',dict(status=phase+'_complete',done=len(receipts),seconds=time.time()-begin))
    except BaseException as e:
        status(OUT/'STATUS.json',dict(status='failed',phase=phase,type=type(e).__name__,error=str(e)))
        raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache();guard.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','calibration','evaluation']);a=ap.parse_args()
    prepare() if a.phase=='prepare' else run(a.phase)
