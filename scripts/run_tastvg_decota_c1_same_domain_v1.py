"""Matched C1/Scale06 evaluation, with label-free reuse and finite GPU stages."""
import os, sys, time, gc, copy, traceback, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_decota_c1_common_v1 import *


def prepare():
    import torch
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser, visual_query, parse_context
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT, EXPERT_SHA256
    from vg_tta.tastvg_decota_c1_same_domain_v1 import configuration
    torch.set_num_threads(4)
    if (BASE/'RUNTIME_LOCK.json').exists(): return verify()
    inputs = {}; plans = {}
    parser = QuerySubjectParser(ROOT/'.cache/stanza')
    for ds in DATASETS:
        p = read(VIEW/ds/'PLAN.json'); old = read(POOL/ds/'PLAN.json')
        lookup = {r['key']:i for i, r in enumerate(old['rows'])}
        assert len(p['rows']) == len({r['source'] for r in p['rows']}) == 48
        rows = []
        for r in p['rows']:
            parent = lookup[r['key']]; assert r['input'] == old['rows'][parent]['input']
            assert sha(r['input']['video_path']) == r['input']['video_sha256']
            q = r['input']['caption']
            parses = dict(old=visual_query(parser, q), context=parse_context(parser, q),
                          subject=read(POOL/ds/'subjects'/f'{parent:05}.json')['parses']['subject'])
            rows.append({**r, 'pool_parent':parent, 'parses':parses})
            for cond in p['conditions']:
                _, rc = cache(ds, parent, cond, content=False)
                f = POOL/ds/'capture'/cond/f'{parent:05}.json'
                inputs[str(f.relative_to(ROOT))] = sha(f)
        cfg = configuration(ds); assert sha(ROOT/cfg.checkpoint) == cfg.checkpoint_sha256
        plans[ds] = {**p, 'rows':rows, 'configuration':cfg.to_dict(), 'sources':48, 'arrivals':576,
            'checkpoint_state_sha256':read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256']}
        write(BASE/ds/'PLAN.json', plans[ds])
        for f in [VIEW/ds/'PLAN.json', POOL/ds/'PLAN.json', POOL/ds/'CAPTURE_BARRIER.json', BASE/ds/'PLAN.json']:
            inputs[str(f.relative_to(ROOT))] = sha(f)
    weight = ROOT/EXPERT_SNAPSHOT/'model.safetensors'
    assert sha(weight) == EXPERT_SHA256
    own = ['protocols/tastvg_decota_c1_same_domain_v1.md', 'vg_tta/tastvg_decota_c1_same_domain_v1.py',
        'scripts/tastvg_decota_c1_common_v1.py', 'scripts/run_tastvg_decota_c1_same_domain_v1.py',
        'vg_tta/c1_luna_tricks_v1.py', 'vg_tta/c1_enabling_tricks_v1.py', 'vg_tta/spatial_online_state_v1.py',
        'vg_tta/spatial_consolidation_v1.py', 'methods/C1_FINAL_RESEARCH_CONFIG.json',
        'methods/C1_TEMPORAL_RESEARCH_STATUS.json', 'vg_tta/tastvg_evidence_capture_v1.py',
        'scripts/run_tastvg_evidence_vulnerability_v2.py', 'scripts/run_spatial_regression_alignment_v1.py',
        'scripts/run_spatial_ssl_gpu_v1.py', 'scripts/run_tastvg_full_b1_experts_v1.py',
        'vg_tta/tastvg_deployment_corruption_v2.py', 'vg_tta/exact_frame_decode_audit_v2.py',
        'vg_tta/tastvg_paper48_hc2_decode_v1.py']
    own += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    pin = read(ROOT/'methods/C1_FINAL_RESEARCH_CONFIG.json')['code_pins']
    for f, h in pin.items(): assert sha(f) == h
    write(BASE/'RUNTIME_LOCK.json', dict(version='tastvg_decota_c1_same_domain_v1', time=time.time(),
        pins={f:sha(ROOT/f) for f in own}, inputs=inputs,
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        expert_checkpoint_sha256=EXPERT_SHA256, configurations={d:configuration(d).to_dict() for d in DATASETS},
        total_arrivals=1152, unique_inputs=576, new_DINO_forward_cap=2304, no_GT_prediction=True,
        historical_exposure=True, no_parameter_search=True, no_total_deadline=True))
    status(BASE/'STATUS.json', dict(status='locked_pending_smoke', predictions=0, GT_read=False))
    archive('固定方法/名单/双序/采样/cache绑定已锁，尚未预测')
    print('PREPARED',1152,'arrivals',576,'unique inputs',flush=True)


def start_gpu():
    import numpy as np, torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    install_clean_loader()
    sys.addaudithook(guard)
    torch.set_num_threads(4); torch.manual_seed(20260920); np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from scripts.run_final_simplification_v1 import lease
    return lease()


def data_input(ds, row, cond):
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    data, rc = cache(ds, row['pool_parent'], cond)
    assert data['frame_ids'] == row['frame_ids']
    assert data['pixel_sha256'] == rc['pixel_sha256']
    return device_tree(data, 'cuda'), rc


def model_for(ds):
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from methods.decota_final_simplified_v1.tensors import state_hash
    model = model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    assert model.cfg.DATASET.NAME == ('VidSTG' if ds=='vidstg' else 'HC-STVG')
    assert state_hash(model.state_dict()) == read(BASE/ds/'PLAN.json')['checkpoint_state_sha256']
    return model


def decode_for(ds):
    # The corruption's off-grid freeze donor must use the same dataset decoder.
    from vg_tta import exact_frame_decode_audit_v2 as binding
    if ds=='hc2':
        from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
        binding.decode=decode
    return binding.decode


def temporal(model, replay, records, ds):
    from methods.decota_final_simplified_v1.replay import TemporalReplay
    from methods.decota_final_simplified_v1.optim import fit_temporal
    from vg_tta.tastvg_decota_c1_same_domain_v1 import configuration
    t = TemporalReplay(model.temp_embed, replay.temporal_inputs, replay.zero)
    z = fit_temporal(t, records, configuration(ds), trace=True)
    assert z['failure'] is None and z['selected_step']==5 and z['backwards']==5
    return z


def smoke():
    import torch
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_luna_tricks_v1 import fit
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.observations import SpatialExpert, observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from methods.decota_final_simplified_v1.backbone import query_subject, full_prediction
    p = verify(); lease = start_gpu(); stats=[]; tick=time.time()
    expert = SpatialExpert(ROOT/EXPERT_SNAPSHOT)
    try:
        for ds in DATASETS:
            model=model_for(ds); row=read(BASE/ds/'PLAN.json')['rows'][0]
            data, rc=data_input(ds,row,'clean'); s=NormalizedSpatialReplay(model,data)
            frames, ids=decode_for(ds)(row['input']); assert ids==row['frame_ids']
            shifted,pixel,_=observation(row,'clean',frames); assert pixel==rc['pixel_sha256']
            batch, records, raw=frozen_forward(model,shifted,row)
            assert records==data['records']
            assert torch.equal(raw.zero['boxes'],s.zero['boxes'])
            assert all(torch.equal(a,b) for a,b in zip(raw.zero['logits'],s.zero['logits']))
            assert all(torch.equal(a,b) for a,b in zip(raw.zero['actions'],s.zero['actions']))
            assert all(torch.equal(a,b) for a,b in zip(raw.temporal_inputs,s.temporal_inputs))
            ex=observations(expert,row['parses'],shifted,ids,data['prediction']['indices'],audit=True)
            ef=BASE/ds/'evidence/clean/00000.pt'
            if not ef.exists(): commit(ef,dict(expert=detached(ex,'cpu'),pixel_sha256=pixel,GT_read=False))
            z=fit(s,s.initial,ex['anchors']['single4'],ids,row['key'],'Scale06')
            old=fit(raw,raw.initial,ex['anchors']['single4'],ids,row['key'],'Scale06')
            assert z['selected_step']==old['selected_step'] and len(z['path'])==len(old['path'])
            for a,b in zip(z['path'],old['path']):
                assert a['loss']==b['loss'] and torch.equal(a['boxes'],b['boxes'])
                assert state_hash(a['state'])==state_hash(b['state'])
                if 'update' in a: assert torch.equal(a['update']['gradient'],b['update']['gradient'])
            tt=temporal(model,s,data['records'],ds)
            baseline_t=temporal(model,raw,records,ds)
            assert tt['losses']==baseline_t['losses']
            assert state_hash(tt['shrunk_state'])==state_hash(baseline_t['shrunk_state'])
            st={**detached(z['state'],'cuda'),**tt['shrunk_state']}
            expected=dict(boxes=z['final'].cuda(),logits=tt['shrunk']['logits'])
            with query_subject(model,batch,row['parses']['subject']):
                full_prediction(model,batch,ids,records,st,expected)
            stats.append(dict(dataset=ds,source_id=0,native_bitwise=True,
                C1_scale06_all_steps_bitwise=True,temporal_all_steps_bitwise=True,
                full_reinsertion_bitwise=True,steps=len(z['path'])-1,new_DINO=ex['new_DINO']))
            del model,s,raw,data,frames,shifted,old,z,tt,baseline_t,batch;gc.collect();torch.cuda.empty_cache()
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',GT_read=False,records=stats,seconds=time.time()-tick))
        status(BASE/'STATUS.json',dict(status='smoke_pass_pending_evidence',GT_read=False))
        print('SMOKE_PASS',stats,flush=True)
    finally: lease.close()


def evidence(ds):
    import torch
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.observations import SpatialExpert, observations
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    verify(); assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
    lease=start_gpu(); p=read(BASE/ds/'PLAN.json'); tick=time.time(); done=forwards=0
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT); modelhash=state_hash(expert.model.state_dict())
    try:
        decode=decode_for(ds)
        for row in p['rows']:
            frames=None
            for cond in p['conditions']:
                budget(); f=BASE/ds/'evidence'/cond/f"{row['ordinal']:05}.pt"
                if f.exists():
                    value=checked(f); done+=1;forwards+=value['expert']['new_DINO'];continue
                if frames is None: frames,ids=decode(row['input']);assert ids==row['frame_ids']
                data,rc=cache(ds,row['pool_parent'],cond)
                shifted,pixel,spec=observation(row,cond,frames);assert pixel==rc['pixel_sha256']
                ex=observations(expert,row['parses'],shifted,ids,data['prediction']['indices'],audit=True)
                value=dict(expert=detached(ex,'cpu'),pixel_sha256=pixel,spec=spec,
                    positions=data['prediction']['indices'],GT_read=False)
                commit(f,value);done+=1;forwards+=ex['new_DINO']
                status(BASE/ds/'EVIDENCE_STATUS.json',dict(status='running',done=done,total=288,
                    new_DINO=forwards,seconds=time.time()-tick,pid=os.getpid(),GT_read=False))
                print('DINO',ds,done,288,round(time.time()-tick,1),flush=True)
                del data,shifted,ex,value
            del frames;gc.collect();torch.cuda.empty_cache()
        assert done==288 and forwards<=1152 and state_hash(expert.model.state_dict())==modelhash
        ff={str(f.relative_to(BASE)):sha(f) for f in (BASE/ds/'evidence').rglob('*.pt')}
        write(BASE/ds/'EVIDENCE_BARRIER.json',dict(status='sealed',files=ff,cells=done,new_DINO=forwards,
            GT_read=False,seconds=time.time()-tick,expert_weights_unchanged=True))
        status(BASE/ds/'EVIDENCE_STATUS.json',dict(status='completed',done=done,new_DINO=forwards,seconds=time.time()-tick))
    finally: lease.close()


def online(ds):
    import torch
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    from vg_tta.c1_luna_tricks_v1 import fit
    from vg_tta.spatial_online_state_v1 import arrival, QUERY
    from vg_tta.spatial_consolidation_v1 import consolidate
    from methods.decota_final_simplified_v1.tensors import detached, state_hash
    from methods.decota_final_simplified_v1.objectives import prediction
    verify();lease=start_gpu();p=read(BASE/ds/'PLAN.json'); eb=read(BASE/ds/'EVIDENCE_BARRIER.json')
    assert eb['status']=='sealed';model=model_for(ds);mh=state_hash(model.state_dict());tick=time.time();done=backwards=0
    try:
        for split, sp in p['splits'].items():
            for cond in p['conditions']:
                for order, seq in sp['orders'].items():
                    previous=prevsha=None
                    for at,parent in enumerate(seq):
                        budget(); row=p['rows'][parent];f=BASE/ds/'online'/split/cond/order/f'{at:05}.pt'
                        if f.exists():
                            x=checked(f);assert x['previous_sha256']==prevsha
                            previous=x['committed'];prevsha=sha(f);done+=1;backwards+=x['fit']['gradient_calls'];continue
                        data,rc=data_input(ds,row,cond);s=NormalizedSpatialReplay(model,data)
                        source=detached(s.initial,'cpu');initial=arrival(s.initial,previous,'O-split')
                        assert torch.count_nonzero(initial[QUERY])==0
                        s.restore(initial)
                        with torch.no_grad(): before=detached(s.values()['boxes'],'cpu')
                        ef=BASE/ds/'evidence'/cond/f'{parent:05}.pt';assert sha(ef)==eb['files'][str(ef.relative_to(BASE))]
                        ex=checked(ef);assert ex['pixel_sha256']==rc['pixel_sha256'];anchors=ex['expert']['anchors']['single4']
                        torch.cuda.synchronize();t=time.perf_counter()
                        z=fit(s,initial,anchors,row['frame_ids'],row['key'],'Scale06')
                        torch.cuda.synchronize();fit_seconds=time.perf_counter()-t
                        committed=consolidate(detached(initial,'cpu'),z['state'],1/16)
                        assert torch.count_nonzero(committed[QUERY])==0
                        tf=BASE/ds/'temporal'/cond/f'{parent:05}.pt'
                        if tf.exists(): tt=checked(tf)
                        else:
                            tt=detached(temporal(model,s,data['records'],ds),'cpu');commit(tf,tt)
                        final=prediction(tt['shrunk']['logits'],z['final'],data['records'],row['frame_ids'])
                        value=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,parent=parent,
                            source_id=parent,key=row['key'],previous_sha256=prevsha,pixel_sha256=rc['pixel_sha256'],
                            frame_ids=row['frame_ids'],source_state=source,initial=detached(initial,'cpu'),committed=committed,
                            native=detached(data['prediction'],'cpu'),before=before,fit=z,final=final,
                            temporal_path=str(tf.relative_to(BASE)),temporal_sha256=sha(tf),
                            evidence_path=str(ef.relative_to(BASE)),evidence_sha256=sha(ef),
                            anchors=anchors,fit_seconds=fit_seconds,GT_read=False)
                        commit(f,detached(value,'cpu'));previous=committed;prevsha=sha(f);done+=1;backwards+=z['gradient_calls']
                        status(BASE/ds/'ONLINE_STATUS.json',dict(status='running',done=done,total=576,split=split,
                            condition=cond,order=order,arrival=at,backwards=backwards,seconds=time.time()-tick,pid=os.getpid(),GT_read=False))
                        print('ONLINE',ds,done,576,round(time.time()-tick,1),flush=True)
                        del data,s,z,value,tt,ex;gc.collect()
                    torch.cuda.empty_cache()
        assert done==576 and state_hash(model.state_dict())==mh
        files={str(f.relative_to(BASE)):sha(f) for folder in ['online','temporal'] for f in (BASE/ds/folder).rglob('*.pt')}
        write(BASE/ds/'PREDICTION_BARRIER.json',dict(status='sealed',arrivals=done,files=files,GT_read=False,
            seconds=time.time()-tick,spatial_backwards=backwards,source_weights_unchanged=True,
            peak_memory_bytes=torch.cuda.max_memory_allocated()))
        status(BASE/ds/'ONLINE_STATUS.json',dict(status='completed',done=done,total=576,backwards=backwards,seconds=time.time()-tick))
    finally: lease.close()


def seal():
    verify();files={};count=0
    for ds in DATASETS:
        b=read(BASE/ds/'PREDICTION_BARRIER.json');assert b['status']=='sealed' and not b['GT_read'];count+=b['arrivals']
        files.update(b['files']);files.update(read(BASE/ds/'EVIDENCE_BARRIER.json')['files'])
        files[str((BASE/ds/'PREDICTION_BARRIER.json').relative_to(BASE))]=sha(BASE/ds/'PREDICTION_BARRIER.json')
    assert count==1152
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',arrivals=count,files=files,GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='predictions_sealed_pending_CPU',arrivals=count,GT_read=False,time=time.time()))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','evidence','online','seal']);ap.add_argument('dataset',nargs='?');a=ap.parse_args()
    try:
        if a.stage in ['evidence','online']: globals()[a.stage](a.dataset)
        else: globals()[a.stage]()
    except BaseException as e:
        status(BASE/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,dataset=a.dataset,error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
