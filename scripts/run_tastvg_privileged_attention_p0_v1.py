"""Finite paired frozen P0. All 192 prediction pairs precede any GT scoring."""
import os, sys, time, gc, traceback, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_privileged_p0_common_v1 import *


def prepare():
    from scripts.tastvg_decota_c1_common_v1 import cache
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT, EXPERT_SHA256
    if (BASE/'RUNTIME_LOCK.json').exists(): return verify()
    inputs = {}; datasets = {}
    for ds in DATASETS:
        p = read(PARENT/ds/'PLAN.json'); selected = list(range(8)) + list(range(32, 40))
        rows = [dict(p['rows'][i], split='search' if i < 32 else 'confirm') for i in selected]
        assert len(rows) == len({r['source'] for r in rows}) == 16
        plan = dict(dataset=ds, rows=rows, conditions=p['conditions'], sources=16, cells=96,
                    configuration=p['configuration'], checkpoint_state_sha256=p['checkpoint_state_sha256'])
        write(BASE/ds/'PLAN.json', plan); datasets[ds] = dict(sources=16, cells=96)
        inputs[str((PARENT/ds/'PLAN.json').relative_to(ROOT))] = sha(PARENT/ds/'PLAN.json')
        inputs[str((BASE/ds/'PLAN.json').relative_to(ROOT))] = sha(BASE/ds/'PLAN.json')
        inputs[str((POOL/ds/'CAPTURE_BARRIER.json').relative_to(ROOT))] = sha(POOL/ds/'CAPTURE_BARRIER.json')
        for r in rows:
            assert sha(r['input']['video_path']) == r['input']['video_sha256']
            for cond in p['conditions']:
                _, receipt = cache(ds, r['pool_parent'], cond, content=False)
                f = POOL/ds/'capture'/cond/f"{r['pool_parent']:05}.json"
                inputs[str(f.relative_to(ROOT))] = sha(f)
        cfg = p['configuration']; assert sha(ROOT/cfg['checkpoint']) == cfg['checkpoint_sha256']
    dependencies = ['scripts/decota_matrix_common_v1.py', 'scripts/tastvg_decota_c1_common_v1.py',
        'scripts/run_tastvg_decota_c1_same_domain_v1.py', 'vg_tta/tastvg_decota_c1_same_domain_v1.py',
        'scripts/run_tastvg_evidence_vulnerability_v1.py', 'scripts/run_tastvg_evidence_vulnerability_v2.py',
        'scripts/run_spatial_regression_alignment_v1.py', 'scripts/run_spatial_ssl_gpu_v1.py',
        'scripts/run_tastvg_full_b1_experts_v1.py', 'vg_tta/tastvg_deployment_corruption_v2.py',
        'vg_tta/exact_frame_decode_audit_v2.py', 'vg_tta/tastvg_paper48_hc2_decode_v1.py',
        'external/TA-STVG/models/grounding_model/query_decoder.py', 'external/TA-STVG/models/grounding_model/attention.py']
    dependencies += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    assert sha(ROOT/EXPERT_SNAPSHOT/'model.safetensors') == EXPERT_SHA256
    write(BASE/'RUNTIME_LOCK.json', dict(version='tastvg_privileged_attention_p0_v1', time=time.time(),
        pins={f:sha(ROOT/f) for f in OWN+dependencies}, inputs=inputs,
        expert_checkpoint_sha256=EXPERT_SHA256, datasets=datasets, cells=192,
        CURRENT_METHOD_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        defaults=dict(threshold=.35, tau_E=1., gaussian_sigma='half_box_width_height', alpha=1., epsilon=1e-6),
        new_DINO_forward_cap=768, parameter_updates=0, historical_exposure=True, GT_used_to_select=False))
    status(BASE/'STATUS.json', dict(status='locked_pending_smoke', cells=0, GT_read=False))
    archive('名单/多候选prior/attention插入点与输入哈希已锁，尚未预测')


def frozen_replay(model, data):
    from vg_tta.tastvg_decota_c1_same_domain_v1 import NormalizedSpatialReplay
    s = NormalizedSpatialReplay(model, data)
    s.decoder.requires_grad_(False); s.head.requires_grad_(False); s.delta.requires_grad_(False)
    assert all(p.grad is None for p in s.decoder.parameters())
    return s


def acquire(expert, row, frames, ids, interval):
    import torch
    from methods.decota_final_simplified_v1.observations import uniform_positions, tensor_hash
    from vg_tta.tastvg_privileged_attention_p0_v1 import proposal_support
    phrase = row['parses']['old']; positions = uniform_positions(ids, interval, 4); records = []
    for pos in positions:
        rgb = frames[pos]; receipts = {}
        if not phrase['phrase']:
            records.append(dict(position=pos, frame_id=ids[pos], empty_phrase=True,
                raw_boxes=torch.empty(0, 4), raw_scores=torch.empty(0), support=proposal_support([], [])))
            continue
        def hook(m, args, kw):
            for k in ('pixel_values', 'pixel_mask', 'input_ids', 'attention_mask'):
                if k in kw: receipts[k] = dict(sha256=tensor_hash(kw[k]), shape=list(kw[k].shape), dtype=str(kw[k].dtype))
        h = expert.model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad(): d = expert(rgb, phrase['phrase'], phrase['entity'])
        finally: h.remove()
        raw_boxes, raw_scores = d['all_boxes'].float().cpu(), d['all_phrase_scores'].float().cpu()
        support = proposal_support(raw_boxes, raw_scores)
        records.append(dict(position=pos, frame_id=ids[pos], empty_phrase=False,
            raw_boxes=raw_boxes, raw_scores=raw_scores, support=support, inputs=receipts,
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), seconds=d['seconds'],
            text=d['text'], entity_tokens=d['entity_tokens'], phrase_tokens=d['phrase_tokens']))
    assert all(interval[0] <= x['position'] <= interval[1] for x in records)
    return dict(positions=positions, observations=records, new_DINO=sum(not x['empty_phrase'] for x in records), GT_read=False)


def predict(s, evidence):
    import torch
    from vg_tta.tastvg_privileged_attention_p0_v1 import run_attention
    from methods.decota_final_simplified_v1.tensors import state_hash
    before = state_hash(s.decoder.state_dict())
    fields = {r['position']:r['support'] for r in evidence['observations']}
    ordinary = run_attention(s, fields, privileged=False)
    assert torch.equal(ordinary['boxes'], s.zero['boxes'].cpu())
    teacher = run_attention(s, fields, privileged=True)
    assert state_hash(s.decoder.state_dict()) == before and torch.count_nonzero(s.delta) == 0
    return dict(ordinary=ordinary, privileged=teacher, decoder_weights_sha256=before,
                parameter_updates=0, backwards=0, weights_unchanged=True)


def run():
    import torch
    from scripts.run_tastvg_decota_c1_same_domain_v1 import start_gpu, data_input, model_for, decode_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    verify(); gpu = start_gpu(); started=time.time(); done=forwards=0; smokes=[]; costs={}
    expert = SpatialExpert(ROOT/EXPERT_SNAPSHOT); expert_hash=state_hash(expert.model.state_dict())
    try:
        for ds in DATASETS:
            p=read(BASE/ds/'PLAN.json'); model=model_for(ds); model_hash=state_hash(model.state_dict()); tick=time.time()
            torch.cuda.reset_peak_memory_stats(); nd=0
            for ri,row in enumerate(p['rows']):
                frames=None
                for cond in p['conditions']:
                    budget(); f=BASE/ds/'predictions'/cond/f"{row['ordinal']:05}.pt"
                    if f.exists():
                        x=checked(f);done+=1;nd+=1;forwards+=x['evidence']['new_DINO'];continue
                    if frames is None: frames,ids=decode_for(ds)(row['input']);assert ids==row['frame_ids']
                    data,rc=data_input(ds,row,cond); shifted,pixel,spec=observation(row,cond,frames)
                    assert pixel==rc['pixel_sha256']; s=frozen_replay(model,data)
                    t=time.perf_counter(); evidence=acquire(expert,row,shifted,ids,data['prediction']['indices'])
                    prediction=predict(s,evidence)
                    if ri==0 and cond=='clean':
                        from contextlib import ExitStack
                        from methods.decota_final_simplified_v1.backbone import query_subject, inserted_state, offset_batch
                        from vg_tta.tastvg_privileged_attention_p0_v1 import fields_for, AttentionCapture
                        batch,records,raw=frozen_forward(model,shifted,row)
                        assert records==data['records'] and torch.equal(raw.zero['boxes'],s.zero['boxes'])
                        assert all(torch.equal(a,b) for a,b in zip(raw.zero['logits'],s.zero['logits']))
                        fields={r['position']:r['support'] for r in evidence['observations']}; rawboxes=[]; calls=[0]
                        with query_subject(model,batch,row['parses']['subject']), inserted_state(model,s.initial):
                            for offset in (0,1):
                                def gate(m,args,kwargs): calls[0]+=1
                                gh=model.ground_decoder.decoder.register_forward_pre_hook(gate,with_kwargs=True)
                                fs=fields_for(s,fields,offset); hooks=[]
                                def add_if_final(li):
                                    tracker=AttentionCapture(li,fs,s.views[offset]['info']['fea_map_size'][0]*s.views[offset]['info']['fea_map_size'][1])
                                    def pre(m,args,kw):
                                        if calls[0]%2==0:return tracker.before(m,args,kw)
                                    return pre
                                try:
                                    for li,l in enumerate(model.ground_decoder.decoder.layers):
                                        hooks.append(l.cross_attn.register_forward_pre_hook(add_if_final(li),with_kwargs=True))
                                    vb=offset_batch(batch,offset)
                                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                                        out=model(vb['videos'],vb['texts'],vb['targets'],iteration_rate=-1)
                                    rawboxes.append(out['pred_boxes'].float())
                                    assert torch.equal(out['pred_sted'],s.zero['logits'][offset]), 'Temporal logits invariant in full privileged path'
                                finally:
                                    gh.remove()
                                    for h in hooks:h.remove()
                        actual=torch.stack([rawboxes[i%2][i//2] for i in range(len(ids))])
                        assert torch.equal(actual.cpu(),prediction['privileged']['boxes']), 'Full privileged reinsertion parity'
                        smokes.append(dict(dataset=ds,source_id=row['ordinal'],ordinary_full_bitwise=True,
                            privileged_full_bitwise=True,GT_read=False,all_parameters_frozen=True))
                        del batch,records,raw,actual
                    value=dict(dataset=ds,source_id=row['ordinal'],split=row['split'],condition=cond,
                        frame_ids=ids,native=data['prediction'],prediction=prediction,evidence=evidence,
                        pixel_sha256=pixel,spec=spec,cache_receipt_sha256=sha(POOL/ds/'capture'/cond/f"{row['pool_parent']:05}.json"),
                        seconds=time.perf_counter()-t,GT_read=False)
                    commit(f,value);done+=1;nd+=1;forwards+=evidence['new_DINO']
                    status(BASE/'STATUS.json',dict(status='running_P0',dataset=ds,done=done,total=192,
                        new_DINO=forwards,pid=os.getpid(),seconds=time.time()-started,GT_read=False))
                    print('P0',ds,nd,96,'total',done,192,'seconds',round(time.time()-started,1),flush=True)
                    del data,s,evidence,prediction,value,shifted;gc.collect()
                del frames;torch.cuda.empty_cache()
            assert nd==96 and state_hash(model.state_dict())==model_hash
            costs[ds]=dict(worker_seconds=time.time()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated(),cells=nd)
            del model;gc.collect();torch.cuda.empty_cache()
        assert done==192 and forwards<=768 and state_hash(expert.model.state_dict())==expert_hash
        write(BASE/'SMOKE_ROOT_ACCEPTANCE.json',dict(status='pass',records=smokes,GT_read=False))
        files={str(f.relative_to(BASE)):sha(f) for f in BASE.glob('*/predictions/*/*') if f.is_file()}
        assert len(files)==384
        write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',cells=192,files=files,new_DINO=forwards,
            GT_read=False,backwards=0,parameter_updates=0,costs=costs,seconds=time.time()-started))
        status(BASE/'STATUS.json',dict(status='predictions_sealed_pending_CPU',cells=done,GT_read=False))
    finally: gpu.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    try:
        prepare() if args.stage=='prepare' else run()
    except BaseException as exc:
        BASE.mkdir(parents=True,exist_ok=True)
        status(BASE/'FAILURE.json',dict(status='engineering_failed',error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        status(BASE/'STATUS.json',dict(status='engineering_failed',error=str(exc),time=time.time()))
        raise
