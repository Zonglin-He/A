"""Fixed common-A -> B0/B1/B2 source contrast, isolated from original source fit."""
from __future__ import annotations
import argparse, copy, fcntl, gc, math, os, random, shutil, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from scripts.desta3d_v2_aux_backflow_run import (
    BASE, OLD, V1, SEED, MODES, read, write, sha, adapter_sha256, cpu_copy,
    scan_nested_gpu_receipts, atomic_pt, empty_buffers, add_buffers, query_gradients,
    gradient_summary, take_step, model_setup, fields_for, Desta3DAdapterV2,
    make_source_optimizer, set_source_lr, source_evidence_losses)
OUT = BASE/'fit'


def register():
    assert (BASE/'panel/COMPLETE.json').exists(), 'real panel must be reviewed first'
    assert (BASE/'panel/ROOT_READBACK.json').exists(), 'root panel readback required'
    assert not (OUT/'LOCK.json').exists()
    config = read(BASE/'CONFIG.json') | {
        'phase_review_seconds': 3600,
        'no_readout_change': 'original decode_two_pass for common A and all B arms',
        'frozen_reuse': 'registered matched source baseline, identities checked after all new seals',
        'stop': 'one A epoch, one B epoch, fixed final endpoints, no best selection',
        'B_schedule_detail': 'unchanged original B horizon775: warmup39 of actual155 updates (25.16%); only first155 schedule positions run, not a full one-epoch cosine cycle',
        'order': 'sorted keys then Random(seed+0) A / Random(seed+1) B',
        'common_rng': 'each paired B query starts with identical CPU/CUDA RNG; checkpoint stores last arm RNG',
        'checkpoint': 'atomic complete all-arm accumulation window, optimizer/RNG/cursor; logs recoverable',
    }
    write(OUT/'CONFIG.json', config)
    pins = dict(read(BASE/'LOCK.json')['pins'])
    for p in (Path(__file__), OUT/'CONFIG.json', BASE/'panel/ROOT_READBACK.json'):
        pins[str(p)] = sha(p)
    write(OUT/'LOCK.json', {'pins': pins, 'time': time.time()})


def save_state(state, adapters, optimizers):
    atomic_pt(OUT/'LATEST.pt', state | {
        'adapters': {a: cpu_copy(m.state_dict()) for a,m in adapters.items()},
        'optimizers': {a: cpu_copy(o.state_dict()) for a,o in optimizers.items()},
        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'lock_sha': sha(OUT/'LOCK.json')})


def commit_history(state):
    log = state.get('last_window')
    if log is None: return
    path = OUT/'history'/f"{log['stage']}_C{log['cursor']:04}.json"
    if path.exists(): assert read(path) == log
    else: write(path, log)


def initialize():
    torch.manual_seed(SEED)
    model = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
    assert adapter_sha256(model) == read(OLD/'INITIAL.json')['adapters']['dual_repaired']
    opt = make_source_optimizer(model, 'repaired', 'A')
    state = {'stage': 'A', 'cursor': 0, 'steps': {'common': 0}, 'last_window': None}
    adapters, optimizers = {'common': model}, {'common': opt}
    save_state(state, adapters, optimizers)
    write(OUT/'INITIAL.json', {'adapter_sha': adapter_sha256(model), 'same_original_dual_initial': True,
          'new_common_evidence_training': True, 'not_original_E0_recovery': True})
    return state, adapters, optimizers


def restore():
    saved = torch.load(OUT/'LATEST.pt', map_location='cpu', weights_only=False)
    assert saved['lock_sha'] == sha(OUT/'LOCK.json')
    adapters, optimizers = {}, {}
    for a, weights in saved['adapters'].items():
        m = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
        m.load_state_dict(weights)
        opt = make_source_optimizer(m, 'repaired', 'A' if saved['stage']=='A' else 'B')
        opt.load_state_dict(saved['optimizers'][a]); adapters[a] = m; optimizers[a] = opt
    state = {k:v for k,v in saved.items() if k not in ('adapters','optimizers','cpu_rng','cuda_rng','lock_sha')}
    torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    commit_history(state)
    return state, adapters, optimizers


def immutable_weights(path, payload):
    if path.exists():
        previous = torch.load(path, map_location='cpu', weights_only=False)
        assert previous.keys() == payload.keys()
        for a in payload:
            assert all(torch.equal(previous[a][n], cpu_copy(p)) for n,p in payload[a].items())
    else: atomic_pt(path, payload)


def transition_to_B(state, adapters, optimizers):
    assert state['stage'] == 'A' and state['cursor'] == 618 and state['steps']['common'] == 155
    common = cpu_copy(adapters['common'].state_dict())
    immutable_weights(OUT/'COMMON_A.pt', {'common': common})
    new_adapters, new_optimizers = {}, {}
    for arm in MODES:
        m = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False).cuda()
        m.load_state_dict(common)
        new_adapters[arm] = m; new_optimizers[arm] = make_source_optimizer(m, 'repaired', 'B')
    hashes = {a: adapter_sha256(m) for a,m in new_adapters.items()}
    assert len(set(hashes.values())) == 1
    state = {'stage': 'B', 'cursor': 0, 'steps': {a:0 for a in MODES}, 'last_window': None}
    save_state(state, new_adapters, new_optimizers)
    if not (OUT/'COMMON_A.json').exists():
        write(OUT/'COMMON_A.json', {'sha': sha(OUT/'COMMON_A.pt'), 'arm_hashes': hashes,
              'queries': 618, 'optimizer_steps': 155, 'validation_GT_read': False,
              'fresh_B_optimizers': True, 'not_recovered_original_E0': True})
    return state, new_adapters, new_optimizers


def fit(guard):
    from scripts.desta3d_source_fit_v1 import _training_inputs
    rows = read(OLD/'INPUTS.json')
    train = sorted([r for r in rows if r['split']=='train'], key=lambda r:r['key'])
    records = {r['key']:r for r in read(V1/'source_fit/SOURCE_TRAIN_RECORDS.json')}
    assert len(train)==618 and len({r['source'] for r in train})==95
    processor, backbone = model_setup()
    state, adapters, optimizers = restore() if (OUT/'LATEST.pt').exists() else initialize()
    while state['stage'] in ('A','B'):
        if state['cursor']==618:
            if state['stage']=='A':
                state, adapters, optimizers = transition_to_B(state, adapters, optimizers)
                gc.collect(); continue
            immutable_weights(OUT/'FINAL_B.pt', {a:cpu_copy(m.state_dict()) for a,m in adapters.items()})
            state['stage']='validation'; save_state(state, adapters, optimizers); break
        if guard(): return 'paused'
        ordered = list(train); random.Random(SEED+(0 if state['stage']=='A' else 1)).shuffle(ordered)
        window = ordered[state['cursor']:state['cursor']+4]; divisor = len(window)
        assert divisor in (2,4)
        log = {'stage':state['stage'], 'cursor':state['cursor']+divisor, 'keys':[r['key'] for r in window],
               'query_details':[], 'arms':{}}
        parts = {a:{k:empty_buffers(m) for k in ('event','spatial','task','aux')} for a,m in adapters.items()}
        for a,m in adapters.items():
            m.train(); m.zero_grad(set_to_none=True)
            set_source_lr(optimizers[a], state['steps'][a], 155 if state['stage']=='A' else 775, 'repaired')
        for row in window:
            prompt, prep, fields = fields_for(backbone, processor, row)
            details = {'key':row['key'], 'source':row['source'], 'preprocess':prep, 'arms':{}}
            if state['stage']=='A':
                m=adapters['common']
                out=m(fields['visual_grid'],fields['query_tokens'],query_mask=fields['query_mask'],frame_times=fields['frame_times'])
                loss=source_evidence_losses(out,records[row['key']])
                total=(loss['ref']+loss['event'])/divisor
                assert torch.isfinite(total); total.backward()
                details['arms']['common']={k:float(v.detach()) for k,v in loss.items()}
                del out,loss,total
            else:
                data,_,_=_training_inputs(processor,backbone,row,records[row['key']])
                rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
                for a,m in adapters.items():
                    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
                    one, info=query_gradients(backbone,processor,m,row,records[row['key']],fields,data,divisor)
                    for k in parts[a]: add_buffers(parts[a][k],one[k])
                    details['arms'][a]=info
                    del one
                del data
            log['query_details'].append(details)
            del prompt,fields
        assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
        for a,m in adapters.items():
            if state['stage']=='A':
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.))
                assert math.isfinite(norm)
                optimizers[a].step();optimizers[a].zero_grad(set_to_none=True)
                log['arms'][a]={'pre_clip_norm':norm,'clip_triggered':norm>1}
            else:
                log['arms'][a]={'gradients':gradient_summary(m,parts[a]),
                    'step':take_step(m,optimizers[a],parts[a]['task'],parts[a]['aux'],a)}
            assert all(torch.isfinite(p).all() for p in m.parameters())
            state['steps'][a]+=1
        state['cursor']+=divisor;log['steps']=dict(state['steps']);state['last_window']=log
        save_state(state,adapters,optimizers);commit_history(state)
        if state['cursor']%100==0 or state['cursor']==618:
            print('COMMITTED',state['stage'],state['cursor'],state['steps'],flush=True)
        del parts;gc.collect()
    if state['stage']=='validation':
        status=validate_all(backbone,processor,state,guard)
        if status=='paused':return status
        write(OUT/'PREDICTIONS_COMPLETE.json',{'seal_sha':sha(OUT/'ALL_PREDICTIONS_SEAL.json'),
              'source_validation_scoring':'pending independent readback','target_GT_read':False})
        return 'predictions_complete'
    raise RuntimeError(state['stage'])


def validate_all(backbone,processor,state,guard):
    from vg_tta.desta3d_v2_ptd import decode_two_pass
    from scripts.desta3d_v2_source_fit import prediction_record,verify_prediction_identity
    rows=sorted([r for r in read(OLD/'INPUTS.json') if r['split']=='validation'],key=lambda r:r['key'])
    weights=torch.load(OUT/'COMMON_A.pt',map_location='cpu',weights_only=False)
    weights.update(torch.load(OUT/'FINAL_B.pt',map_location='cpu',weights_only=False))
    m=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda()
    hashes={}
    for a,weight in weights.items():m.load_state_dict(weight);hashes[a]=adapter_sha256(m)
    paths=[]
    for index,row in enumerate(rows):
        if guard():return 'paused'
        ready=[]
        for a in weights:
            p=OUT/'predictions'/a/f'{index:03}.pt'
            if p.exists():verify_prediction_identity(torch.load(p,map_location='cpu',weights_only=False),row,hashes[a]);ready.append(a)
        if len(ready)==len(weights):continue
        prompt,prep,fields=fields_for(backbone,processor,row)
        for a,weight in weights.items():
            if a in ready:continue
            m.load_state_dict(weight);m.set_train_stage('frozen');m.eval();backbone.eval()
            with torch.inference_mode():result=decode_two_pass(backbone,processor,prompt,m,fields)
            atomic_pt(OUT/'predictions'/a/f'{index:03}.pt',prediction_record(result,row,prep,hashes[a]))
            del result
        del prompt,fields
    for a in weights:
        for index in range(len(rows)):paths.append(OUT/'predictions'/a/f'{index:03}.pt')
    write(OUT/'ALL_PREDICTIONS_SEAL.json',{'pins':{str(p):sha(p) for p in paths},'adapter_hashes':hashes,
          'queries_per_arm':198,'parents':31,'arms':list(weights),'new_predictions':792,
          'source_validation_GT_read':False,'target_GT_read':False,
          'frozen_baseline_pending_identity_crosscheck':str(V1/'source_fit/FROZEN_SOURCE_VAL_BASELINE.json')})
    return 'complete'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('register','fit'))
    ap.add_argument('--allocation',default='contrast001');ap.add_argument('--phase-seconds',type=float,default=3600)
    args=ap.parse_args()
    if args.action=='register':return register()
    for p,h in read(OUT/'LOCK.json')['pins'].items():assert sha(p)==h,p
    assert not (OUT/'PREDICTIONS_COMPLETE.json').exists()
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=OLD.parent/'receipts'/f'aux_backflow_{args.allocation}.json';assert not receipt.exists()
    began=time.monotonic();status='running';failure=None
    def guard():
        if shutil.disk_usage(ROOT).free<8*2**30:raise RuntimeError('8GiB disk reserve')
        return time.monotonic()-began>args.phase_seconds-30
    try:
        assert not guard();torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
        write(OUT/'starts'/f'{args.allocation}.json',{'pid':os.getpid(),'time':time.time(),
              'prior_seconds':sum(scan_nested_gpu_receipts(p)[0] for p in (V1,OLD.parent)),
              'phase_seconds':args.phase_seconds,'cumulative_cap_seconds':None})
        status=fit(guard)
    except BaseException:
        failure=traceback.format_exc();status='failed';raise
    finally:
        write(receipt,{'stage':'v2_aux_backflow_source_contrast','allocation':args.allocation,
              'status':status,'seconds':time.monotonic()-began,'failure':failure,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
              'cumulative_cap_seconds':None})


if __name__=='__main__':main()
