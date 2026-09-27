"""Finite source-only real-PTD semantic/gradient acceptance; no optimizer or TTA."""
from __future__ import annotations
import argparse, fcntl, gc, hashlib, json, math, os, shutil, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.desta3d_tta_run_v1 import (sha256_file as sha, read_json as read,
    write_json_once as write, tensor_sha256, cpu_copy,
    scan_nested_gpu_receipts)
OUT=ROOT/'artifacts/desta3d_v2/p0'
PARENT=OUT.parent
CK=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B'

def adapter_sha256(adapter):
    """Hash tensor dtype/shape/content, including v2's scalar gates."""
    h=hashlib.sha256()
    for name,value in sorted(adapter.state_dict().items()):
        tensor=value.detach().contiguous().cpu()
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(repr(tuple(tensor.shape)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def register(run_id):
    run=OUT/run_id
    if run.exists():raise FileExistsError(run)
    rows=read(ROOT/'artifacts/desta3d_v1/tta_run/pilot_v1/INPUTS.json')
    assert len(rows)==len({r['source'] for r in rows})==4
    records=read(ROOT/'artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json')
    chosen=next(r for r in sorted(records,key=lambda r:r['key']) if r['response_eligible'])
    inputs=read(ROOT/'artifacts/desta3d_v1/SOURCE_INPUTS.json')
    source=next(r for r in inputs if r['key']==chosen['key'])
    assert source['split']=='train' and source['source'] not in {r['source'] for r in rows}
    write(run/'INPUTS.json',rows)
    write(run/'GRADIENT_INPUT.json',source)
    config={'run_id':run_id,'seed':20260927,'condition':'clean',
      'architecture':'dual3d','hidden_dim':128,'p1_enabled':False,
      'inference_queries':4,'inference_parents':4,'source_gradient_queries':1,
      'source_gradient_key':chosen['key'],'gradient_selection':'lexical first eligible source-train response; not task score',
      'optimizer_steps':0,'TTA_steps':0,'target_GT_read':False,'validation_GT_read':False,
      'source_GT_use':'only one registered source-train example for branch-separated task gradients; no parameter step',
      'tests':['stock vs two-pass exact zero gate','independent nonzero injections',
               'actual branch isolation by first-query gate interventions','real first-step branch task gradients'],
      'initial_gate':'sigmoid(-6)','out_projection_std':.001,
      'cumulative_cap_seconds':None,'phase_cap_seconds':1800,'free_disk_floor_bytes':8*2**30,
      'prior_v1_seconds':15185.645719446977,
      'note':'engineering acceptance only; independent query capture prefill plus two generation prefills; no scientific gain gate'}
    write(run/'CONFIG.json',config)
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v2.py',ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_source.py',
      ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',ROOT/'scripts/desta3d_source_fit_v1.py',
      ROOT/'scripts/ptd_8b_teacher_feasibility_v1.py',ROOT/'vg_tta/desta3d_v1.py',
      ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
      ROOT/'artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json',
      ROOT/'methods/CURRENT_METHOD.json',ROOT/'external/ParallelTubeDecoding/src/dataset/sft_dataset.py',
      ROOT/'external/ParallelTubeDecoding/src/train/monkey_patch_forward.py',
      ROOT/'external/ParallelTubeDecoding/src/dataset/data_utils.py',
      ROOT/'vg_tta/exact_frame_decode_audit_v2.py',ROOT/'scripts/corruption_route_retest_v1.py',
      ROOT/'scripts/c1_controlled_corruption_v1.py',
      run/'CONFIG.json',run/'INPUTS.json',run/'GRADIENT_INPUT.json',CK/'OFFICIAL_RECEIPT.json',CK/'model.safetensors']
    write(run/'LOCK.json',{'registered_unix':time.time(),'pins':{str(p):sha(p) for p in paths}})
    print(json.dumps({'registered':str(run),'source_gradient_key':chosen['key']}),flush=True)

def brief(pred):
    return {k:pred.get(k) for k in ('interval','format_ok','completion','positions','boxes','semantic','temporal')}

def compact_decode(result):
    record={k:result.get(k) for k in ('event','interval','format_ok','GT_used','cache_policy')}
    record['spatial']=brief(result['spatial']) if result['spatial'] is not None else None
    for name in ('event_injection','spatial_injection'):
        state=result[name]
        if state is None:record[name]=None;continue
        fields=state['fields']
        record[name]={k:state[k] for k in ('branch','calls','zero_exact','changed_elements','relative_injection_norm')}
        record[name]['updated_tokens_sha']=tensor_sha256(state['updated_tokens'])
        record[name]['field_hashes']={k:tensor_sha256(fields[k]) for k in ('branch_features_spatial','branch_features_event','delta_spatial','delta_event')}
        record[name]['evidence']={k:cpu_copy(fields[k]) for k in ('referent_logits','event_logits','gate_spatial','gate_event','alpha_spatial','alpha_event')}
    return cpu_copy(record)

def run(run_id):
    base=OUT/run_id;config=read(base/'CONFIG.json');lock=read(base/'LOCK.json')
    for path,h in lock['pins'].items():assert sha(path)==h, f'pin changed: {path}'
    assert not (base/'STARTED.json').exists(),'no silent replay of existing invocation'
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running';failure=None
    v1_used=v2_used=0.;prior_verified=False
    def check():
        if shutil.disk_usage(ROOT).free<config['free_disk_floor_bytes']:raise RuntimeError('8GiB disk reserve')
        if time.monotonic()-start>config['phase_cap_seconds']:raise RuntimeError('finite P0 phase cap; save and review')
    try:
        v1_used,_=scan_nested_gpu_receipts(ROOT/'artifacts/desta3d_v1')
        v2_used,_=scan_nested_gpu_receipts(PARENT)
        assert abs(v1_used-config['prior_v1_seconds'])<.01
        prior_verified=True
        check();write(base/'STARTED.json',{'pid':os.getpid(),'unix':time.time(),'prior_seconds':v1_used+v2_used})
        torch.set_num_threads(4);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        torch.cuda.reset_peak_memory_stats()
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,frames_for,inputs_for,infer
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields,decode_two_pass,branch_injection
        pr=processor_load();model=model_load()
        adapter=Desta3DAdapterV2(hidden_dim=config['hidden_dim'],architecture=config['architecture'],
                                p1_enabled=config['p1_enabled'],train_stage='B integration').to('cuda')
        ahash=adapter_sha256(adapter)
        write(base/'PARAMETERS.json',{'by_group':adapter.parameter_count_by_group(),
          'tta_count':adapter.tta_parameter_count(),'initial_sha':ahash,
          'backbone_all_frozen':all(not p.requires_grad for p in model.parameters())})
        checks=[]
        for index,row in enumerate(read(base/'INPUTS.json')):
            check();frames,ids=frames_for(row,'clean');data,prep=inputs_for(row,pr,frames)
            fields=capture_stock_fields(model,pr,data,row['input']['caption'],ids,row['input']['fps'])
            stock=infer(model,pr,data)
            zero=decode_two_pass(model,pr,data,adapter,fields,gate_override=0.)
            assert zero['event_injection']['zero_exact']
            if zero['spatial'] is not None:
                assert zero['spatial_injection']['zero_exact']
                assert zero['spatial']['completion']==stock['completion'],'zero-gate full official decode mismatch'
                assert torch.equal(zero['spatial'].get('boxes',torch.empty(0)),stock.get('boxes',torch.empty(0)))
            assert zero['interval']==stock['interval']
            initial=decode_two_pass(model,pr,data,adapter,fields)
            for name in ('event_injection','spatial_injection'):
                actual=initial[name]
                if actual is not None:
                    assert not actual['zero_exact'] and actual['relative_injection_norm']>0 and actual['changed_elements']>0, f'BF16 rounded away {name}'
            rowcheck={'key':row['key'],'stock_format':stock['format_ok'],'zero_format':zero['format_ok'],
              'initial_format':initial['format_ok'],'zero_exact':True,'caption_tokens':fields['metadata']['token_count'],
              'grid':list(fields['visual_grid'].shape),'event_spatial_probe_count':initial['event']['spatial_probe_count'],
              'event_injection_norm':initial['event_injection']['relative_injection_norm'],
              'event_changed_elements':initial['event_injection']['changed_elements'],
              'spatial_changed_elements':initial['spatial_injection']['changed_elements'] if initial['spatial'] is not None else None,
              'spatial_injection_norm':initial['spatial_injection']['relative_injection_norm'] if initial['spatial'] is not None else None}
            if index==0:
                # Controls concern dataflow, not task scores: branch gate ablation.
                onlyevent=decode_two_pass(model,pr,data,adapter,fields,gate_override=(0.,float(torch.sigmoid(adapter.gate_event))))
                onlyspatial=decode_two_pass(model,pr,data,adapter,fields,gate_override=(float(torch.sigmoid(adapter.gate_spatial)),0.))
                assert onlyevent['event']['completion']==initial['event']['completion']
                assert torch.equal(onlyevent['event_injection']['updated_tokens'],initial['event_injection']['updated_tokens'])
                assert onlyspatial['event']['completion']==zero['event']['completion']
                if onlyevent['spatial'] is not None:assert onlyevent['spatial_injection']['zero_exact']
                if onlyspatial['spatial'] is not None and initial['spatial'] is not None:
                    assert torch.equal(onlyspatial['spatial_injection']['updated_tokens'],initial['spatial_injection']['updated_tokens'])
                rowcheck['branch_isolation_controls_pass']=bool(
                    onlyevent['spatial'] is not None and onlyspatial['spatial'] is not None
                    and initial['spatial'] is not None)
                del onlyevent,onlyspatial
            payload={'row':row,'preprocess':prep,'input_ids_sha':tensor_sha256(data['input_ids']),
              'pixels_sha':tensor_sha256(data['pixel_values_videos']),'query_metadata':fields['metadata'],
              'stock':brief(stock),'zero':compact_decode(zero),'initial':compact_decode(initial),'checks':rowcheck}
            torch.save(payload,base/f'query_{index}.pt')
            checks.append(rowcheck);write(base/f'CHECK_{index}.json',rowcheck)
            print('QUERY',json.dumps(rowcheck),flush=True)
            del data,fields,stock,zero,initial,payload,frames;gc.collect();torch.cuda.empty_cache()
        assert any(r['initial_format'] for r in checks), 'no complete legal two-pass decode exercised'
        assert checks[0].get('branch_isolation_controls_pass'), 'registered first-query spatial isolation did not execute'
        check()
        # Source-only teacher-forced gradient test. Does not take an optimizer step.
        from scripts.desta3d_source_fit_v1 import _training_inputs
        from scripts.ptd_8b_teacher_feasibility_v1 import joint_loss
        from vg_tta.desta3d_v2_source import split_source_loss_masks
        row=read(base/'GRADIENT_INPUT.json')
        record=next(r for r in read(ROOT/'artifacts/desta3d_v1/source_fit/SOURCE_TRAIN_RECORDS.json') if r['key']==row['key'])
        assert record['split']=='train'
        frames,ids=frames_for(row,'clean');prompt,prep=inputs_for(row,pr,frames)
        fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
        data,_,_= _training_inputs(pr,model,row,record)
        masks=split_source_loss_masks(data,pr.tokenizer)
        model.train();model.model.visual.eval()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        grads={}
        for branch in ('event','spatial'):
            check()
            adapter.zero_grad(set_to_none=True)
            branch_data=dict(data);branch_data['labels']=data['labels'].clone()
            branch_data['labels'][~masks[branch].to(data['labels'].device)]=-100
            with branch_injection(model,adapter,data,fields,branch) as state:
                loss,stats=joint_loss(model,branch_data)
                assert torch.isfinite(loss);loss.backward()
            byname={n:float(p.grad.float().norm()) if p.grad is not None else 0. for n,p in adapter.named_parameters()}
            assert all(math.isfinite(g) for g in byname.values())
            event_groups=('event_reader.','event_temporal_reader.','query_pool_event.','film_event.','out_proj_event.','gate_event')
            spatial_groups=('spatial_reader.','query_pool_spatial.','film_spatial.','out_proj_spatial.','gate_spatial')
            wanted=event_groups if branch=='event' else spatial_groups
            opposite=spatial_groups if branch=='event' else event_groups
            for name in ('input_proj.','shared_stem.'):
                assert sum(v for n,v in byname.items() if n.startswith(name))>0, f'missing shared task gradient {branch}/{name}'
            for name in wanted:assert sum(v for n,v in byname.items() if n.startswith(name))>0, f'missing real task gradient {branch}/{name}'
            for name in opposite:assert sum(v for n,v in byname.items() if n.startswith(name))==0, f'cross-branch gradient {branch}/{name}'
            assert all(p.grad is None for p in model.parameters())
            grads[branch]={'loss':float(loss.detach()),'tokens':int(masks[branch].sum()),'stats':stats,'gradients':byname,
                            'frozen_backbone_grad_none':True,'opposite_branch_gradient_zero':True}
            write(base/f'GRADIENT_{branch}.json',grads[branch]);print('GRADIENT',branch,grads[branch]['loss'],flush=True)
            check()
            del loss,state,branch_data;gc.collect();torch.cuda.empty_cache()
        assert adapter_sha256(adapter)==ahash
        check()
        predictions={str(p):sha(p) for p in sorted(base.glob('query_*.pt'))}
        write(base/'PREDICTIONS_SEAL.json',{'pins':predictions,'target_GT_read':False,'validation_GT_read':False})
        write(base/'COMPLETE.json',{'status':'P0_engineering_pass_not_source_or_TTA_gain','checks':checks,
              'source_gradient_key':row['key'],'real_gradients_pass':True,'adapter_unchanged':True,
              'source_GT_used_only_for_gradient':True,'target_GT_read':False,'validation_GT_read':False})
        status='completed'
    except BaseException:
        status='failed';failure=traceback.format_exc();write(base/'FAILURE.json',{'traceback':failure,'unix':time.time()});raise
    finally:
        elapsed=time.monotonic()-start
        write(PARENT/'receipts'/f'{run_id}.json',{'stage':'v2_P0_real_interface','seconds':elapsed,
          'status':status,'failure':failure,'prior_seconds':v1_used+v2_used if prior_verified else None,
          'prior_accounting_verified':prior_verified,
          'cumulative_seconds':v1_used+v2_used+elapsed if prior_verified else None,'cumulative_cap_seconds':None,
          'peak_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None})
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run']);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();globals()[args.action](args.run_id)
