"""Frozen PTD runner for the complete DESTA hyperparameter experiment.

No label-loading or scoring function is imported. Offline selection is a separate
CPU process operating only after each complete stage has been sealed.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta_native_common import D, A0, A3, GAP, read, write, sha, load, stage, complete, verified as original_verified, tensor_sha
from vg_tta.desta3d_v3_oracle_io import allocation


def verified(directory):
    if not (directory/'COMPLETE.json').exists() and (directory/'RESUME.json').exists():
        from scripts.desta_native_common import check_pins
        check_pins({str(directory/p):h for p,h in read(directory/'RESUME.json')['files'].items()})
        return False
    return original_verified(directory)


def inputs(processor, model, row):
    from scripts.ptd_spatial_adapter_ab_v1 import inputs_for
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    import hashlib
    frames, ids = decode(row['input'])
    assert ids == row['input']['frame_ids']
    prompt, pre = inputs_for(row, processor, frames)
    fields = capture_stock_fields(model, processor, prompt, row['input']['caption'], ids, row['input']['fps'])
    return prompt, pre, fields, hashlib.sha256(frames.tobytes()).hexdigest()


def run(phase, name):
    dependencies = [Path(__file__), ROOT/'desta3d/native_adaptation.py', ROOT/'scripts/desta_native_common.py',
                    ROOT/'vg_tta/desta3d_v3_actuation_full_vocab.py', ROOT/'vg_tta/desta3d_v3_decomposition.py',
                    ROOT/'vg_tta/desta3d_v2_output_anchor_memory_v7.py',
                    D/'ROOT_EXPERT_READBACK.json', D/'EXPERT_EVIDENCE_SEAL.json']
    assert read(D/'ROOT_EXPERT_READBACK.json')['status']=='passed'
    dest = stage(name, dependencies)
    with allocation(dest) as (cfg, guard):
        import torch
        from desta3d.native_adaptation import (AdaptationConfig, temporal_target, spatial_target, native_loss,
                                               project_branch, normalized_update, l2)
        from scripts.desta3d_v3_a05_signal import setup
        from scripts.desta3d_v3_a0_fast_screen import ADAPTER_SHA
        from scripts.desta3d_v3_joint_learnability import equal
        from vg_tta.desta3d_v3_decomposition import shared_fields
        from vg_tta.desta3d_v3_actuation_support import spatial_positions
        from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher, replay_branch
        from vg_tta.desta3d_v2_output_anchor_memory_v7 import release_free_host_arenas
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_p0 import adapter_sha256
        rows = read(D/'DEV64.json')
        if phase == 'dev16':
            indices = cfg['dev16_indices']; configurations = cfg['grid']
        elif phase == 'dev64':
            indices = list(range(64)); chosen = read(D/'scores/dev16/SELECTION.json')['selected']
            configurations = {key: cfg['grid'][key] for key in chosen}
        elif phase == 'ablations':
            indices = list(range(64)); chosen = read(D/'scores/dev64/SELECTION.json')['selected'][0]
            selected = cfg['grid'][chosen]
            configurations = {'T-only':{**selected,'spatial_weight':0.}, 'S-only':{**selected,'temporal_weight':0.}}
        else: raise ValueError(phase)
        for expert in ('temporal','spatial'):
            assert (D/'experts'/expert/'COMPLETE.json').exists()
        processor, model, adapter = setup(cfg['seed'])
        basis = load(D/'QC.pt').cuda(); backwards = new_predictions = 0; started = time.monotonic()
        for i in indices:
            guard(); row = rows[i]; ep = D/'native'/f'{i:02}'
            if all(verified(ep/key) for key in configurations): continue
            prompt, pre, fields, pixels_sha = inputs(processor, model, row)
            stock = fields['visual_grid'].detach(); stock_norm = l2(stock)
            support = {k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            cached = read(A0/'cache/dev'/f'{i:04}'/'INPUT.json')
            assert pre==cached['preprocess'] and support==cached['support']
            old = GAP/'episodes'/f"{row['gap_index']:04}"
            old_trace = load(old/'BASE_TRACE.pt'); old_base = load(old/'B1.pt')
            for expert in ('temporal','spatial'):
                assert verified(D/'experts'/expert/f'{i:02}')
            temporal = read(D/'experts/temporal'/f'{i:02}'/'EVIDENCE.json')
            spatial = read(D/'experts/spatial'/f'{i:02}'/'EVIDENCE.json')
            assert temporal['pixel_sha256']==spatial['pixel_sha256']==pixels_sha
            assert temporal['frame_ids']==spatial['frame_ids']==row['input']['frame_ids']
            assert old_base['support']==support and old_base['adapter_sha']==ADAPTER_SHA
            identity = ep/'identity'
            if not verified(identity):
                zero = torch.zeros((*stock.shape[:-1],16), device=stock.device)
                corrected = stock + zero@basis.T; assert torch.equal(corrected,stock)
                with torch.no_grad(), shared_fields(adapter, {'event':corrected,'spatial':corrected}, ['event','spatial'], allow_prefix=True):
                    native, trace = capture_teacher(model, processor, prompt, adapter, fields)
                baseline = prediction_record(native,row,pre,ADAPTER_SHA)
                baseline.update(readout=details(native),support=support)
                keys = ['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval',
                        'format_ok','preprocess','adapter_sha','event_completion','spatial_completion','readout','video_sha256','support']
                checks = {k:equal(baseline[k],old_base[k]) for k in keys}
                trace_checks = [torch.equal(a['logits'][k], b['logits'][k]) for a,b in zip(trace['branches'],old_trace['branches']) for k in a['logits']]
                assert len(trace['branches'])==len(old_trace['branches']) and all(trace_checks) and all(checks.values())
                write(identity/'IDENTITY.json', dict(checks=checks, all_native_logits_exact=True, pixel_sha256=pixels_sha,
                    stock_sha=tensor_sha(stock), stock_norm=stock_norm, support=support, GT_read=False))
                torch.save(baseline,identity/'B1.pt')
                complete(identity, numerical_no_update=True)
                del native,trace,baseline,corrected,zero
            trace = old_trace
            targets = {}; branch_specs = [('event',0,'time'),('spatial',1,'coordinate')]
            for branch,j,kind in branch_specs:
                if len(trace['branches'])<=j or kind not in trace['branches'][j]['logits']:
                    targets[branch] = None; continue
                if branch=='event':
                    targets[branch] = temporal_target(temporal['interval_physical'],row['input']['frame_ids'])
                else:
                    targets[branch] = spatial_target(spatial['boxes_by_frame'],row['input']['frame_ids'],
                                                     spatial_positions(trace),trace['coordinate_ids'])
            target_path = ep/'PSEUDO_TARGETS.pt'
            if not target_path.exists():
                torch.save(targets,target_path)
                write(ep/'INPUT.json',dict(key=row['key'],support=support,preprocess=pre,stock_sha=tensor_sha(stock),
                    stock_norm=stock_norm,trace_sha=sha(old/'BASE_TRACE.pt'),targets_sha=sha(target_path),
                    temporal_evidence_sha=sha(D/'experts/temporal'/f'{i:02}'/'EVIDENCE.json'),
                    spatial_evidence_sha=sha(D/'experts/spatial'/f'{i:02}'/'EVIDENCE.json'),GT_read=False))
            else: assert equal(targets,load(target_path))

            def gradient_pair(c, active):
                nonlocal backwards
                projected = {}; norms = {}; records = {}
                for branch,j,kind in branch_specs:
                    target = targets[branch]
                    if branch not in active or target is None or not target[1].any():
                        projected[branch]=torch.zeros_like(c); norms[branch]=0.
                        records[branch]={'missing_or_disabled':True}; continue
                    guard(); leaf=(stock+c@basis.T).detach().requires_grad_()
                    cleanup=release_free_host_arenas(); torch.cuda.empty_cache()
                    print('HOST_BEFORE_BRANCH',branch,cleanup,flush=True)
                    with shared_fields(adapter,{branch:leaf},[branch]):
                        logits,cache=replay_branch(model,prompt,adapter,fields,trace,branch)
                        assert logits.shape[-1]==(stock.shape[1] if branch=='event' else 152775)
                        identity_match = torch.equal(logits.detach().cpu(),trace['branches'][j]['logits'][kind]) if not c.any() else None
                        if not c.any(): assert identity_match, 'C0 native logits mismatch'
                        y,v=target; loss=native_loss(logits,y.cuda(),v.cuda())
                        assert torch.isfinite(loss)
                        grad,=torch.autograd.grad(loss,leaf); backwards+=1
                    proj,full_norm=project_branch(grad,basis)
                    projected[branch]=proj; norms[branch]=full_norm
                    records[branch]=dict(loss=float(loss.detach()),full_F_norm=full_norm,projected_norm=l2(proj),
                        full_gradient_sha=tensor_sha(grad),classes=logits.shape[-1],actions=int(v.sum()),C0_exact=identity_match)
                    if i==cfg['dev16_indices'][0] and not c.any() and not (ep/(branch+'_C0_FULL_GRAD.pt')).exists():
                        torch.save(grad.cpu(),ep/(branch+'_C0_FULL_GRAD.pt'))
                    del leaf,logits,cache,loss,grad,proj
                    gc.collect(); torch.cuda.empty_cache()
                assert all(not p.requires_grad and p.grad is None for m in (model,adapter) for p in m.parameters())
                return projected,norms,records

            # The exact same C0 and fixed pseudo-targets occur in every grid arm.
            # Reuse their raw initial gradients; later states are recomputed.
            zero_path=ep/'INITIAL_GRADIENTS.pt'
            if not zero_path.exists():
                c=torch.zeros((*stock.shape[:-1],16),device=stock.device)
                p,n,r=gradient_pair(c,{'event','spatial'})
                torch.save(dict(projected={k:v.cpu() for k,v in p.items()},full_norms=n,records=r),zero_path)
                del c,p,n,r
            initial=load(zero_path)
            for key, values in configurations.items():
                arm=ep/key
                if verified(arm): continue
                # Stop only at a sealed configuration boundary.
                if time.monotonic()-started>cfg['phase_seconds']-180:
                    write(dest/'COMPLETE.json',dict(status='safe_pause',new_predictions=new_predictions,backwards=backwards)); return
                guard(); t0=time.monotonic(); arm.mkdir(parents=True,exist_ok=True)
                config=AdaptationConfig(**values); c=torch.zeros((*stock.shape[:-1],16),device=stock.device)
                active={b for b,w in [('event',config.temporal_weight),('spatial',config.spatial_weight)] if w>0}
                start_step=0
                if (arm/'RESUME.json').exists():
                    start_step=read(arm/'RESUME.json')['completed_steps']
                    c=load(arm/f'STEP{start_step:02}.pt')['coeff_after'].cuda()
                for k in range(start_step,config.steps):
                    if k==0:
                        p={b:v.cuda() for b,v in initial['projected'].items()}; n=initial['full_norms']; records=initial['records']
                    else: p,n,records=gradient_pair(c,active)
                    before_sha=tensor_sha(c)
                    c,meta=normalized_update(c,p,n,basis,stock_norm,config)
                    assert meta['delta_norm']<=config.radius*stock_norm*(1+2e-6)
                    torch.save(dict(projected={b:v.cpu() for b,v in p.items()},full_norms=n,records=records,
                        coeff_after=c.cpu(),before_sha=before_sha,metadata=meta),arm/f'STEP{k+1:02}.pt')
                    del p,n,records
                corrected=stock+c@basis.T
                with torch.no_grad(),shared_fields(adapter,{'event':corrected,'spatial':corrected},['event','spatial'],allow_prefix=True) as calls:
                    result=decode_shared_reference_two_pass(model,processor,prompt,adapter,fields)
                pred=prediction_record(result,row,pre,ADAPTER_SHA)
                pred.update(arm=key,readout=details(result),support=support,GT_read=False,decoder_GT_prefix=False,
                    target_read=False,config=values,latent_updates=config.steps,optimizer_steps=0,
                    injection=dict(calls=calls,same_field_both_passes=True,common_F_sha=tensor_sha(stock),
                        corrected_F_sha=tensor_sha(corrected),relative_norm=l2(corrected-stock)/stock_norm))
                validate_prediction(pred,len(row['input']['frame_ids']))
                torch.save(pred,arm/'PREDICTION.pt'); torch.save(c.cpu(),arm/'FINAL_COEFFICIENTS.pt')
                assert adapter_sha256(adapter)==ADAPTER_SHA and torch.equal(basis.cpu(),load(D/'QC.pt'))
                complete(arm, config=values,index=i,seconds=time.monotonic()-t0,GT_read=False,frozen_scope=True)
                new_predictions+=1
                print('NATIVE_COMPLETE',phase,i,key,'seconds',time.monotonic()-t0,flush=True)
                del c,corrected,result,pred,initial
                initial=load(zero_path); gc.collect(); torch.cuda.empty_cache()
                # Recycle this process after one sealed configuration; v7 stays unchanged.
                write(dest/'COMPLETE.json',dict(status='safe_pause',new_predictions=new_predictions,backwards=backwards,process_recycle=True))
                return
            del prompt,fields,stock,trace,old_trace,old_base,initial
            gc.collect(); torch.cuda.empty_cache()
        seal={}
        for i in indices:
            for key in configurations:
                arm=D/'native'/f'{i:02}'/key; assert verified(arm)
                seal[str((arm/'COMPLETE.json').relative_to(D))]=sha(arm/'COMPLETE.json')
        write(D/(phase+'_SEAL.json'),dict(files=seal,indices=indices,configurations=configurations,GT_read=False))
        write(dest/'COMPLETE.json',dict(status='complete',new_predictions=new_predictions,backwards=backwards))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['dev16','dev64','ablations']); p.add_argument('--run',required=True)
    args=p.parse_args(); run(args.phase,args.run)
