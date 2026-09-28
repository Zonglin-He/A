"""Four fixed source-only GT branch-latent oracle arms, with native PTD readout."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import *
from scripts.desta3d_v3_privileged_ptd_qualification import B1,B1_SHA,ADAPTER_SHA,equal
ARMS=('original','temporal','spatial','dual','wrong_temporal','wrong_spatial')

def register(name):
    assert sha(B1)==B1_SHA
    rows=read(PANEL/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
    labels=read(PANEL/'SOURCE_RECORDS.json');assert [r['key'] for r in labels]==[r['key'] for r in rows]
    paths=[Path(__file__),ROOT/'vg_tta/desta3d_v3_latent_oracle.py',B1,PANEL/'INPUTS.json',PANEL/'SOURCE_RECORDS.json',
        ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors',
        ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py']
    old=read(PANEL/'ALL_PREDICTIONS_SEAL.json')['pins']
    for i,row in enumerate(rows):
        p=PANEL/'episodes'/f'{i:02}'/'no_update.pt';assert old[str(p)]==sha(p);paths.append(p)
        assert labels[i]['source']==row['source'] and labels[i]['frame_ids']==row['input']['frame_ids']
        assert labels[i]['video_sha256']==row['input']['video_sha256']
    register_base(name,dict(stage='GT_privileged_3D_branch_latent_oracle',phase_seconds=1800,seed=20260927,alpha=.25,
        arms=list(ARMS),PTD='frozen official PTD4B + fixed B1; cancelled v3 excluded',checkpoint=str(B1),checkpoint_sha=B1_SHA,
        adapter_sha=ADAPTER_SHA,source_label_sha=sha(PANEL/'SOURCE_RECORDS.json'),intervention='post reader/LN/SiLU before output projection',
        spatial_missing_box='neutral1, no spatial event mask',primary=['temporal-original tIoU','spatial-original sIoU'],
        extra_native_identity_control=1,matched_wrong_controls=True,pixel_views=False,external_teacher=False),rows,paths)

def injection_effect(current,baseline):
    import torch
    a=baseline.detach().cpu().flatten();b=current.detach().cpu().flatten();assert a.shape==b.shape
    idx=(a!=b).nonzero().flatten();before=a[idx];after=b[idx];d=after.float()-before.float()
    return {'shape':list(current.shape),'dtype':str(current.dtype),'indices':idx,'before':before,'after':after,
        'numel':a.numel(),'changed_elements':idx.numel(),'delta_l2':float(d.double().norm()),
        'baseline_sha':tensor_sha(baseline),'current_sha':tensor_sha(current),
        'limitation':'All changed positions retained; unchanged endpoints verified and hashed in worker, not all stored'}

def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';dest=OUT/name
    with allocation(dest) as (cfg,guard):
        import torch,gc
        from scripts.ptd_spatial_adapter_ab_v1 import processor_load,model_load,inputs_for
        from scripts.desta3d_v2_source_fit import prediction_record
        from scripts.desta3d_v2_reference_audit_cached_v3 import details
        from scripts.desta3d_v2_tta8_recovery_v2 import observe_time
        from scripts.desta3d_v2_p0 import adapter_sha256
        from vg_tta.desta3d_v2 import Desta3DAdapterV2
        from vg_tta.desta3d_v2_ptd import capture_stock_fields
        from vg_tta.desta3d_v2_shared_reference_cached import decode_shared_reference_two_pass
        from vg_tta.desta3d_v2_prediction_contract import validate_prediction
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record,privileged_branch_latents,matched_wrong_masks
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        processor=processor_load();model=model_load().eval().requires_grad_(False)
        import model.ptd_generation as pg
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(torch.load(B1,map_location='cpu',weights_only=False)['adapter']);adapter.set_train_stage('frozen')
        assert adapter_sha256(adapter)==ADAPTER_SHA
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')}
        for i,row in enumerate(read(dest/'INPUTS.json')):
            guard();ep=dest/'episodes'/f'{i:02}';ep.mkdir(parents=True)
            frames,ids=decode(row['input']);prompt,pre=inputs_for(row,processor,frames)
            fields=capture_stock_fields(model,processor,prompt,row['input']['caption'],ids,row['input']['fps'])
            shape=fields['visual_grid'].shape;assert shape[1]==len(ids)
            masks=masks_from_source_record(labels[row['key']],ids,shape[2],shape[3]);torch.save(masks,ep/'ORACLE_MASKS.pt')
            wrong=matched_wrong_masks(labels[row['key']],ids,shape[2],shape[3]);torch.save(wrong,ep/'WRONG_MASKS.pt')
            support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)}
            support.update({k:tensor_sha(prompt[k]) for k in ('input_ids','pixel_values_videos','video_grid_thw')})
            write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],frame_ids=ids,preprocess=pre,support=support,
                source_GT_mask_sha=sha(ep/'ORACLE_MASKS.pt'),source_label_sha=cfg['source_label_sha'],pixel_intervention=False))
            baseline_tokens={};base_pred=None
            for arm in ARMS:
                guard()
                arm_masks=wrong[arm.removeprefix('wrong_')] if arm.startswith('wrong_') else masks
                with privileged_branch_latents(adapter,arm_masks,event=arm in ('temporal','dual','wrong_temporal'),spatial=arm in ('spatial','dual','wrong_spatial'),alpha=cfg['alpha']) as latent_log:
                    result,time_dist=observe_time(pg,processor,len(ids),lambda:decode_shared_reference_two_pass(model,processor,prompt,adapter,fields))
                p=prediction_record(result,row,pre,ADAPTER_SHA)
                p.update(arm=arm,readout=details(result),time_distribution=time_dist,support=support,
                    GT_read=arm!='original',worker_source_GT_read=True,oracle_latent=arm!='original',
                    decoder_GT_prefix=False,target_GT_read=False,optimizer_steps=0,latent_log=latent_log)
                torch.save(p,ep/(arm+'.pt'));validate_prediction(p,len(ids))
                if arm=='original':
                    old=torch.load(PANEL/'episodes'/f'{i:02}'/'no_update.pt',map_location='cpu',weights_only=False)
                    keys=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess',
                        'adapter_sha','event_completion','spatial_completion','time_distribution','readout']
                    checks={k:equal(p[k],old[k]) for k in keys};write(ep/'BASELINE_REPLAY.json',checks);assert all(checks.values()),checks
                    base_pred=p
                    for branch in ('event','spatial'):
                        state=result[branch+'_injection']
                        if state is not None:baseline_tokens[branch]=state['updated_tokens']
                effect={}
                for branch in ('event','spatial'):
                    state=result[branch+'_injection']
                    if state is not None and branch in baseline_tokens:
                        effect[branch]=injection_effect(state['updated_tokens'],baseline_tokens[branch])
                torch.save(effect,ep/(arm+'_INJECTION_EFFECT.pt'))
                if arm in ('spatial','wrong_spatial') and result['event']['format_ok']:
                    assert equal(p['time_distribution'],base_pred['time_distribution']) and p['event_completion']==base_pred['event_completion'],'spatial-only altered event branch'
                assert adapter_sha256(adapter)==ADAPTER_SHA
                assert all(p.grad is None and not p.requires_grad for p in adapter.parameters())
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
                assert all(tensor_sha(fields[k])==v for k,v in support.items() if k in fields)
                print('ORACLE',i+1,16,arm,p['format_ok'],flush=True)
                del result,p;gc.collect()
            if i==0:
                ones={b:torch.ones_like(masks[b]) for b in ('event','spatial')}
                with privileged_branch_latents(adapter,ones,event=True,spatial=True,alpha=cfg['alpha']) as log:
                    result,td=observe_time(pg,processor,len(ids),lambda:decode_shared_reference_two_pass(model,processor,prompt,adapter,fields))
                control=prediction_record(result,row,pre,ADAPTER_SHA);control.update(readout=details(result),time_distribution=td)
                torch.save(control,ep/'IDENTITY_CONTROL.pt')
                checks={k:equal(control[k],base_pred[k]) for k in keys};checks['latent_exact']=all(r['changed_elements']==0 for r in log)
                write(ep/'IDENTITY_CONTROL.json',checks);assert all(checks.values()),checks
                del result,control
            write(ep/'COMPLETE.json',dict(key=row['key'],primary_predictions=6,source_GT_for_masks=True,optimizer_steps=0,exact_state=True))
            del prompt,fields,baseline_tokens,masks,wrong,base_pred;gc.collect()
        write(dest/'COMPLETE.json',dict(status='all_oracle_predictions_complete_not_scored',queries=16,parents=16,predictions=96,
            extra_native_identity_control=1,seal_sha=seal(dest,96),source_GT_for_masks=True,target_read=False,optimizer_steps=0,
            GPU_peak_allocated=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True)
    a=p.parse_args();globals()[a.action](a.name)
