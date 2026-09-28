"""Stage B: frozen B1/PTD original, temporal, spatial, combined evidence views.

No training, source labels, or target inputs. All predictions are sealed before
an independent scorer opens the existing source-only label record.
"""
import argparse,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import *
ARMS=('original','temporal','spatial','combined')
B1=ROOT/'artifacts/desta3d_v2/aux_backflow_v1/final_B1_reference_v1/B1_FIXED_FINAL.pt'
B1_SHA='6fe8d07e2a11c16bcb68eed11264a18b71985e97fdbf7939422f306ba81bf0d1'
ADAPTER_SHA='4a2ef2cf87e1753fad7c0c5c1f582f499c14b0b2945a839680b296abb0d9c7eb'

def register(name,teacher):
    source=OUT/teacher;done,sealed=verify_seal(source)
    assert done['queries']==16 and sealed['predictions']==16 and sha(B1)==B1_SHA
    rows=read(source/'INPUTS.json');assert rows==read(PANEL/'INPUTS.json')
    old=read(PANEL/'ALL_PREDICTIONS_SEAL.json')['pins']
    baselines=[PANEL/'episodes'/f'{i:02}'/'no_update.pt' for i in range(16)]
    assert all(old[str(p)]==sha(p) for p in baselines)
    paths=[Path(__file__),ROOT/'vg_tta/external_privileged_views.py',ROOT/'vg_tta/desta3d_v2.py',
        ROOT/'vg_tta/desta3d_v2_ptd.py',ROOT/'vg_tta/desta3d_v2_shared_reference_cached.py',
        ROOT/'vg_tta/desta3d_v2_prediction_contract.py',ROOT/'vg_tta/exact_frame_decode_audit_v2.py',
        ROOT/'scripts/ptd_spatial_adapter_ab_v1.py',ROOT/'scripts/desta3d_v2_source_fit.py',
        ROOT/'scripts/desta3d_v2_reference_audit_cached_v3.py',ROOT/'scripts/desta3d_v2_tta8_recovery_v2.py',
        ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py',
        ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/model.safetensors',B1,
        PANEL/'INPUTS.json',PANEL/'SOURCE_RECORDS.json',source/'CONFIG.json',source/'COMPLETE.json',source/'PREDICTIONS_SEAL.json',*baselines]
    paths += [source/p for p in sealed['files']]
    register_base(name,{'stage':'same_PTD_privileged_policy_Q0','teacher':str(source),'phase_seconds':3600,
        'seed':20260927,'checkpoint':str(B1),'checkpoint_sha':B1_SHA,'adapter_sha':ADAPTER_SHA,'arms':list(ARMS),
        'PTD':'official frozen PTD4B with fixed B1 adapter; NOT cancelled 35-step v3','dim':.25,'blur_radius':8,
        'gap_limit':None,'primary_comparisons':['temporal-original tIoU','spatial-original sIoU'],
        'source_label_sha':sha(PANEL/'SOURCE_RECORDS.json'),'baseline_reuse':'fresh original physical/native replay must match saved B1'},rows,paths)

def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run(name):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    dest=OUT/name
    with allocation(dest) as (cfg,guard):
        import torch,numpy as np,gc,hashlib
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
        from vg_tta.external_privileged_views import temporal_view,spatial_view
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        processor=processor_load();model=model_load().eval().requires_grad_(False)
        import model.ptd_generation as pg
        adapter=Desta3DAdapterV2(hidden_dim=128,architecture='dual3d',p1_enabled=False).cuda().eval()
        adapter.load_state_dict(torch.load(B1,map_location='cpu',weights_only=False)['adapter']);adapter.set_train_stage('frozen')
        assert adapter_sha256(adapter)==ADAPTER_SHA
        teacher=Path(cfg['teacher']);verify_seal(teacher)
        for i,row in enumerate(read(dest/'INPUTS.json')):
            guard();frames,ids=decode(row['input']);ep=dest/'episodes'/f'{i:02}';te=teacher/'episodes'/f'{i:02}'
            evidence=read(te/'EVIDENCE.json');ident=read(te/'INPUT.json')
            pixel=hashlib.sha256(frames.tobytes()).hexdigest()
            assert ident['key']==row['key'] and ident['frame_ids']==ids and ident['pixel_sha256']==pixel
            assert ident['clip_bounds']==[row['input']['start_frame'],row['input']['end_frame']]
            tv,td=temporal_view(frames,ids,evidence['interval_physical'],cfg['dim'])
            sv,sd=spatial_view(frames,ids,evidence,cfg['blur_radius']);tsv,cs=spatial_view(tv,ids,evidence,cfg['blur_radius'])
            write(ep/'INPUT.json',{'key':row['key'],'source':row['source'],'frame_ids':ids,'physical_pixel_sha':pixel,
                'teacher_input_sha':sha(te/'INPUT.json'),'evidence_sha':sha(te/'EVIDENCE.json'),'temporal_view':td,'spatial_view':sd,
                'combined_order':'temporal dim then spatial blur','original_shape':list(frames.shape)})
            original_support=None
            for arm,view in zip(ARMS,[frames,tv,sv,tsv]):
                guard();assert view.shape==frames.shape and view.dtype==frames.dtype
                prompt,pre=inputs_for(row,processor,view)
                fields=capture_stock_fields(model,processor,prompt,row['input']['caption'],ids,row['input']['fps'])
                support={k:tensor_sha(prompt[k]) for k in ['input_ids','video_grid_thw']}
                support['frame_times']=tensor_sha(fields['frame_times'])
                if original_support is None:original_support=support
                else:assert support==original_support,'view changed time/query/grid support'
                result,time_dist=observe_time(pg,processor,len(ids),lambda:decode_shared_reference_two_pass(model,processor,prompt,adapter,fields))
                pred=prediction_record(result,row,pre,adapter_sha256(adapter))
                pred.update(arm=arm,readout=details(result),time_distribution=time_dist,sourcefit_adapter_sha=ADAPTER_SHA,
                    decoder_uses_GT=False,target_GT_read=False,optimizer_steps=0,evidence_sha=sha(te/'EVIDENCE.json'),support=support)
                torch.save(pred,ep/(arm+'.pt'))  # save before any semantic/format assertions
                validate_prediction(pred,len(ids))
                if arm=='original':
                    old=torch.load(PANEL/'episodes'/f'{i:02}'/'no_update.pt',map_location='cpu',weights_only=False)
                    keys=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess',
                        'adapter_sha','event_completion','spatial_completion','time_distribution','readout']
                    checks={k:equal(pred[k],old[k]) for k in keys}
                    write(ep/'BASELINE_REPLAY.json',checks);assert all(checks.values()),checks
                assert adapter_sha256(adapter)==ADAPTER_SHA and all(p.grad is None and not p.requires_grad for p in adapter.parameters())
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
                del prompt,fields,result,pred;gc.collect()
            write(ep/'COMPLETE.json',{'key':row['key'],'predictions':4,'exact_state':True,'optimizer_steps':0})
            print('PRIVILEGED_PTD',i+1,16,flush=True)
        write(dest/'COMPLETE.json',{'status':'all_predictions_complete_not_scored','queries':16,'parents':16,'predictions':64,
            'seal_sha':seal(dest,64),'source_GT_read':False,'target_read':False,'optimizer_steps':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);p.add_argument('--teacher')
    a=p.parse_args();register(a.name,a.teacher) if a.action=='register' else run(a.name)
