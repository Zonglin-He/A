"""At most eight native TubeDETR interface probes, never a transfer-effect claim."""
from pathlib import Path
import gc,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
BASE=c.BASE/'tubedetr_qualification8';read,sha,put=c.read,c.sha,c.write
def verify():
    c.verify()
    for p,h in read(BASE/'RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def worker():
    import torch,numpy as np
    from scripts.stvg_native_instance_common_v1 import guard,write
    from scripts import run_stvg_table1_source_v1 as source
    from scripts.run_decota_paper_main_v1 import frames_for
    from vg_tta import tubedetr_runtime as rt
    from vg_tta.native_support_decode_fig1_v1 import TubeAdapter
    from collections import OrderedDict
    verify();sys.addaudithook(guard);torch.set_num_threads(2);torch.manual_seed(20261016);np.random.seed(20261016);rows=[];model=None;domain=None
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        if time.time()>=c.deadline():raise TimeoutError('night window before next TubeDETR probe')
        ds,row=source._validate_roster_record(rec)
        if ds!=domain:
            if model is not None:del model,adapter;gc.collect();torch.cuda.empty_cache()
            checkpoint=ROOT/read(BASE/'RUNTIME.json')['checkpoints'][ds]['path'];adapter=TubeAdapter(checkpoint);model=adapter.model;domain=ds
        frames,ids,_=frames_for(ds,row,OrderedDict());assert list(ids)==rec['frame_ids'];rt.add_repo_to_path(adapter.repo)
        from datasets.video_transforms import make_video_transforms
        from models.postprocessors import PostProcessSTVG
        video,_=make_video_transforms('val',cautious=True,resolution=224)(frames,None);caption=row['input']['caption'];duration=int(video.shape[1]);assert duration==len(ids)
        norm=model.transformer.decoder.layers[-1].norm3;params=list(norm.parameters());assert len(params)==2;initial=[v.detach().clone() for v in params];names=[n for n,p in model.named_parameters() if any(p is x for x in params)];assert len(names)==2
        def full():return rt.forward_video(model,video,caption,repo=adapter.repo,stride=2,device='cuda',use_bf16=True)
        def cached(memory):return rt.decode_video(model,memory,duration=duration,caption=caption,device='cuda',use_bf16=True)
        def exact(a,b):return all(torch.equal(a[k],b[k]) for k in ('pred_boxes','pred_sted'))
        tick=time.perf_counter()
        with torch.no_grad():
            memory=rt.encode_video(model,video,caption,repo=adapter.repo,stride=2,device='cuda',use_bf16=True);before=cached(memory);fresh=full();assert exact(before,fresh)
            # Zero parameter update must be exactly identity on all native frames.
            for p,x in zip(params,initial):p.copy_(x+torch.zeros_like(x))
            zero=cached(memory);assert exact(before,zero)
        for p in params:p.requires_grad_(True)
        pred=cached(memory);target=before['pred_boxes'].detach().float()+.01;loss=((pred['pred_boxes'].float()-target)**2).mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);assert sum(float(p.grad.double().square().sum()) for p in params)>0
        with torch.no_grad():
            # Engineering-only step: fixed LR, one proposal, no GT and no chosen
            # Teacher evidence. It does not represent the scientific B updater.
            for p in params:p.add_(p.grad,alpha=-1.)
            assert any(not torch.equal(p,x) for p,x in zip(params,initial));updated=cached(memory);updated_full=full();assert exact(updated,updated_full)
            changed=not exact(before,updated)
            for p,x in zip(params,initial):p.copy_(x)
            rollback=cached(memory);rollback_full=full();assert exact(rollback,before) and exact(rollback,rollback_full)
        for p in params:p.requires_grad_(False);p.grad=None
        boxes=before['pred_boxes'].detach().float().cpu().numpy();assert boxes.shape==(len(ids),4) and np.isfinite(boxes).all() and ((boxes>=0)&(boxes<=1)).all();interval=PostProcessSTVG()(before,frames_id=[ids],video_ids=[0])[0];assert len(interval)==2 and interval[0]<interval[1]
        result=dict(status='pass',case_id=rec['probe_alias'],dataset=ds,source_checkpoint_sha256=read(BASE/'RUNTIME.json')['checkpoints'][ds]['sha256'],native_frames=len(ids),native_stride=2,real_offsets=1,trainable_parameter_names=names,trainable_parameter_count=sum(p.numel() for p in params),encoder_cache_before_trainable_decoder_LN=True,cached_full_exact=True,zero_update_exact=True,actual_nonzero_parameter_step=True,output_changed=changed,rollback_full_exact=True,native_coordinate_and_physical_time_interface=True,qualification_step_is_engineering_only=True,B_method_transfer_effect_not_run=True,wall_seconds=time.perf_counter()-tick,GT_read=False)
        put(BASE/'probes'/(rec['probe_alias']+'.json'),result);rows.append(result);write(BASE/'STATUS.json',dict(status='running',done=len(rows),total=8,count=8,GT_read=False),mutable=True);del memory,before,fresh,zero,pred,loss,updated,updated_full,rollback,rollback_full;gc.collect();torch.cuda.empty_cache()
    put(BASE/'QUALIFICATION_COMPLETION.json',dict(status='pass_native_interface_only',count=8,records=rows,actual_gpu_sources=8,actual_scientific_effect_comparisons=0,frozen_B_effect_adapter_still_requires_independent_root=True,GT_read=False));put(BASE/'NEXT_ACTION.json',dict(status='requires_root',scope='if confirmation passes, qualify frozen-B native TubeDETR effect adapter then one finite matched effect study; do not substitute this engineering probe for transfer performance',GT_read=False))
if __name__=='__main__':worker()
