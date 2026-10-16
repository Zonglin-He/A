"""Matched physical-input and sparse-observation capture for later paper stages."""
import hashlib,time
import numpy as np
from scripts.stvg_opd_paper_common_v1 import *

def observation(dataset,row,condition,frames):
    from vg_tta.tastvg_deployment_corruption_v2 import burst_spec,apply_burst
    if condition=='clean':return frames,hashlib.sha256(frames.tobytes()).hexdigest(),None
    family,coverage=condition.rsplit('_',1);coverage=float(coverage)
    assert family in FAMILIES and coverage in [2.5,5.,10.]
    spec=burst_spec(row['source'],row['input']['frame_count'],row['frame_ids'],coverage)
    if family=='frame_freeze':
        shifted=frames.copy();positions=spec['positions']
        if positions:
            donor=max(0,spec['physical_start']-1)
            if donor in row['frame_ids']:image=frames[row['frame_ids'].index(donor)]
            else:
                if dataset=='hc2':from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
                else:from vg_tta.exact_frame_decode_audit_v2 import decode
                image=decode({**row['input'],'frame_ids':[donor]})[0][0]
            shifted[positions]=image
        spec={**spec,'freeze_donor_dataset_decoder':dataset}
    else:shifted=apply_burst(frames,row['input'],spec,family,row['source'])
    return shifted,hashlib.sha256(shifted.tobytes()).hexdigest(),spec

def capture(model,expert,stage,row,condition,cache):
    import torch
    from scripts.run_decota_paper_main_v1 import frames_for,pack_expert,unpack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from methods.decota_final_simplified_v1.objectives import prediction
    from vg_tta.stvg_opd_paper_budget_v1 import budget_observations
    from scripts.decota_paper_common_v1 import load_npz
    ds=stage['dataset'];source=stage['source'];k=stage['observation_budget']
    tick=time.perf_counter();frames,ids,reuse=frames_for(ds,row,cache)
    shifted,pixel,spec=observation(ds,row,condition,frames);decode_seconds=time.perf_counter()-tick
    torch.cuda.synchronize();begin=time.perf_counter()
    batch,records,base=frozen_forward(model,shifted,row)
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    torch.cuda.synchronize();forward_seconds=time.perf_counter()-begin
    f=BASE/'matched_inputs'/f'{source}_to_{ds}'/f'K{k}'/condition/f'{row["ordinal"]:05}.pt'
    newcalls=0;oldrc=None
    if f.exists():
        rc=read(f.with_suffix('.json'));assert sha(f)==rc['sha256'];inp=load(f)
        assert inp['pixel_sha256']==pixel and inp['source_model_state_sha256']==model._fixed_full_state_hash
        assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and inp['interval']==native['physical_interval'] and inp['observation_budget']==k
        ex=unpack_expert(inp['expert'])
    else:
        oldjob='t1_ours_hc2' if ds=='hc2' else 't1_ours_vid'
        oldpath=PAPER/oldjob/'inputs'/'clean'/f'{row["ordinal"]:05}.npz'
        cross=source==('vidstg' if ds=='hc2' else 'hcstvg2')
        if cross and k==4 and condition=='clean' and oldpath.exists():
            a,md,oldrc=load_npz(oldpath)
            assert md['source_model_state_sha256']==model._fixed_full_state_hash
            assert md['pixel_sha256']==pixel and md['interval']==native['physical_interval']
            assert torch.equal(native['boxes'].cpu(),torch.from_numpy(a['native_boxes']))
            ex=unpack_expert(md['expert'])
        else:
            ex=budget_observations(expert,row['parses'],shifted,ids,native['indices'],budget=k,audit=True)
            newcalls=ex['new_DINO']
        inp=dict(dataset=ds,source=source,query_ordinal=row['ordinal'],pixel_sha256=pixel,frame_ids=ids,
            native_boxes=native['boxes'].detach().cpu(),interval=native['physical_interval'],indices=native['indices'],
            expert=pack_expert(ex),corruption_spec=spec,observation_budget=k,
            source_model_state_sha256=model._fixed_full_state_hash,
            old_cache_sha256=None if oldrc is None else oldrc['sha256'],GT_read=False)
        save(f,inp);rc=dict(sha256=sha(f),bytes=f.stat().st_size,GT_read=False,time=time.time())
        write(f.with_suffix('.json'),rc)
    expert_forward_seconds=sum(float(v['receipt']['forward_seconds']) for v in ex['observations'].values())
    del frames,shifted,batch
    return base,native,ex,dict(path=str(f.relative_to(BASE)),sha256=sha(f),new_DINO_calls=newcalls,
        nominal_observation_budget=k,eligible_actual_observations=len(ex['observations']),decoded_frame_reuse=reuse,
        decode_corruption_seconds=decode_seconds,STVG_frozen_forward_seconds=forward_seconds,
        recorded_expert_forward_seconds=expert_forward_seconds,expert_forward_reused=newcalls==0)
