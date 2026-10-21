"""Finite existing-query numerical/timing regression; no labels or experts."""
import sys,json,time,gc,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import prepare,read,sha,atomic,savept,OLD
from scripts.trajectory_opd.evidence import TrajectoryEvidence as Dense
from scripts.trajectory_fast.evidence import TrajectoryEvidence as Sparse
from scripts.trajectory_opd.policy import fit as old_fit
from scripts.trajectory_fast.policy import fit as new_fit


def main():
    _,records,items,oldlock=prepare();cfg=read(ROOT/'scripts/trajectory_fast/config.json')
    assert cfg['forward_budget']==256
    from stvg.runtime import setup,lease,load_source
    setup();torch.set_num_threads(4);leasefile=lease();outdir=ROOT/'outputs/trajectory_fast_checks/GPU'
    outdir.mkdir(parents=True,exist_ok=True)
    assert not (outdir/'completed.json').exists(),'Do not repeat completed regression'
    model=load_source(read(OLD/'config.json')['datasets']['hc2']).cpu().eval().requires_grad_(False)
    model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
    rows=[];start=time.perf_counter()
    try:
        for rec,it in zip(records[:2],items[:2]):
            key=rec['id'];cp=OLD/'student_cache'/f'{key}.pt';mp=OLD/'research/native_masks'/it['mask_path']
            assert sha(cp)==it['cache_sha256'] and sha(mp)==it['native_mask_sha256']
            cache=torch.load(cp,map_location='cpu',weights_only=False);pack=torch.load(mp,map_location='cpu',weights_only=False)
            dense=Dense(pack,cache,device='cuda');sparse=Sparse(pack,cache,device='cuda')
            for attr in ('targets','confidence','W'):assert torch.equal(getattr(dense,attr),getattr(sparse,attr))
            E=torch.as_tensor(cache['E'],device='cuda',dtype=torch.float64)
            oldp=ROOT/'outputs/trajectory_opd_hc64/predictions'/f'{key}.npz'
            with np.load(oldp)as a:
                bb=torch.as_tensor(np.stack([a[n]for n in ('TrajectoryOPD','Direct','DirectMulti','DirectMC')]),device='cuda',dtype=torch.float64)
            boxes=bb.repeat(8,1,1).clone().requires_grad_(True)
            oldU=torch.stack([dense.utility(b,E)['U']for b in boxes]);oldg=torch.autograd.grad(oldU.sum(),boxes)[0]
            newU=sparse.utility_batch(boxes,E)['U'];newg=torch.autograd.grad(newU.sum(),boxes)[0]
            reward_error=float((oldU-newU).abs().max());gradient_error=float((oldg-newg).abs().max())
            assert reward_error<=1e-12 and gradient_error<=1e-10
            timings={}
            with torch.no_grad():
                batch=boxes.detach()
                funcs=dict(dense_scalar=lambda:torch.stack([dense.utility(b,E)['U']for b in batch]),sparse_scalar=lambda:torch.stack([sparse.utility(b,E)['U']for b in batch]),sparse_batch=lambda:sparse.utility_batch(batch,E)['U'])
                for name,fn in funcs.items():
                    for _ in range(2):fn()
                    torch.cuda.synchronize();t=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                    for _ in range(10):fn()
                    b.record();torch.cuda.synchronize()
                    timings[name]=dict(seconds_per32=(time.perf_counter()-t)/10,CUDA_stream_seconds_per32=a.elapsed_time(b)/10000)
            comparisons={}
            for arm in cfg['arms']:
                before=old_fit(model,cache,dense,it['key'],arm,cfg)
                after=new_fit(model,cache,sparse,it['key'],arm,cfg)
                assert before['best_component']==after['best_component']
                assert len(before['rounds'])==len(after['rounds'])
                assert [r['accepted_components']for r in before['rounds']]==[r['accepted_components']for r in after['rounds']]
                for k in ('full_trajectory_evals','decoder_backwards','updates','accepted_updates','setup_full_trajectory_evals'):
                    assert before['costs'][k]==after['costs'][k],(arm,k)
                zerr=float((before['final_z']-after['final_z']).abs().max());berr=float((before['final']-after['final']).abs().max())
                assert zerr<=2e-5 and berr<=1e-6,(arm,zerr,berr)
                comparisons[arm]=dict(state_max_abs_error=zerr,boxes_max_abs_error=berr,same_final_mode=True,same_all_acceptance_decisions=True,same_callcounts=True,old_seconds=before['costs']['seconds'],new_seconds=after['costs']['seconds'],old_evals=before['costs']['full_trajectory_evals'],new_evals=after['costs']['full_trajectory_evals'],profiling=after['profiling'])
                savept(outdir/f'{key}_{arm}.pt',dict(old=before,new=after))
                print(json.dumps(dict(query=len(rows)+1,arm=arm,regression=comparisons[arm])),flush=True)
                del before,after;gc.collect();torch.cuda.empty_cache()
            rows.append(dict(query=len(rows)+1,reward_max_abs_error=reward_error,gradient_max_abs_error=gradient_error,repeated_reward_timings=timings,arms=comparisons))
            del sparse,dense,boxes,cache,pack;gc.collect();torch.cuda.empty_cache()
        assert sha(ROOT/'outputs/formal/state.json')==oldlock['formal_state_sha256']
        atomic(outdir/'completed.json',dict(status='pass',queries=2,all_same_acceptance_and_final_modes=True,rows=rows,seconds=time.perf_counter()-start,GT_read=False,new_encoder_expert_forwards=0,scope='Two existing queries; same256 budget dense/scalar vs sparse/batched reward. Timings are incremental regression, not final full-panel deployment cost.'))
    finally:
        del model;gc.collect();torch.cuda.empty_cache();leasefile.close()


if __name__=='__main__':main()
