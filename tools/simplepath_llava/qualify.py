"""Two existing-query real checks; no labels, no new expert and no retries."""
import fcntl
import gc
import json
from pathlib import Path
import numpy as np
import torch
from stvg.runtime import ROOT,sha,setup
from stvg.tube_runner import write,digest
from tools.reuse_cache.catalog import Catalog

PUBLIC=ROOT/'results/simplepath_llava';OUT=ROOT/'outputs/simplepath_llava';PLAN=PUBLIC/'qualification_plan.json'


def checked():
    p=json.loads(PLAN.read_text())
    for name,h in {**p['code_sha256'],**p['upstream_code_sha256']}.items():assert sha(ROOT/name)==h,name
    assert sha(ROOT/'runtime/llava_st/checkpoint_seal.json')==p['checkpoint_seal_sha256']
    return p


def run():
    setup();p=checked();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'failure.json').exists():raise ValueError('Prior failure; no automatic retry')
    write(OUT/'status.json',dict(status='qualification_running',plan_sha256=sha(PLAN)))
    from tools.llava_st.backbone import load
    from stvg.tube_decode import decode
    from stvg.tensors import state_hash
    from .adapter import capture,finish
    cat=Catalog(ROOT/'runtime/reuse_cache/catalog.sqlite');checks=[]
    with (ROOT/'runtime/gpu.lock').open('a') as lock:
        print('Waiting for shared gpu.lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        model,tok,proc,audit=load();before=state_hash(model.state_dict())
        for ds in p['dataset_order']:
            d=p['datasets'][ds];assert sha(ROOT/d['manifest'])==d['manifest_sha256']
            row=json.loads((ROOT/d['manifest']).read_text().splitlines()[0]);frames,ids=decode(row['metadata'],ds)
            print(ds,'cold Source +99 true conditional starts',flush=True)
            cold=capture(model,tok,proc,frames,ids,row['metadata']['caption'])
            oldroot=ROOT/'outputs/llava_st_development'/ds/'full';seal=json.loads((oldroot/'sealed.json').read_text());e=seal['files'][row['id']]
            assert sha(oldroot/e['path'])==e['sha256']
            with np.load(oldroot/e['path'],allow_pickle=False) as old:
                for key in ['Source','Native_interval','frame_ids','positions8']:assert np.array_equal(old[key],np.asarray(cold[key])),(ds,key,'Source drift')
            lookup={t:i for i,t in enumerate(ids)};rgb=frames[[lookup[cold['frame_ids'][j]] for j in cold['positions8']]]
            teacher,trace=cat.teacher('llava_st',rgb,row['metadata']['caption'],p['teacher_fingerprint'])
            value=finish(cold,teacher,d['configuration'])
            print(ds,'cached SigLIP Source +99 true conditional starts',flush=True)
            cached=capture(model,tok,proc,frames,ids,row['metadata']['caption'],prefix=cold['visual_prefix'])
            for key in ['Source','features','valid_cells','native_prior','conditional_end_log_prob','start_log_prob','separator_log_prob']:
                assert np.array_equal(np.asarray(cold[key]),np.asarray(cached[key])),(ds,key,'cold/cache drift')
            replay=finish(cached,teacher,d['configuration'])
            for key in value:
                if isinstance(value[key],np.ndarray):assert np.array_equal(value[key],replay[key]),(ds,key,'Full drift')
            target=OUT/'qualification'/ds;target.mkdir(parents=True,exist_ok=True)
            arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
            assert all(np.isfinite(a).all() for a in arrays.values())
            np.savez_compressed(target/'output.npz',**arrays)
            checks.append(dict(dataset=ds,previous_query_used=True,native_generations=2,
                actual_conditional_start_branches=198,conditional_path_max_abs_error=cold['conditional_native_path_max_abs_error'],
                true_conditional_dependence=True,cold_cache_all_bitexact=True,Source_exact_old=True,
                actual_local_shape=list(cold['features'].shape),native_prior_sum=float(cold['native_prior'].sum()),
                native_prior_positive_intervals=int((cold['native_prior']>0).sum()),
                all_five_outputs_finite=True,output_bytes=(target/'output.npz').stat().st_size,
                output_sha256=sha(target/'output.npz'),new_teacher_calls=0,teacher_cache_key=trace['cache_key'],GT_read=False))
            write(OUT/'worker_progress.json',dict(stage='qualification',dataset=ds,complete=len(checks),total=2,labels_read=False))
            print(ds,'real qualification PASS',flush=True)
            del cold,cached,value,replay,frames,teacher,arrays;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==before and all(not w.requires_grad and w.grad is None for w in model.parameters())
    cat.db.close()
    write(PUBLIC/'qualification.json',dict(status='pass',plan_sha256=sha(PLAN),checks=checks,
          native_generations=4,conditional_start_branches=396,new_teacher_calls=0,GT_reads=0,
          model_state_unchanged=True,model_load=audit,task_performance_evaluated=False))
    write(OUT/'status.json',dict(status='qualification_pass_pending_fixed_run_plan',plan_sha256=sha(PLAN)))


if __name__=='__main__':
    try:run()
    except Exception as error:
        write(OUT/'failure.json',dict(status='stopped',error=type(error).__name__+': '+str(error),automatic_retry=False));raise
