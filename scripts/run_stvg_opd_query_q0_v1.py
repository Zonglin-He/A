"""Actual serial Q0 GPU qualification/formal and bounded CPU root readback."""
import collections, fcntl, gc, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PUB,PYTHON,read,write,status,sha,save,load,verify


def gpu_lease():
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    pause=read(BASE/'PRIORITY_PAUSE_RECEIPT.json')
    for identity in pause['must_remain_stopped']:
        p=process(identity['pid']);assert p['start_ticks']==identity['start_ticks'] and p['state']=='T'
    f=(BASE/'priority_GPU.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch,numpy as np
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.decota_paper_common_v1 import guard
    install_clean_loader();sys.addaudithook(guard)
    torch.set_num_threads(2);torch.manual_seed(20260920);np.random.seed(20260920)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write(BASE/f'{os.getpid()}_GPU_LEASE.json',dict(status='one_active_Q0_GPU_worker_originals_stopped',
          pid=os.getpid(),GT_read=False,time=time.time()))
    return f


def capture(model,r,old,cache):
    import torch
    from scripts.run_decota_paper_main_v1 import read_row,frames_for,unpack_expert
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from scripts.stvg_opd_paper_inputs_v1 import observation
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from methods.decota_final_simplified_v1.objectives import prediction
    row=read_row(r['dataset'],r['query_ordinal']);frames,ids,reuse=frames_for(r['dataset'],row,cache)
    shifted,pixel,spec=observation(r['dataset'],row,'clean',frames)
    batch,records,base=frozen_forward(model,shifted,row)
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    assert sha(ROOT/r['input']['path'])==r['input']['sha256']
    inp=normalized(load(ROOT/r['input']['path']))
    assert inp['pixel_sha256']==pixel and inp['corruption_spec']==spec
    assert inp['source_model_state_sha256']==model._fixed_full_state_hash
    assert inp['frame_ids']==ids and inp['indices']==native['indices']
    assert inp['interval']==old['interval']==native['physical_interval']
    assert torch.equal(inp['native_boxes'],native['boxes'].cpu())
    del batch,frames,shifted
    return base,row,unpack_expert(inp['expert'])


def run(qualification):
    verify()
    import torch,numpy as np
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_query_q0_v1 import episode
    design=read(BASE/'DESIGN_LOCK.json');records=design['records']
    if qualification:records=[r for r in records if r['qualify']]
    else:assert read(BASE/'ROOT_QUALIFICATION.json')['status']=='pass'
    dest=BASE/('qualification' if qualification else 'formal');dest.mkdir(exist_ok=True)
    lease=gpu_lease();model=None;dataset=None;cache=collections.OrderedDict();files={};start=time.time()
    try:
        for r in records:
            assert sha(ROOT/r['original_Full']['path'])==r['original_Full']['sha256']
            old=load(ROOT/r['original_Full']['path'])
            if dataset!=r['dataset']:
                if model is not None:
                    assert state_hash(model.state_dict())==mh;del model;gc.collect();torch.cuda.empty_cache()
                torch.manual_seed(20260920);np.random.seed(20260920)
                model=model_for(r['source']);mh=state_hash(model.state_dict());dataset=r['dataset'];cache.clear()
            base,row,expert=capture(model,r,old,cache)
            z=episode(base,old,expert,r['dataset'],row['key'],design['pooled_output_RMS_budget'])
            evidence=dict(**r,episode=z,runtime_sha256=sha(BASE/'RUNTIME.json'),source_model_sha256=mh,
                          interval=old['interval'],GT_read=False)
            p=dest/f"{r['index']:05}.pt";save(dest/f"UNACCEPTED_{r['index']:05}.pt",evidence)
            counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            if qualification:
                again=episode(base,old,expert,r['dataset'],row['key'],design['pooled_output_RMS_budget'])
                rp=dest/f"REPEAT_{r['index']:05}.pt";save(rp,again)
                equal(z,again,counts);evidence['repeat']=dict(sha256=sha(rp),complete_bitwise=counts)
            elif r['qualify']:
                q=load(BASE/'qualification'/f"{r['index']:05}.pt")
                equal(z,q['episode'],counts);evidence['qualified_formal']=counts
            save(p,evidence);write(p.with_suffix('.json'),dict(sha256=sha(p),bytes=p.stat().st_size,GT_read=False,time=time.time()))
            files[str(p.relative_to(ROOT))]=sha(p)
            status(BASE/'GPU_STATUS.json',dict(status='running_Q0_qualification' if qualification else 'running_Q0_formal',
                  dataset=dataset,done=len(files),total=len(records),query_ordinal=r['query_ordinal'],pid=os.getpid(),GT_read=False,time=time.time()))
            print('Q0_PROGRESS',qualification,dataset,len(files),len(records),flush=True)
            base.restore(base.initial);del base,row,expert,z,evidence,old;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh
        write(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'),dict(status='pass' if qualification else 'sealed',
              files=files,cells=len(records),all_three_groups_and_true_mean_decodes=True,GT_read=False,
              new_DINO_calls=0,optimizer_steps=0,source_state_restored=True,elapsed_seconds=time.time()-start,time=time.time()))
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache();lease.close()


def root(qualification=False):
    import torch,numpy as np
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.stvg_opd_query_q0_common_v1 import spectrum,canonical,weights,GROUPS
    barrier=read(BASE/('QUALIFICATION_BARRIER.json' if qualification else 'GLOBAL_PREDICTION_BARRIER.json'))
    assert barrier['status']==('pass' if qualification else 'sealed') and barrier['cells']==(4 if qualification else 64)
    checks=0;pairs=0;directions=0;mean_decodes=0;statecoords=0
    for p,h in barrier['files'].items():
        assert sha(ROOT/p)==h;z=load(ROOT/p);r=z;ep=z['episode']
        assert not z['GT_read'] and sha(ROOT/r['input']['path'])==r['input']['sha256']
        assert sha(ROOT/r['original_Full']['path'])==r['original_Full']['sha256']
        old=load(ROOT/r['original_Full']['path']);assert torch.equal(ep['before'],old['fit']['before'])
        assert all(torch.equal(ep['initial'][n],v) and torch.equal(ep['restored'][n],v) for n,v in old['fit']['initial'].items())
        assert ep['optimizer_steps']==0 and ep['actual_new_DINO_calls']==0 and ep['no_candidate_writeback']
        statecoords+=1792
        for label in ['Query','LN']:
            ss=spectrum(ep['jacobian'][label]);assert ss['rank']==ep['SVD'][label]['rank']
            assert np.array_equal(ss['directions'],ep['SVD'][label]['directions'])
        for group in GROUPS:
            g=ep['groups'][group];w=weights(g['rewards'],ep['tau']);assert np.array_equal(w,g['weights'])
            delta=sum(w[i]*c['delta'].astype(float) for i,c in enumerate(g['candidates'])).astype(np.float32)
            assert np.array_equal(delta,g['mean_delta'])
            assert np.argmax(g['rewards'])==g['expert_top1_index']
            for c in g['candidates']:
                before=ep['before'].numpy().astype(float);boxes=c['boxes'].numpy().astype(float)
                d=float(np.sqrt(np.mean((boxes-before)**2)));assert abs(d-c['output_RMS'])<2e-12 and d<=ep['output_RMS_budget']+1e-12
                for t in c['probes']:
                    d=float(np.sqrt(np.mean((t['boxes'].numpy().astype(float)-before)**2)));assert abs(d-t['output_RMS'])<2e-12;checks+=1
                directions+=c['sign']!=0
            mean_decodes+=1;checks+=len(g['candidates'])+3
        counts=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
        if qualification:
            rp=Path(p).parent/f"REPEAT_{z['index']:05}.pt";assert sha(ROOT/rp)==z['repeat']['sha256'];equal(ep,load(ROOT/rp),counts);pairs+=1
        elif z['qualify']:
            q=load(BASE/'qualification'/f"{z['index']:05}.pt");equal(ep,q['episode'],counts);pairs+=1
    write(BASE/('ROOT_QUALIFICATION.json' if qualification else 'ROOT_DEPLOYMENT_READBACK.json'),dict(status='pass',
          scope='complete saved frozen states/actual candidates/rewards/SVD/true mean readouts and repeat/formal pairing; not independent complete model Jacobian replay',
          cells=len(barrier['files']),complete_pairs=pairs,scalar_checks=checks,signed_directions=directions,
          true_parameter_mean_decodes=mean_decodes,state_coordinates=statecoords,
          CPU_only=True,GT_read=False,task_complete=False,time=time.time()))


if __name__=='__main__':
    job=sys.argv[1]
    try:
        if job in ['qualify','formal']:run(job=='qualify')
        elif job in ['qualification_root','deployment_root']:root(job=='qualification_root')
        else:raise ValueError(job)
    except BaseException:
        p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
        status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',stage=job,evidence=str(p.relative_to(ROOT)),time=time.time()));raise
