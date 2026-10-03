"""Complete omitted U post-tubes from immutable first-step states; no learning/GT."""
import os,sys,time,functools,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_oracle_event5_common_v1 import *
def run(ds):
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    from scripts.run_final_simplification_v1 import lease
    from vg_tta.tastvg_native_spatial_rollout_s05_v1 import predict
    from vg_tta.tastvg_ur_write_decomposition_v1 import sgd
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_tastvg_paper48_p5_online_v1 import compact_prediction
    verified();install_clean_loader();sys.addaudithook(guard);h=lease();tick=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from scripts.run_spatial_regression_alignment_v1 import model_load
    model=model_load('hcstvg1_test' if ds=='vidstg' else 'vidstg_test').eval().requires_grad_(False)
    bar=read(POOL/ds/'CAPTURE_BARRIER.json');mh=state_hash(model.state_dict());assert mh==bar['checkpoint_state_sha256']
    @functools.lru_cache(maxsize=12)
    def cached(parent,cond):
        f=POOL/ds/'capture'/cond/f'{parent:05}.json';assert sha(f)==bar['files'][str(f.relative_to(POOL/ds))]
        r=read(f);cf=POOL/ds/r['cache'];assert sha(cf)==r['sha256'];return load(cf),r
    jobs=[c for c in read(BASE/'EVENT_PLAN.json')['events'] if c['dataset']==ds and old_a(c)['update_steps'][0].get('post_prediction') is None]
    assert len(jobs)==48 and all(c['split']=='confirm' for c in jobs);out=BASE/ds/'cached_U_readout';controls=0
    for done,c in enumerate(jobs,1):
        budget();f=out/'predictions'/f'{prefix(c)}.pt'
        if f.with_suffix('.json').exists():checked(f);continue
        a=old_a(c);x=cached_payload(c);step=a['update_steps'][0];st=step['post_state']
        expected=sgd(x['pre_state'],x['gradients']['U'],plan(ds)['params']['lr']) if x['rewards']['U'] is not None else x['pre_state']
        assert state_hash(st)==state_hash(expected)==step['post_state_sha256']
        data,r=cached(c['parent'],c['condition']);assert r['pixel_sha256']==c['pixel_sha256'];data=device_tree(data,'cuda')
        if controls<2:
            _,_,pre=predict(model,data,device_tree(x['pre_state'],'cuda'))
            assert torch.equal(pre['boxes'],a['slow']['boxes']) and pre['indices']==a['slow']['indices'];controls+=1
        _,_,cp=predict(model,data,device_tree(st,'cuda'));assert state_hash(model.state_dict())==mh
        commit(f,dict(cell=c,prediction=compact_prediction(cp),cached_first_post_state_sha256=state_hash(st),
            cached_gradient_arithmetic_bitwise=True,no_learning=True,raw_GT_read=False,GT_assisted_observation=False))
        status(out/'STATUS.json',dict(status='running',done=done,total=48,worker_pid=os.getpid(),raw_GT_read=False,time=time.time()))
        print('CACHED_U_SUFFIX',ds,done,48,flush=True);del data;gc.collect();torch.cuda.empty_cache()
    verified();write(out/'PREDICTION_BARRIER.json',dict(status='sealed',cells=48,no_learning=True,raw_GT_read=False,
        model_restored=True,files={str(f.relative_to(out)):sha(f) for f in (out/'predictions').glob('*.json')},time=time.time()))
    write(out/'RESOURCES.json',dict(suffix_replays=48+controls,backwards=0,new_expert_calls=0,new_backbone_calls=0,
        omitted_cached_U_post_tubes=48,live_pre_state_controls=controls,worker_wall_seconds=time.monotonic()-tick,time=time.time()))
    status(out/'STATUS.json',dict(status='completed',done=48,total=48,time=time.time()));h.close()
if __name__=='__main__':
    try:run(sys.argv[1])
    except BaseException as e:
        status(BASE/'CACHED_U_FAILURE.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
