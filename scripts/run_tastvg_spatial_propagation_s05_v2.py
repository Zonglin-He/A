"""S0.5 cached-only propagation and native suffix worker; no GT access."""
import sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT as S0,OLD,CONDS,observation
OUT=ROOT/'artifacts/tastvg_spatial_propagation_s05_v2'

def prepare():
    old=read(S0/'LOCK.json');paths=['protocols/tastvg_spatial_propagation_s05_v2.md','vg_tta/tastvg_spatial_propagation_s05_v2.py','vg_tta/tastvg_spatial_expansion_s0_v1.py','vg_tta/tastvg_evidence_capture_v1.py','vg_tta/tastvg_causal_round2_v1.py','scripts/run_tastvg_spatial_propagation_s05_v2.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=old['rows'],conditions=CONDS,pins={f:sha(ROOT/f) for f in paths},prior_barriers={n:sha(S0/n) for n in ['LOCK.json','H_BARRIER.json','EXPERT_BARRIER.json','PREDICTION_BARRIER.json','ROWS.json']},steps=3,relative_step=.004,cap_seconds=1800,GT_read=False,new_expert_calls=0,supersedes="threshold_v1_completed_before_new_user_definition",time=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in p['prior_barriers'].items():assert sha(S0/f)==h,f
    return p

def run():
    tick=time.monotonic();p=verify();done=0;state='failed';failure=None;lease=None
    try:
        import numpy as np,torch
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.tastvg_spatial_propagation_s05_v2 import propagate
        from vg_tta.tastvg_spatial_propagation_s05_v2 import expand
        from vg_tta.tastvg_temporal_fourarm_v1 import reinsert
        from vg_tta.exact_frame_decode_audit_v2 import decode
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        # Deny the retained-label file as well as the clean loader's annotation guard.
        def no_labels(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes)) and 'GT_SUBSET' in str(args[0]):raise PermissionError('S0.5 prediction worker forbids GT')
        sys.addaudithook(no_labels);install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        model=model_load('hcstvg1_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
        hb=read(S0/'H_BARRIER.json');eb=read(S0/'EXPERT_BARRIER.json');rein=[]
        for row in p['rows']:
            for cond in CONDS:
                assert time.monotonic()-tick<1780 and shutil.disk_usage(ROOT).free>8*2**30
                name=f"{row['ordinal']:03}.pt";f=OUT/'ta'/cond/name;assert not f.exists()
                hr=f'capture/{cond}/{name}';er=f'expert/{cond}/{name}';assert sha(S0/hr)==hb['files'][hr] and sha(S0/er)==eb['files'][er]
                data=load(S0/hr);expert=load(S0/er);assert data['pixel_sha256']==expert['pixel_sha256']
                start=time.monotonic();dense=propagate(data,expert);save(OUT/'dense'/cond/name,dense)
                data=device_tree(data,'cuda');result,fields,ev=expand(model,data,dense)
                if cond in ['clean','frame_drop_5'] and cond not in [v['condition'] for v in rein] and result['diagnostics'][-1]['relative_to_native']>0:
                    frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,_=observation(row,cond,frames)
                    result['reinsertion']=reinsert(model,shifted,row,fields,ev);rein.append(dict(parent=row['ordinal'],condition=cond,relative_to_native=result['diagnostics'][-1]['relative_to_native'],**result['reinsertion']));del frames,shifted
                result.update(parent=row['ordinal'],condition=cond,pixel_sha256=data['pixel_sha256'],model_state_sha256=mh,GT_read=False,seconds=time.monotonic()-start,propagation=dense['diagnostics'])
                save(f,result);done+=1;print(done,96,cond,'dense',dense['diagnostics']['propagated_frames'],'loss',result['diagnostics'][-1]['loss_after'],flush=True)
                status(OUT/'STATUS.json',dict(status='running',done=done,total=96));del data,expert,dense,result,fields,ev;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh;verify();assert len(rein)==2
        write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',cells=rein))
        write(OUT/'PREDICTION_BARRIER.json',dict(cells=96,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'ta').rglob('*.pt')},dense_files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'dense').rglob('*.pt')},GT_read=False,new_expert_calls=0,time=time.time()));state='completed'
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        receipt=dict(status=state,done=done,seconds=time.monotonic()-tick,failure=failure,new_expert_calls=0,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'STATUS.json',receipt)
        if lease:lease.close()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
