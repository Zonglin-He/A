"""No-expert/no-GT parameter support screen on exact cached TA-STVG suffix."""
import sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.run_tastvg_spatial_expansion_s0_v1 import OUT as S0,OLD,CONDS,observation
OUT=ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1'

def prepare():
    old=read(S0/'LOCK.json');paths=['protocols/tastvg_native_spatial_rollout_s05_v1.md','vg_tta/tastvg_native_spatial_rollout_s05_v1.py','scripts/run_tastvg_native_spatial_rollout_s05_v1.py','vg_tta/tastvg_evidence_capture_v1.py','methods/decota_final_simplified_v1/backbone.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=old['rows'],conditions=CONDS,pins={f:sha(ROOT/f) for f in paths},prior_barriers={n:sha(S0/n) for n in ['LOCK.json','H_BARRIER.json']},old_capture_barrier=sha(OLD/'CAPTURE_BARRIER.json'),rho=.05,directions=4,seed=20260929,candidates=9,dimensions=1792,cap_seconds=1800,GT_read=False,expert_read=False,time=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in p['prior_barriers'].items():assert sha(S0/f)==h,f
    assert sha(OLD/'CAPTURE_BARRIER.json')==p['old_capture_barrier'];return p

def run():
    tick=time.monotonic();p=verify();done=0;state='failed';failure=None;lease=None
    try:
        import numpy as np,torch
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from methods.decota_final_simplified_v1.tensors import state_hash,detached
        from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state,rollout_states,predict,reinsert
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.box_stability_diagnostics_v1 import overlap
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def guard(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes)):
                path=str(args[0])
                if any(tag in path for tag in ['GT_SUBSET','/expert/','tastvg_spatial_propagation_s05_']):raise PermissionError('Native rollout forbids expert, propagation and GT files: '+path)
        sys.addaudithook(guard);install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        model=model_load('hcstvg1_test').eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
        center=central_state(model);states,config,basis=rollout_states(center,p['rho']);flat=lambda s:torch.cat([v.detach().cpu().double().flatten() for v in s.values()]);base=flat(center)
        ratios=[float((flat(s)-base).norm()/base.norm()) for s in states];assert ratios[0]==0 and all(abs(r-.05)<1e-7 for r in ratios[1:])
        save(OUT/'PARAMETER_SUPPORT.pt',dict(center=detached(center,'cpu'),states=detached(states,'cpu'),basis=basis,configuration=config));write(OUT/'PARAMETER_SUPPORT.json',dict(**config,actual_relative_norms=ratios,basis_orthogonality_error=float((basis@basis.T-torch.eye(4)).abs().max()),sha256=sha(OUT/'PARAMETER_SUPPORT.pt'),expert_read=False,GT_read=False))
        hb=read(S0/'H_BARRIER.json');rein=[]
        for row in p['rows']:
            for cond in CONDS:
                assert time.monotonic()-tick<1780 and shutil.disk_usage(ROOT).free>8*2**30
                name=f"{row['ordinal']:03}.pt";hr=f'capture/{cond}/{name}';assert sha(S0/hr)==hb['files'][hr]
                data=load(S0/hr);data=device_tree(data,'cuda');start=time.monotonic();trajectory=[];selfscores=[]
                for k,param in enumerate(states):
                    ev,boxes,pred=predict(model,data,param)
                    if k==0:
                        assert torch.equal(boxes,data['prediction']['boxes']);assert all(torch.equal(a,b.cpu()) for a,b in zip(pred['logits'],data['prediction']['logits']))
                    assert all(torch.equal(v,center[key]) for key,v in central_state(model).items())
                    if row['ordinal']==p['rows'][0]['ordinal'] and cond in ['clean','frame_drop_5'] and k==1:
                        frames,ids=decode(row['input']);assert ids==row['frame_ids'];shifted,_=observation(row,cond,frames)
                        result=reinsert(model,shifted,row,param,ev);rein.append(dict(parent=row['ordinal'],condition=cond,candidate=k,**result));del frames,shifted
                    trajectory.append(dict(prediction=detached(pred,'cpu')));selfscores.append(float(overlap(pred['boxes'],data['prediction']['boxes'].cpu()).mean()))
                    del ev,boxes,pred
                result=dict(parent=row['ordinal'],condition=cond,trajectory=trajectory,self_sIoU=selfscores,unique_tubes=len({z['prediction']['boxes'].numpy().tobytes() for z in trajectory}),pixel_sha256=data['pixel_sha256'],model_state_sha256=mh,GT_read=False,expert_read=False,H_unchanged=True,source_parameters_restored=True,seconds=time.monotonic()-start)
                save(OUT/'ta'/cond/name,result);done+=1;print(done,96,cond,'unique',result['unique_tubes'],'self_min',min(selfscores),flush=True);status(OUT/'STATUS.json',dict(status='running',done=done,total=96));del data,result,trajectory;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh;verify();assert len(rein)==2
        write(OUT/'REINSERTION_AUDIT.json',dict(status='pass',cells=rein));write(OUT/'PREDICTION_BARRIER.json',dict(cells=96,candidates=864,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'ta').rglob('*.pt')},parameter_support_sha256=sha(OUT/'PARAMETER_SUPPORT.pt'),GT_read=False,expert_read=False,time=time.time()));state='completed'
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        receipt=dict(status=state,done=done,seconds=time.monotonic()-tick,failure=failure,new_expert_calls=0,backward_calls=0,time=time.time());write(OUT/'allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'STATUS.json',receipt)
        if lease:lease.close()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
