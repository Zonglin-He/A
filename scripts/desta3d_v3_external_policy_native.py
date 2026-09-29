"""Fixed Dev16 four native observation policies; wrong controls only after locked gate."""
import sys,os,time,json,subprocess,gc,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_provider import D,P,GAP,OUT,read,write,sha,check_pins,local_dependencies,allocation,total_prior,tensor_sha
from scripts.desta3d_v3_a05_signal import setup,A0,ADAPTER_SHA,B1,B1_SHA,load
from scripts.desta3d_v3_joint_learnability import equal
from vg_tta.external_qualification_io import seal,verify_seal
TEACHER=OUT/'extgate_evidence_parse_v3'

def register(stage):
    done,sealed=verify_seal(TEACHER);assert done['queries']==16 and sealed['predictions']==16;assert read(TEACHER/'ROOT_EVIDENCE_READBACK.json')['status']=='passed'
    run=OUT/('extgate_'+stage);assert not run.exists();rows=read(D/'INPUTS.json');arms=['B1','T','S','TS']
    if stage=='wrong001':
        report=read(D/'evaluation/REPORT.json');assert report['preliminary_pass'];arms=['TS_wrong']+[b+'_wrong' for b,k in [('T','tIoU'),('S','sIoU')] if report['comparisons'][b][k]['mean_delta_pp']>0]
    cfg={**read(D/'CONFIG.json'),'phase_seconds':1800,'arms':arms,'stage':stage,'teacher':str(TEACHER),'backward_steps':0}
    write(run/'CONFIG.json',cfg);write(run/'INPUTS.json',rows)
    paths=local_dependencies([Path(__file__),P,B1,TEACHER/'ROOT_EVIDENCE_READBACK.json',TEACHER/'PREDICTIONS_SEAL.json',D/'INPUTS.json',run/'CONFIG.json',run/'INPUTS.json',ROOT/'scripts/score_desta3d_v3_external_policy_gate.py',ROOT/'vg_tta/desta3d_v3_policy_gate_views.py',ROOT/'tests/test_desta3d_v3_policy_gate_views.py',ROOT/'tests/test_desta3d_v3_external_policy_gate.py',ROOT/'scripts/crosscheck_desta3d_v3_external_policy_gate.py',TEACHER/'LOCK.json',D/'NATIVE_CPU_PREFLIGHT.json',ROOT/'protocols/desta3d_v3_external_policy_parser_repair_v3.md'])
    reuse={}
    for r in rows:
        old=GAP/'episodes'/f"{r['gap_index']:04}"/'B1.pt';assert sha(old)==read(GAP/'PREDICTIONS_SEAL.json')['files'][str(old.relative_to(GAP))];reuse[str(old)]=sha(old)
        old_trace=old.parent/'BASE_TRACE.pt';assert sha(old_trace)==read(GAP/'PREDICTIONS_SEAL.json')['files'][str(old_trace.relative_to(GAP))];reuse[str(old_trace)]=sha(old_trace)
    assert sha(B1)==B1_SHA;write(run/'REUSE.json',reuse);pins={str(p):sha(p) for p in paths};pins.update(reuse)
    write(run/'LOCK.json',dict(pins=pins));write(run/'REGISTRATION.json',dict(status='registered_before_GPU',time=time.time(),source_GT_worker=False,arms=arms));print('NATIVE REGISTERED',stage,arms,flush=True)

def worker(stage):
    import torch,hashlib
    from scripts.ptd_spatial_adapter_ab_v1 import inputs_for
    from vg_tta.desta3d_v2_ptd import capture_stock_fields
    from vg_tta.desta3d_v3_actuation_full_vocab import capture_teacher
    from vg_tta.desta3d_v3_decomposition import shared_fields
    from vg_tta.desta3d_v3_policy_gate_views import build_views
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from scripts.desta3d_v2_source_fit import prediction_record
    from scripts.desta3d_v2_reference_audit_cached_v3 import details
    from scripts.desta3d_v2_p0 import adapter_sha256
    from vg_tta.desta3d_v2_prediction_contract import validate_prediction
    run=OUT/('extgate_'+stage)
    with allocation(run) as (cfg,guard):
        pr,model,adapter=setup(cfg['seed']);verify_seal(TEACHER)
        for i,row in enumerate(read(run/'INPUTS.json')):
            guard();ep=run/'episodes'/f'{i:02}';assert not ep.exists();frames,ids=decode(row['input']);te=TEACHER/'episodes'/f'{i:02}'
            ev=read(te/'EVIDENCE.json');ti=read(te/'INPUT.json');pixel=hashlib.sha256(frames.tobytes()).hexdigest();assert ti['key']==row['key'] and ti['frame_ids']==ids and ti['pixel_sha256']==pixel
            views,meta=build_views(frames,ids,ev,cfg['dim'],cfg['blur_radius']);write(ep/'INPUT.json',dict(key=row['key'],physical_pixel_sha=pixel,evidence_sha=sha(te/'EVIDENCE.json'),view_metadata=meta,view_pixel_sha={a:hashlib.sha256(v.tobytes()).hexdigest() for a,v in views.items()}))
            orig=load(GAP/'episodes'/f"{row['gap_index']:04}"/'B1.pt');grid=None
            for arm in cfg['arms']:
                guard();prompt,pre=inputs_for(row,pr,views[arm]);fields=capture_stock_fields(model,pr,prompt,row['input']['caption'],ids,row['input']['fps'])
                support={k:tensor_sha(v) for k,v in fields.items() if isinstance(v,torch.Tensor)};fixed={k:tensor_sha(prompt[k]) for k in ['input_ids','video_grid_thw']};fixed['frame_times']=tensor_sha(fields['frame_times'])
                if grid is None:grid=fixed
                else:assert grid==fixed
                stock=fields['visual_grid'].detach()
                with torch.no_grad(),shared_fields(adapter,{'event':stock,'spatial':stock},['event','spatial'],allow_prefix=True):
                    result,trace=capture_teacher(model,pr,prompt,adapter,fields)
                pred=prediction_record(result,row,pre,ADAPTER_SHA);pred.update(arm=arm,readout=details(result),support=support,physical_support=fixed,GT_read=False,decoder_GT_prefix=False,optimizer_steps=0,target_read=False,evidence_sha=sha(te/'EVIDENCE.json'),native_time_logits=trace['branches'][0]['logits'].get('time') if trace['branches'] else None)
                torch.save(pred,ep/(arm+'.pt'));validate_prediction(pred,len(ids))
                if arm=='B1':
                    keys=['key','source','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess','adapter_sha','event_completion','spatial_completion','event_logits','readout','support']
                    checks={k:equal(pred[k],orig[k]) for k in keys};old_trace=load(GAP/'episodes'/f"{row['gap_index']:04}"/'BASE_TRACE.pt');checks['native_time_logits']=equal(pred['native_time_logits'],old_trace['branches'][0]['logits'].get('time'));del old_trace;write(ep/'BASELINE_REPLAY.json',checks);assert all(checks.values()),checks
                assert adapter_sha256(adapter)==ADAPTER_SHA and all(not p.requires_grad and p.grad is None for mod in (model,adapter) for p in mod.parameters())
                del prompt,fields,result,trace,pred,stock;gc.collect();torch.cuda.empty_cache()
            write(ep/'COMPLETE.json',dict(index=i,arms=cfg['arms'],frozen_scope=True));print('POLICY',stage,i+1,16,flush=True)
        write(run/'COMPLETE.json',dict(status='completed_unscored',queries=16,parents=16,predictions=16*len(cfg['arms']),seal_sha=seal(run,16*len(cfg['arms'])),source_GT_worker=False,peak_GPU_bytes=torch.cuda.max_memory_allocated()))

def launch(stage):
    run=OUT/('extgate_'+stage);assert not (run/'STARTED.json').exists();start=time.monotonic();cmd=[sys.executable,'-B',str(Path(__file__)),'worker','--stage',stage]
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (D/(stage+'.log')).open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);(D/'ACTIVE.json').write_text(json.dumps(dict(status='running',stage=stage,pid=p.pid,wrapper_pid=os.getpid(),time=time.time(),cmd=cmd))+'\n');code=p.wait()
    wall=time.monotonic()-start;r=read(run/'RECEIPT.json') if (run/'RECEIPT.json').exists() else dict(seconds=0);prior=total_prior();seconds=max(0,wall-r['seconds']);write(OUT/('extgate_'+stage+'_wrapper')/'RECEIPT.json',dict(status='completed' if code==0 else 'failed',seconds=seconds,worker_seconds=r['seconds'],prior_seconds=prior,cumulative_seconds=prior+seconds,cap=None));write(run/'EXIT.json',dict(code=code));(D/'ACTIVE.json').write_text(json.dumps(dict(status='completed_pending_audit' if code==0 else 'failed',stage=stage,time=time.time()))+'\n');assert code==0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','launch','worker']);p.add_argument('--stage',choices=['policy001','wrong001'],default='policy001');a=p.parse_args();globals()[a.action](a.stage)
