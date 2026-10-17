"""Readout-only continuation for two absent Source prefix caches, no refits."""
from pathlib import Path
import gc,hashlib,inspect,os,sys,subprocess,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_hypothesis256_v1 import BASE,OLD,CORE,read,write,sha,verify as original_verify
REC=BASE/'recovery/missing_prefix002';UNIT='stvg-temporal-hypothesis256-002.service'
def verify():
    original_verify()
    for p,h in read(REC/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
def init():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    original_verify();REC.mkdir(parents=True,exist_ok=False);rr=read(BASE/'ROSTER_LOCK.json')['records'];old=set(read(BASE/'REUSE_LOCK.json')['case_ids']);fits={f['case_id']:f for f in read(BASE/'FIT_BARRIER.json')['files']};targets={f['case_id']:f for f in read(BASE/'TARGET_BARRIER.json')['files']};completed=[];missing=[]
    for r in rr:
        cid=r['probe_alias'];p=BASE/'predictions'/(cid+'.pt')
        if cid not in old and not (ROOT/'artifacts/stvg_pipeline_speed_v1/source_features'/(cid+'.receipt.json')).exists():missing.append(cid)
        if p.exists():
            c,_=load_case(r);d=torch.load(p,map_location='cpu',weights_only=False);assert d['source_case_sha256']==c['source_case_sha256'] and d['fit_sha256']==fits[cid]['sha256'] and d['target_sha256']==targets[cid]['sha256'];assert d['actual_full_native_prefix_both_decoder_temporal_head_readout'] and d['frozen_spatial_boxes_exact'];assert np.array_equal(d['arms']['HYP_STUDENT']['boxes'],c['native_boxes_xyxy_normalized']);completed.append(dict(case_id=cid,path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    assert len(missing)==2 and len(completed)==33
    write(REC/'RECOVERY_LOCK.json',dict(status='locked',count=256,missing_cache_cases=missing,preserved_complete_readouts=completed,Teacher_and_fits_repeated=0,scientific_parameters_changed=False,original_failure_preserved=True,original_deadline_epoch=read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch'],GT_read=False))
    pins={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str((REC/'RECOVERY_LOCK.json').relative_to(ROOT)):sha(REC/'RECOVERY_LOCK.json')};pins.update(read(BASE/'EXECUTION_RUNTIME.json')['pins']);pins.update({str((BASE/n).relative_to(ROOT)):sha(BASE/n) for n in ['INPUT_BARRIER.json','TEACHER_BARRIER.json','FIT_BARRIER.json','TARGET_BARRIER.json','FAILURE.json']});pins.update({f['path']:f['sha256'] for f in completed});write(REC/'EXECUTION_RUNTIME.json',dict(status='locked',pins=pins,GT_read=False))
def readout():
    import numpy as np,torch
    from scripts import stvg_temporal_semantic_grounder32_v1 as old
    from scripts.stvg_temporal_semantic_grounder32_v2 import clean
    from scripts import stvg_factorial_native256_v1 as rgb
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    verify();clean();sys.addaudithook(old.guard);rr=read(BASE/'ROSTER_LOCK.json');lock=read(REC/'RECOVERY_LOCK.json');done={f['case_id'] for f in lock['preserved_complete_readouts']};oldids=set(read(BASE/'REUSE_LOCK.json')['case_ids']);op={f['case_id']:f for f in read(OLD/'PRIMARY_GLOBAL_PREDICTION_BARRIER.json')['files']};fresh=[];rgb.BASE=REC
    def cached(model,c):
        if c['case_id'] not in lock['missing_cache_cases']:return old.cached_source(model,c)
        from methods.decota_final_simplified_v1.objectives import prediction
        req=read(CORE/'temporal/requests'/(c['case_id']+'.json'));rgb.CURRENT_CASE=c;pixels=rgb.native_rgb(req,set(map(int,c['frame_ids'])));frames=np.stack([pixels[int(i)] for i in c['frame_ids']]);assert hashlib.sha256(frames.tobytes()).hexdigest()==c['pixel_sha256'];r=next(r for r in rr['records'] if r['probe_alias']==c['case_id']);row=old.row_for(r);assert model._fixed_full_state_hash==c['source_model_state_sha256']
        with torch.inference_mode():_,records,base=frozen_forward(model,frames,row)
        assert records==c['records'] and torch.equal(base.zero['boxes'].cpu(),c['zero']['boxes']);assert all(torch.equal(z.cpu(),v) for z,v in zip(base.zero['logits'],c['zero']['logits']));assert all(torch.equal(z[-1,0].cpu(),v) for z,v in zip(base.temporal_inputs,c['temporal_hidden']));p=prediction(base.zero['logits'],base.zero['boxes'],records,list(map(int,c['frame_ids'])));assert p['physical_interval']==list(c['Native_interval']);f=dict(case_id=c['case_id'],status='pass',fresh_frozen_prefix=True,RGB_Source_logits_hidden_boxes_records_and_Native_exact=True,source_case_sha256=c['source_case_sha256'],GT_read=False);write(REC/'fresh_prefix'/(c['case_id']+'.json'),f);fresh.append(f);del frames,pixels;return base
    def read_new(p):
        if Path(p)==BASE/'ROSTER_LOCK.json':return dict(rr,records=[r for r in rr['records'] if r['probe_alias'] not in oldids|done])
        return read(p)
    def write_full(p,d,**kw):
        if Path(p).name=='PRIMARY_GLOBAL_PREDICTION_BARRIER.json':
            d['files']+=lock['preserved_complete_readouts']+[dict(op[r['probe_alias']],reused_prior32=True) for r in rr['records'] if r['probe_alias'] in oldids];assert len(d['files'])==256 and len({f['case_id'] for f in d['files']})==256
        if 'count' in d:d['count']=256
        if 'total' in d:d['total']=191
        return write(p,d,**kw)
    src=inspect.getsource(old.readout).replace("['SEM','DIRECT','PARA']","['HYP']");ns=dict(old.__dict__);ns.update(BASE=BASE,verify=verify,read=read_new,write=write_full,cached_source=cached);exec(compile(src,__file__,'exec'),ns);ns['readout']();assert len(fresh)==2;write(REC/'PREFIX_RECONSTRUCTION_COMPLETION.json',dict(status='pass',count=2,records=fresh,Teacher_and_fit_repeated=0,GT_read=False));write(BASE/'PRIMARY_COMPLETION.json',dict(status='complete',count=256,old32_reused=32,preserved_new33=33,remaining_new191=191,GT_read=False))
def main():
    from scripts import launch_stvg_instance_identity_v1 as resource
    from scripts.continue_stvg_semantic_preference_4anchor32_v1 import props
    verify();resource.BASE=REC;resource.UNIT=UNIT;resource.STAGE='readout_recovery002';g=2**30;p=props(UNIT,'MemoryHigh','MemoryMax','MemorySwapMax');assert [int(p[k]) for k in ['MemoryHigh','MemoryMax','MemorySwapMax']]==[16*g,24*g,g]
    def trigger(v):
        if v['current']>=23*g:return 'own_RAM_23GiB'
        if v['anon']>=21*g:return 'own_nonreclaimable_RAM_21GiB'
        if v['available']<6*g:return 'host_RAM_headroom_6GiB'
        if v['own_gpu_active'] and v['gpu_free']<4*g:return 'GPU_headroom_4GiB'
        if v['disk_free']<30*g:return 'disk_floor_30GiB'
    resource.trigger=trigger;v=resource.sample();assert v['available']>=8*g and v['gpu_free']>=26*g and v['disk_free']>=30*g;write(REC/'ACTUAL_LIVE_RESOURCE_LIMITS.json',dict(status='pass',properties=p,admission=v,GT_read=False));stop=threading.Event();threading.Thread(target=resource.monitor,args=(stop,),daemon=True).start()
    try:
        for label,gt in [('readout_recovery002',False),('score',True)]:
            verify();write(BASE/'STATUS.json',dict(status='running',stage=label,count=256,GT_read=gt),mutable=True);folder=BASE/'execution'/label;folder.mkdir(parents=True,exist_ok=False);py=ROOT/'.conda/tubedetr/bin/python';cmd=[str(py),'-B',str(ROOT/'scripts'/('score_stvg_temporal_hypothesis256_v1.py' if gt else Path(__file__).name))]+([] if gt else ['readout']);env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (folder/'worker.log').open('xb') as log:
                t=time.perf_counter();child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);write(folder/'LAUNCH.json',dict(status='running',pid=child.pid,GT_read=gt));rc=child.wait(timeout=max(1,read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']-time.time()))
            write(folder/'WORKER_EXIT.json',dict(status='exited',exit_code=rc,wall_seconds=time.perf_counter()-t,GT_read=gt));assert rc==0,label
        write(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready',count=256,GT_read=False));write(BASE/'COMPLETION.json',dict(status='complete',count=256,GT_read=True));write(BASE/'STATUS.json',dict(status='measurement_complete_pending_root_publication',count=256,GT_read=False),mutable=True)
    except BaseException as e:write(REC/'FAILURE.json',dict(status='technical_failure',error=str(e),traceback=traceback.format_exc(),GT_read=False));write(BASE/'STATUS.json',dict(status='failed',stage='recovery002',count=256,GT_read=False),mutable=True);raise
    finally:stop.set()
if __name__=='__main__':globals()[sys.argv[1] if len(sys.argv)>1 else 'main']()
