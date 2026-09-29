"""Serial, bounded Round2 controller: register -> causal seal -> GT subset -> oracle seal."""
import argparse,gc,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree,full_reinsert
BASE=ROOT/'artifacts/tastvg_evidence_vulnerability_v1/full64_v1'
OUT=ROOT/'artifacts/tastvg_causal_correctability_round2_v1'
ARMS=['OS','OS_PT','OT','OT_PS','OST','Oselective']
OWN=['protocols/tastvg_causal_correctability_round2_v1.md','vg_tta/tastvg_causal_round2_v1.py','scripts/run_tastvg_causal_round2_v1.py','tests/test_tastvg_causal_round2_v1.py']

def prepare():
    p=read(BASE/'LOCK.json');v=read(OUT/'VULNERABILITY_LOCK.json');assert v['protocol_sha256']==sha(ROOT/OWN[0])
    pins=dict(p['pins']);pins.update({f:sha(ROOT/f) for f in OWN})
    labels=read(ROOT/'artifacts/stvg_fullscale_diagnostics_v1/lock.json')
    cfg=dict(rho=.02,steps=10,alphas=[1.,.5,.25,.125],seed=20260929,cap_seconds=7200,minimum_free_bytes=8*2**30,maximum_new_bytes=30*2**30,arms=ARMS,source_loss_weights={'hcstvg1_test':[5,3,2],'vidstg_test':[5,4,10]})
    write(OUT/'LOCK.json',dict(rows=p['rows'],config=cfg,pins=pins,baseline_barrier_sha256=sha(BASE/'CAPTURE_BARRIER.json'),attack_barrier_sha256=sha(BASE/'ATTACK_BARRIER.json'),vulnerability_lock_sha256=sha(OUT/'VULNERABILITY_LOCK.json'),labels_path=labels['labels'],labels_sha256=labels['labels_sha256'],GT_values_read=False,time=time.time()))
    status(OUT/'STATUS.json',dict(status='registered',GT_read=False));print('REGISTERED64, PartA + seven main arms, fixed science')

def verify():
    p=read(OUT/'LOCK.json');pins=dict(p['pins'])
    for rev in sorted(OUT.glob('ENGINEERING_REVISION_*.json')):
        for name,z in read(rev)['files'].items():assert pins[name]==z['old_sha256'];pins[name]=z['new_sha256']
    for name,digest in pins.items():assert sha(ROOT/name)==digest,('pin',name)
    assert sha(BASE/'CAPTURE_BARRIER.json')==p['baseline_barrier_sha256']
    assert sha(BASE/'ATTACK_BARRIER.json')==p['attack_barrier_sha256']
    assert sha(OUT/'VULNERABILITY_LOCK.json')==p['vulnerability_lock_sha256']
    return p

def completed(f):
    if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];return True
    if f.exists():raise RuntimeError('Unreceipted artifact retained '+str(f))
    return False

def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),lock_sha256=sha(OUT/'LOCK.json'),time=time.time()))

def labels():
    import ijson
    p=verify();bar=read(OUT/'CAUSAL_BARRIER.json');assert len(bar['files'])==64
    for f,h in bar['files'].items():assert sha(OUT/f)==h
    assert sha(p['labels_path'])==p['labels_sha256'];keys={r['key'] for r in p['rows']};selected={}
    # Stream one existing container; only the authorized64 records are retained/used.
    with open(p['labels_path'],'rb') as f:
        for key,value in ijson.kvitems(f,'',use_float=True):
            if key in keys:selected[key]={k:value[k] for k in ('interval','boxes','valid','event_mask','missing_event_frames')}
    assert set(selected)==keys
    write(OUT/'GT_SUBSET.json',selected)
    write(OUT/'GT_EXPOSURE.json',dict(keys=sorted(keys),retained_records=64,container_sha256=p['labels_sha256'],container_streamed=True,other_records_retained_or_scored=False,GT_for_oracle_updates=True,GT_for_unlabeled_method=False,PartA_barrier_sha256=sha(OUT/'CAUSAL_BARRIER.json'),V_lock_sha256=sha(OUT/'VULNERABILITY_LOCK.json'),subset_sha256=sha(OUT/'GT_SUBSET.json'),time=time.time()))
    print('Opened authorized64 GT after PartA and V lock; subset sealed')

def run(stage,limit=0):
    start=time.monotonic();p=verify();cfg=p['config'];alloc=OUT/'allocations'/f'{time.time_ns()}.json';prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'));model=None;lease=None;done=0;state='failed';failure=None
    def guard():
        if prior+time.monotonic()-start>cfg['cap_seconds']-20:raise TimeoutError('Round2 cumulative process cap')
        if shutil.disk_usage(ROOT).free<cfg['minimum_free_bytes']:raise RuntimeError('8GiB disk floor')
    try:
        import numpy as np,torch
        from methods.decota_final_simplified_v1.tensors import state_hash
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from vg_tta.tastvg_causal_round2_v1 import causal,forward,make_targets,baseline_gradients,native_contract_audit,run_oracle,swapped_probe
        if stage=='oracle':
            exposure=read(OUT/'GT_EXPOSURE.json');assert sha(OUT/'GT_SUBSET.json')==exposure['subset_sha256'];gt=read(OUT/'GT_SUBSET.json')
        install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in line for line in procs.splitlines() if line.strip()),procs
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;cohort=None;mhash=None
        for row in p['rows']:
            if limit and done>=limit:break
            key=row['key'];stem=key.replace(':','_');f=OUT/stage/f'{stem}.pt'
            if completed(f):done+=1;continue
            guard();status(OUT/'STATUS.json',dict(stage=stage,status='running',done=done,key=key,pid=os.getpid(),seconds=prior+time.monotonic()-start))
            if cohort!=row['cohort']:
                if model is not None:assert state_hash(model.state_dict())==mhash;del model;gc.collect();torch.cuda.empty_cache()
                model=model_load(row['cohort']);cohort=row['cohort'];mhash=state_hash(model.state_dict())
                assert [model.cfg.SOLVER.BBOX_COEF,model.cfg.SOLVER.GIOU_COEF,model.cfg.SOLVER.TEMP_COEF]==cfg['source_loss_weights'][cohort]
            cp=BASE/'capture'/f'{stem}.pt';assert sha(cp)==read(cp.with_suffix('.json'))['sha256'];data=device_tree(load(cp),'cuda');assert data['model_state_sha256']==mhash;tick=time.monotonic()
            with torch.no_grad():
                ev,boxes,pred=forward(model,data,[v['H'] for v in data['views']]);assert torch.equal(pred['boxes'].cpu(),data['prediction']['boxes'].cpu());assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(pred['logits'],data['prediction']['logits']))
            if stage=='causal':
                ap=BASE/'attack'/f'{stem}_r0.02.pt';assert sha(ap)==read(ap.with_suffix('.json'))['sha256'];a=load(ap);result=causal(model,data,a);del a
            else:
                targets=make_targets(data,gt[key],'cuda');initial=baseline_gradients(model,data,targets);initial['target_exceptions']=targets['exceptions'];contract=None
                if row==next(r for r in p['rows'] if r['cohort']==cohort):contract=native_contract_audit(model,data,targets,load(row['input_path']))
                arm_results={}
                for arm in ARMS:
                    af=OUT/'arms'/f'{stem}_{arm}.pt'
                    if completed(af):arm_results[arm]=dict(path=str(af),sha256=sha(af));continue
                    status(OUT/'STATUS.json',dict(stage=stage,status='running',done=done,key=key,arm=arm,pid=os.getpid(),seconds=prior+time.monotonic()-start))
                    r=run_oracle(model,data,targets,arm,cfg,guard)
                    if row==next(r for r in p['rows'] if r['cohort']==cohort) and arm in ('OST','Oselective'):
                        r['reinsertion']=full_reinsert(model,load(row['input_path']),data,{'selected':{'delta':r['delta'],'evidence':r['evidence']}})
                    r.update(key=key,cohort=cohort,source=row['source'],all_parameters_frozen=all(not v.requires_grad and v.grad is None for v in model.parameters()),capture_sha256=sha(cp))
                    assert r['all_parameters_frozen'];commit(af,r);arm_results[arm]=dict(path=str(af),sha256=sha(af));print('ARM',done+1,64,key,arm,'accepted',sum(x['accepted_alpha']>0 for x in r['path']),flush=True);del r;gc.collect();torch.cuda.empty_cache()
                probes={task:swapped_probe(model,data,targets,initial['gradients'],task,cfg) for task in ('S','T')}
                result=dict(initial=initial,arms=arm_results,swapped_probes=probes,native_contract=contract,GT_oracle=True)
            result.update(key=key,cohort=cohort,source=row['source'],seconds=time.monotonic()-tick,capture_sha256=sha(cp));commit(f,result);done+=1;del result,data;gc.collect();torch.cuda.empty_cache();print('DONE',stage,done,64,key,round(time.monotonic()-tick,3),flush=True)
        if model is not None:assert state_hash(model.state_dict())==mhash
        verify();state='completed' if done==64 else 'smoke_complete'
        if done==64:
            files={str(f.relative_to(OUT)):sha(f) for f in (OUT/stage).glob('*.pt')};assert len(files)==64
            if stage=='oracle':
                arms={str(f.relative_to(OUT)):sha(f) for f in (OUT/'arms').glob('*.pt')};assert len(arms)==384;files.update(arms)
            b=OUT/(stage.upper()+'_BARRIER.json')
            if not b.exists():write(b,dict(files=files,queries=64,GT_oracle=stage=='oracle',time=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        seconds=time.monotonic()-start;write(alloc,dict(stage=stage,status=state,seconds=seconds,prior_seconds=prior,total_seconds=prior+seconds,done=done,failure=failure,pid=os.getpid(),time=time.time()));status(OUT/'STATUS.json',dict(stage=stage,status=state,done=done,total_seconds=prior+seconds,failure=failure,pid=os.getpid()))
        if lease:lease.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','causal','labels','oracle']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='labels':labels()
    else:run(a.stage,a.limit)
