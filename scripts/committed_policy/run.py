"""Immutable cached HC64 real adaptation. Inference never imports a scorer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import read,sha,atomic,savept
from scripts.evidence_policy.run import validate as validate_previous,TRIAL as PREVIOUS
from scripts.box_propagation.run import TRIAL as BANKS
from scripts.box_propagation.bank import CandidateBank
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/committed_policy_hc64'
STAGE=ROOT/'outputs/propagation_v0_hc64/stage1'
BASE_NAMES=('ERefStrong','GeometryOPD','BoxFixed','B_OPD')
NEW_NAMES=('CommitInitial','CommitDP','CommitOPD','CommitFixedKD','CommitDirect')
NAMES=BASE_NAMES+NEW_NAMES

def lock_inputs():
    old=read(PREVIOUS/'lock.json');validate_previous(PREVIOUS,old)
    cfg=read(HERE/'config.json');assert cfg['queries']==64 and cfg['inner_steps']==10
    paths=sorted([p for p in HERE.glob('*.py')if not p.name.startswith('test_')]+[HERE/'config.json'])
    sources={**old['sources'],**{str(p.relative_to(ROOT)):sha(p)for p in paths}}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    assert sha(STAGE/'sealed.json')==read(BANKS/'lock.json')['old_seal_sha256']
    stage_seal=read(STAGE/'sealed.json');assert stage_seal['GT_read']is False and stage_seal['queries']==64 and set(stage_seal['files'])==set(old['ids'])
    inputs=[]
    for item in old['inputs']:
        k=item['id'];r=read(STAGE/'predictions'/f'{k}.json');p=STAGE/r['evidence_path']
        assert sha(STAGE/'predictions'/f'{k}.json')==stage_seal['files'][k]['receipt_sha256']
        assert r['GT_read']is False and sha(p)==r['evidence_sha256']
        for field in ('cache_sha256','native_mask_sha256','reference_sha256'):assert r[field]==item[field]
        inputs.append({**item,'evidence_path':str(p.relative_to(ROOT)),'evidence_sha256':sha(p),'evidence_receipt_sha256':sha(STAGE/'predictions'/f'{k}.json')})
    ident=dict(config=cfg,sources=sources,previous_lock_sha256=sha(PREVIOUS/'lock.json'),previous_seal_sha256=sha(PREVIOUS/'sealed.json'),
        stage_seal_sha256=sha(STAGE/'sealed.json'),formal_state_sha256=old['formal_state_sha256'],registered_config_sha256=old['registered_config_sha256'],
        manifest_sha256=old['manifest_sha256'],inputs=inputs,ids=old['ids'],GT_read=False)
    L={**ident,'fingerprint':hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()}
    if(TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==L,'Immutable trial changed'
    else:
        atomic(TRIAL/'lock.json',L)
        for rel in sources:
            dest=TRIAL/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    return cfg,L

def validate_one(item,L):
    k=item['id'];r=read(TRIAL/'receipts'/f'{k}.json');p=TRIAL/'predictions'/f'{k}.npz'
    assert r['fingerprint']==L['fingerprint'] and r['GT_read']is False and sha(p)==r['sha256']
    oldr=read(BANKS/'receipts'/f'{k}.json');assert r['bank_sha256']==oldr['bank_sha256']==sha(BANKS/oldr['bank_path'])
    assert r['reference_prediction_sha256']==sha(PREVIOUS/'predictions'/f'{k}.npz')
    assert sha(ROOT/item['evidence_path'])==item['evidence_sha256']==r['evidence_sha256']
    assert sha(STAGE/'predictions'/f'{k}.json')==item['evidence_receipt_sha256']
    for a,b in [('trace_path','trace_sha256'),('extra_path','extra_sha256')]:assert sha(TRIAL/r[a])==r[b]
    bank=CandidateBank.from_state(torch.load(BANKS/oldr['bank_path'],map_location='cpu',weights_only=False))
    trace=torch.load(TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
    extra=torch.load(TRIAL/r['extra_path'],map_location='cpu',weights_only=False)
    assert extra['GT_read']is False
    with np.load(p,allow_pickle=False)as f,np.load(PREVIOUS/'predictions'/f'{k}.npz',allow_pickle=False)as prev:
        assert set(f.files)==set(NAMES)|{'frame_ids','positions8','Strong_interval'}
        for n in BASE_NAMES+('frame_ids','positions8','Strong_interval'):assert np.array_equal(f[n],prev[n])
        for n in NEW_NAMES:
            out=trace[n];assert np.array_equal(f[n],out['final'].numpy())
            assert torch.equal(out['final'][~bank.active],bank.E[~bank.active])
            assert ((bank.boxes==out['final'][:,None]).all(-1)&bank.valid).any(-1).all()
            if n in ('CommitInitial','CommitDP'):continue
            c=out['costs'];assert c['updates']==c['backward']==30 and len(out['rounds'])==3
            assert c['accepted_outer']+c['rejected_outer']==3 and out['initialtheta'].shape==(4,) and out['initialtheta'].eq(0).all()
            assert c['heavy_forwards']==c['decoder_calls']==c['expert_calls']==0
            assert torch.equal(out['trace']['initial_rollout']['boxes'],trace['CommitInitial']['final'])
            for row in out['rounds']:
                assert len(row['inner'])==10 and row['after_score']>=row['before_score']-1e-12
                if row['rolled_back']:assert torch.equal(row['before'],row['after']) and row['optimizer_before_sha256']==row['optimizer_after_sha256']
    return r

def validate(trial=TRIAL,lock=None):
    assert Path(trial).resolve()==TRIAL.resolve();L=read(TRIAL/'lock.json')if lock is None else lock;s=read(TRIAL/'sealed.json')
    assert s['queries']==64 and s['GT_read']is False and s['fingerprint']==L['fingerprint'] and s['lock_sha256']==sha(TRIAL/'lock.json')
    assert set(s['files'])==set(L['ids'])
    for p,h in L['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(PREVIOUS/'lock.json')==L['previous_lock_sha256'] and sha(PREVIOUS/'sealed.json')==L['previous_seal_sha256']
    assert sha(STAGE/'sealed.json')==L['stage_seal_sha256']
    assert sha(ROOT/'outputs/formal/state.json')==L['formal_state_sha256'] and sha(ROOT/'configs/default.json')==L['registered_config_sha256']
    assert sha(ROOT/'data/evaluation/development.jsonl')==L['manifest_sha256']
    receipts={}
    for item in L['inputs']:
        k=item['id'];r=validate_one(item,L);e=s['files'][k]
        assert sha(TRIAL/e['path'])==e['sha256'] and sha(TRIAL/e['receipt_path'])==e['receipt_sha256'];receipts[k]=r
    assert not torch.cuda.is_initialized();return receipts

def main(run=False):
    cfg,L=lock_inputs()
    if not run:print(json.dumps(dict(status='prepared_not_run',fingerprint=L['fingerprint'])));return
    assert not(TRIAL/'run.json').exists() and not(TRIAL/'sealed.json').exists(),'No overwrite or implicit restart'
    from scripts.committed_policy.policy import fit,rollout,exact_dp
    from scripts.committed_policy.evidence import build
    from scripts.propagation_opd.run import check_idle
    check_idle();torch.set_num_threads(1);files={};begin=time.perf_counter()
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),fingerprint=L['fingerprint'],GT_read=False))
    try:
        for i,item in enumerate(L['inputs']):
            k=item['id'];t0=time.perf_counter();oldr=read(BANKS/'receipts'/f'{k}.json')
            bank=CandidateBank.from_state(torch.load(BANKS/oldr['bank_path'],map_location='cpu',weights_only=False))
            ev=torch.load(ROOT/item['evidence_path'],map_location='cpu',weights_only=False)
            t=time.perf_counter();extra=build(bank,ev);evidence_seconds=time.perf_counter()-t
            ep=TRIAL/'evidence'/f'{k}.pt';savept(ep,extra)
            with np.load(PREVIOUS/'predictions'/f'{k}.npz',allow_pickle=False)as f:result={n:f[n].copy()for n in BASE_NAMES+('frame_ids','positions8','Strong_interval')}
            with torch.no_grad():path=rollout(bank,extra,torch.zeros(4,dtype=torch.float64),cfg)
            trace={'CommitInitial':dict(final=path['boxes'],trace=path),'CommitDP':exact_dp(bank,extra,cfg)}
            for n in ('CommitInitial','CommitDP'):result[n]=trace[n]['final'].numpy()
            costs={}
            for arm in ('OPD','FixedKD','Direct'):
                name='Commit'+arm;out=fit(bank,extra,item['key'],arm,cfg);trace[name]=out;result[name]=out['final'].numpy();costs[name]=out['costs']
                print(json.dumps(dict(completed=i,total=64,arm=name,seconds=out['costs']['seconds'],backward=out['costs']['backward'])),flush=True)
            tp=TRIAL/'traces'/f'{k}.pt';savept(tp,trace);dest=TRIAL/'predictions'/f'{k}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.with_suffix('.tmp').open('wb')as f:np.savez(f,**result)
            os.replace(dest.with_suffix('.tmp'),dest)
            rr=dict(fingerprint=L['fingerprint'],sha256=sha(dest),bank_path=oldr['bank_path'],bank_source='outputs/box_propagation_hc64',bank_sha256=oldr['bank_sha256'],
                reference_prediction_sha256=sha(PREVIOUS/'predictions'/f'{k}.npz'),evidence_sha256=item['evidence_sha256'],extra_path=str(ep.relative_to(TRIAL)),extra_sha256=sha(ep),
                trace_path=str(tp.relative_to(TRIAL)),trace_sha256=sha(tp),costs=costs,timings=dict(evidence_seconds=evidence_seconds),total_seconds=time.perf_counter()-t0,GT_read=False,model_calls=0,expert_calls=0,decoder_calls=0,RGB_reads=0)
            rp=TRIAL/'receipts'/f'{k}.json';atomic(rp,rr);validate_one(item,L)
            files[k]=dict(path=str(dest.relative_to(TRIAL)),sha256=sha(dest),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp))
            atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,seconds=time.perf_counter()-begin,GT_read=False));print(json.dumps(dict(completed=i+1,total=64,seconds=time.perf_counter()-begin)),flush=True)
            del bank,trace,result,extra,ev;gc.collect()
        atomic(TRIAL/'sealed.json',dict(queries=64,fingerprint=L['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),files=files,GT_read=False));validate()
        atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,fingerprint=L['fingerprint'],seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),fingerprint=L['fingerprint'],traceback=traceback.format_exc(),GT_read=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();main(a.run)
