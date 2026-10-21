"""A/B evidence-interface and four-node feedback on immutable sealed banks.

Only cached lightweight policy calculations. No RGB, teacher, backbone, decoder,
GT or scoring modules are read by inference. All six arms seal before scoring.
"""
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
from scripts.box_propagation.run import validate as validate_banks,TRIAL as BANKS
from scripts.box_propagation.bank import CandidateBank
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/evidence_policy_hc64'
NEW_NAMES=tuple(v+'_'+a for v in ('A','B')for a in ('OPD','FixedKD','Direct'))
BASE_NAMES=('ERefStrong','GeometryOPD','BoxFixed','BoxOPD','BoxFixedKD','BoxDirect')
NAMES=BASE_NAMES+NEW_NAMES

def lock_inputs():
    old=read(BANKS/'lock.json');validate_banks(BANKS,old)
    cfg=read(HERE/'config.json');assert cfg['inner_steps']==10 and cfg['rounds']==3 and cfg['paths']==2 and cfg['queries']==64
    paths=[p for p in HERE.glob('*.py')if not p.name.startswith('test_')]+[HERE/'config.json']
    sources={**old['sources'],**{str(p.relative_to(ROOT)):sha(p)for p in paths}}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    ident=dict(config=cfg,sources=sources,bank_source_lock_sha256=sha(BANKS/'lock.json'),bank_source_seal_sha256=sha(BANKS/'sealed.json'),bank_source_fingerprint=old['fingerprint'],formal_state_sha256=old['formal_state_sha256'],registered_config_sha256=old['registered_config_sha256'],manifest_sha256=old['manifest_sha256'],inputs=old['inputs'],ids=old['ids'],GT_read=False)
    L={**ident,'fingerprint':hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()}
    if(TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==L,'Immutable A/B experiment changed'
    else:
        atomic(TRIAL/'lock.json',L)
        for p in paths:
            dest=TRIAL/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    return cfg,L

def validate_one(item,L):
    k=item['id'];r=read(TRIAL/'receipts'/f'{k}.json');p=TRIAL/'predictions'/f'{k}.npz'
    assert r['fingerprint']==L['fingerprint'] and r['GT_read'] is False and sha(p)==r['sha256']
    oldr=read(BANKS/'receipts'/f'{k}.json');assert r['bank_sha256']==oldr['bank_sha256']==sha(BANKS/oldr['bank_path'])
    assert r['reference_prediction_sha256']==oldr['sha256']==sha(BANKS/'predictions'/f'{k}.npz')
    assert sha(TRIAL/r['trace_path'])==r['trace_sha256']
    bank=CandidateBank.from_state(torch.load(BANKS/oldr['bank_path'],map_location='cpu',weights_only=False))
    trace=torch.load(TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
    with np.load(p,allow_pickle=False)as f,np.load(BANKS/'predictions'/f'{k}.npz',allow_pickle=False)as prev:
        assert set(f.files)==set(NAMES)|{'frame_ids','positions8','Strong_interval'}
        for name in BASE_NAMES+('frame_ids','positions8','Strong_interval'):assert np.array_equal(f[name],prev[name])
        assert np.array_equal(f['ERefStrong'],bank.E.numpy())
        for name in NEW_NAMES:
            out=trace[name];c=out['costs'];assert out['initialtheta'].shape==(8,) and out['initialtheta'].eq(0).all()
            assert 1<=c['updates']<=30 and c['backward']==c['updates'] and len(out['rounds'])==c['updates']
            assert c['heavy_forwards']==c['decoder_calls']==0
            assert np.array_equal(f[name],out['final'].numpy()) and torch.equal(out['final'][~bank.active],bank.E[~bank.active])
            assert ((bank.boxes==out['final'][:,None]).all(-1)&bank.valid).any(-1).all()
            for row in out['rounds']:
                assert row['after_score']>=row['before_score']-1e-12
                if row['rolled_back']:assert torch.equal(row['before'],row['after']) and row['optimizer_before_sha256']==row['optimizer_after_sha256']
    return r

def validate(trial=TRIAL,lock=None):
    assert Path(trial).resolve()==TRIAL.resolve();L=read(TRIAL/'lock.json')if lock is None else lock;s=read(TRIAL/'sealed.json')
    assert s['queries']==64 and s['GT_read'] is False and s['fingerprint']==L['fingerprint'] and s['lock_sha256']==sha(TRIAL/'lock.json')
    assert set(s['files'])==set(L['ids'])
    for p,h in L['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(BANKS/'lock.json')==L['bank_source_lock_sha256'] and sha(BANKS/'sealed.json')==L['bank_source_seal_sha256']
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
    from scripts.evidence_policy.policy import fit
    from scripts.propagation_opd.run import check_idle
    check_idle();torch.set_num_threads(1);files={};begin=time.perf_counter()
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),fingerprint=L['fingerprint'],GT_read=False))
    try:
        for i,item in enumerate(L['inputs']):
            k=item['id'];t0=time.perf_counter();oldr=read(BANKS/'receipts'/f'{k}.json')
            bank=CandidateBank.from_state(torch.load(BANKS/oldr['bank_path'],map_location='cpu',weights_only=False))
            result={};trace={};costs={}
            with np.load(BANKS/'predictions'/f'{k}.npz',allow_pickle=False)as f:
                result={n:f[n].copy()for n in BASE_NAMES+('frame_ids','positions8','Strong_interval')}
            for v in ('A','B'):
                for arm in ('OPD','FixedKD','Direct'):
                    name=v+'_'+arm;t=time.perf_counter();out=fit(bank,item['key'],arm,cfg,variant=v)
                    result[name]=out['final'].numpy();trace[name]=out;costs[name]=out['costs']
                    print(json.dumps(dict(completed=i,total=64,arm=name,seconds=time.perf_counter()-t,updates=out['costs']['updates'])),flush=True)
            tp=TRIAL/'traces'/f'{k}.pt';savept(tp,trace);dest=TRIAL/'predictions'/f'{k}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.with_suffix('.tmp').open('wb')as f:np.savez(f,**result)
            os.replace(dest.with_suffix('.tmp'),dest)
            rr=dict(fingerprint=L['fingerprint'],sha256=sha(dest),bank_path=oldr['bank_path'],bank_source='outputs/box_propagation_hc64',bank_sha256=oldr['bank_sha256'],reference_prediction_sha256=oldr['sha256'],trace_path=str(tp.relative_to(TRIAL)),trace_sha256=sha(tp),costs=costs,total_seconds=time.perf_counter()-t0,GT_read=False,model_calls=0,expert_calls=0,decoder_calls=0,RGB_reads=0)
            rp=TRIAL/'receipts'/f'{k}.json';atomic(rp,rr);validate_one(item,L)
            files[k]=dict(path=str(dest.relative_to(TRIAL)),sha256=sha(dest),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp));atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,seconds=time.perf_counter()-begin,GT_read=False));print(json.dumps(dict(completed=i+1,total=64,seconds=time.perf_counter()-begin)),flush=True)
            del bank,trace,result;gc.collect()
        atomic(TRIAL/'sealed.json',dict(queries=64,fingerprint=L['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),files=files,GT_read=False));validate()
        atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,fingerprint=L['fingerprint'],seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),fingerprint=L['fingerprint'],traceback=traceback.format_exc(),GT_read=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();main(a.run)
