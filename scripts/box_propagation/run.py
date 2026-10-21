"""Complete cached HC64 adaptation; immutable old/new banks, same v0 policy.

The user authorized this independent comparison after the old gate failed.
That gate and historical stage2 record are never changed or reused as training.
No GT, scorer, RGB, model, expert or decoder is imported or executed here.
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
from scripts.state_opd.run import read,sha,atomic,savept,OLD
from scripts.propagation_v0.run import validate_phase,TRIAL as V0
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/box_propagation_hc64'
ARMS=('OPD','FixedKD','Direct')
ALL_ARMS=tuple(p+a for p in ('Old','Box')for a in ARMS)
NAMES=('ERefStrong','GeometryOPD','OldFixed','OldOPD','OldFixedKD','OldDirect','BoxFixed','BoxOPD','BoxFixedKD','BoxDirect')

class SavedEvidence:
    def __init__(self,state):
        assert state['GT_read'] is False
        for k,v in state.items():setattr(self,k,v)
        self.positions=list(state['positions8']);self.height,self.width=state['seeds'].shape[-2:]
    def transport_adjacent(self,values,reverse=False):
        p='adjacent_reverse'if reverse else'adjacent_forward';idx=getattr(self,p+'_indices');w=getattr(self,p+'_weights').to(values)
        edges=torch.arange(len(self.frame_ids)-1)[:,None,None]
        return (values[...,edges,idx]*w).sum(-1)

def locked_inputs():
    old=read(V0/'lock.json');validate_phase(V0,'stage1',old)
    cfg=read(HERE/'config.json');assert cfg['queries']==64
    paths=[p for p in HERE.glob('*.py')if not p.name.startswith('test_')]+[HERE/'config.json']
    sources={**old['sources'],**old['frozen_sources'],**{str(p.relative_to(ROOT)):sha(p)for p in paths}}
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    identity=dict(config=cfg,sources=sources,old_lock_sha256=sha(V0/'lock.json'),old_seal_sha256=sha(V0/'stage1/sealed.json'),old_gate_sha256=sha(V0/'gate.json'),formal_state_sha256=sha(ROOT/'outputs/formal/state.json'),registered_config_sha256=sha(ROOT/'configs/default.json'),manifest_sha256=old['manifest_sha256'],inputs=old['inputs'],ids=old['ids'],GT_read=False)
    fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();L={**identity,'fingerprint':fp}
    if (TRIAL/'lock.json').exists():assert read(TRIAL/'lock.json')==L,'Do not mix source/config identities'
    else:
        atomic(TRIAL/'lock.json',L)
        for p in paths:
            dest=TRIAL/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    return cfg,L

def validate_one(item,L):
    k=item['id'];p=TRIAL/'predictions'/f'{k}.npz';r=read(TRIAL/'receipts'/f'{k}.json')
    assert r['fingerprint']==L['fingerprint'] and r['GT_read'] is False and sha(p)==r['sha256']
    for x in ('trace','bank'):assert sha(TRIAL/r[x+'_path'])==r[x+'_sha256']
    oldr=read(V0/'stage1/predictions'/f'{k}.json');assert oldr['bank_sha256']==r['old_bank_sha256']
    for f in ('reference_sha256','cache_sha256','native_mask_sha256'):assert r[f]==item[f]
    bank=torch.load(TRIAL/r['bank_path'],map_location='cpu',weights_only=False)
    oldbank=torch.load(V0/'stage1'/oldr['bank_path'],map_location='cpu',weights_only=False)
    traces=torch.load(TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
    assert bank['GT_read'] is False and torch.equal(bank['E'],oldbank['E'])
    E=bank['E'];pos=list(bank['positions8'])
    with np.load(p,allow_pickle=False)as f:
        assert set(f.files)==set(NAMES)|{'frame_ids','positions8','Strong_interval'}
        assert np.array_equal(f['ERefStrong'],E.numpy())
        assert f['positions8'].tolist()==pos
        for name in ALL_ARMS+('BoxFixed',):
            b=oldbank if name.startswith('Old')else bank;out=traces[name];final=out['final']
            assert np.array_equal(f[name],final.numpy())
            assert torch.equal(final[~b['active']],E[~b['active']])
            member=(b['boxes']==final[:,None]).all(-1)&b['valid'];assert member.any(-1).all()
            if name in ALL_ARMS:
                assert 1<=len(out['rounds'])<=3 and out['costs']['updates']==len(out['rounds'])
                assert out['initialphi'].eq(0).all() and out['costs']['heavy_forwards']==out['costs']['decoder_calls']==0
                assert len(out['trace']['fixed_teacher_paths'])==(2 if name.endswith('FixedKD')else 0)
                for row in out['rounds']:
                    assert row['after_score']>=row['before_score']-1e-12
                    if row['rolled_back']:assert torch.equal(row['before'],row['after']) and row['optimizer_after_sha256']==row['optimizer_before_sha256']
    return r

def validate(trial=TRIAL,lock=None):
    assert Path(trial).resolve()==TRIAL.resolve();L=read(TRIAL/'lock.json')if lock is None else lock
    seal=read(TRIAL/'sealed.json');assert seal['queries']==64 and seal['GT_read'] is False and seal['fingerprint']==L['fingerprint'] and seal['lock_sha256']==sha(TRIAL/'lock.json')
    assert set(seal['files'])==set(L['ids'])
    for p,h in L['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'outputs/formal/state.json')==L['formal_state_sha256'] and sha(ROOT/'configs/default.json')==L['registered_config_sha256']
    assert sha(V0/'gate.json')==L['old_gate_sha256'] and sha(V0/'stage1/sealed.json')==L['old_seal_sha256']
    assert sha(ROOT/'data/evaluation/development.jsonl')==L['manifest_sha256']
    receipts={}
    for item in L['inputs']:
        k=item['id'];r=validate_one(item,L);e=seal['files'][k]
        assert sha(TRIAL/e['path'])==e['sha256'] and sha(TRIAL/e['receipt_path'])==e['receipt_sha256'];receipts[k]=r
    assert not torch.cuda.is_initialized();return receipts

def main(run=False):
    cfg,L=locked_inputs()
    if not run:print(json.dumps(dict(status='prepared_locked',fingerprint=L['fingerprint'])));return
    assert not(TRIAL/'sealed.json').exists() and not(TRIAL/'run.json').exists(),'Refuse overwrite or implicit restart'
    from scripts.propagation_v0.bank import CandidateBank as OldBank
    from scripts.box_propagation.bank import CandidateBank as BoxBank
    from scripts.box_propagation.policy import fit,evidence_dp
    from scripts.propagation_opd.run import check_idle
    check_idle();torch.set_num_threads(1);start=time.perf_counter();files={}
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),fingerprint=L['fingerprint'],GT_read=False))
    try:
        for i,item in enumerate(L['inputs']):
            t0=time.perf_counter();k=item['id'];oldr=read(V0/'stage1/predictions'/f'{k}.json')
            ob=OldBank.from_state(torch.load(V0/'stage1'/oldr['bank_path'],map_location='cpu',weights_only=False))
            ev=SavedEvidence(torch.load(V0/'stage1'/oldr['evidence_path'],map_location='cpu',weights_only=False))
            cp=OLD/'student_cache'/f'{k}.pt';assert sha(cp)==item['cache_sha256'];cache=torch.load(cp,map_location='cpu',weights_only=False)
            begin=time.perf_counter();bank=BoxBank(ev,cache,cfg);bank_seconds=time.perf_counter()-begin
            assert torch.equal(ob.E,bank.E) and torch.equal(ob.frame_ids,bank.frame_ids) and torch.equal(ob.event,bank.event)
            # No geometry dedup is allowed to remove native evidence at observations.
            assert torch.equal(bank.boxes[pos:=torch.tensor(bank.positions),1],torch.as_tensor(cache['teacher']['boxes']).double()) and bank.valid[pos,1].all()
            with np.load(V0/'stage1/predictions'/f'{k}.npz')as oldp:
                result={n:oldp[n].copy()for n in ('frame_ids','positions8','Strong_interval','ERefStrong','GeometryOPD')};result['OldFixed']=oldp['EvidenceDP'].copy()
            traces={};costs={}
            for prefix,b in [('Old',ob),('Box',bank)]:
                for arm in ARMS:
                    name=prefix+arm;out=fit(b,item['key'],arm,cfg);traces[name]=out;result[name]=out['final'].numpy();costs[name]=out['costs']
            out=evidence_dp(bank);traces['BoxFixed']=out;result['BoxFixed']=out['final'].numpy();costs['BoxFixed']=out['costs']
            bp=TRIAL/'banks'/f'{k}.pt';tp=TRIAL/'traces'/f'{k}.pt';savept(bp,bank.private_state());savept(tp,traces)
            dest=TRIAL/'predictions'/f'{k}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.with_suffix('.tmp').open('wb')as f:np.savez(f,**result)
            os.replace(dest.with_suffix('.tmp'),dest)
            rr=dict(fingerprint=L['fingerprint'],sha256=sha(dest),trace_path=str(tp.relative_to(TRIAL)),trace_sha256=sha(tp),bank_path=str(bp.relative_to(TRIAL)),bank_sha256=sha(bp),old_bank_sha256=oldr['bank_sha256'],cache_sha256=item['cache_sha256'],native_mask_sha256=item['native_mask_sha256'],reference_sha256=item['reference_sha256'],timings=dict(bank_seconds=bank_seconds,total_seconds=time.perf_counter()-t0),costs=costs,GT_read=False,model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0)
            receipt=TRIAL/'receipts'/f'{k}.json';atomic(receipt,rr);validate_one(item,L)
            files[k]=dict(path=str(dest.relative_to(TRIAL)),sha256=sha(dest),receipt_path=str(receipt.relative_to(TRIAL)),receipt_sha256=sha(receipt))
            atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,GT_read=False));print(json.dumps(dict(completed=i+1,total=64,seconds=time.perf_counter()-start)),flush=True)
            del bank,ob,cache,ev,traces,result;gc.collect()
        atomic(TRIAL/'sealed.json',dict(queries=64,fingerprint=L['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),files=files,GT_read=False));validate()
        atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,fingerprint=L['fingerprint'],seconds=time.perf_counter()-start,GT_read=False))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),traceback=traceback.format_exc(),fingerprint=L['fingerprint'],GT_read=False));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();main(a.run)
