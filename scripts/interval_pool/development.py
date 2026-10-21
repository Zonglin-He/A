"""Round1 label-free HC64 artifacts, retaining only predeclared diagnostic decisions."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_fusion import run as fusion
from scripts.tube_temporal import run as tube
from scripts.tube_fusion.core import load_head,head_logits,tensor_digest
from scripts.interval_pool.core import global_distributions,map_view,choose
read=tube.read;sha=tube.sha;atomic=tube.atomic;savept=tube.savept
TRIAL=ROOT/'outputs/interval_pool/development';HERE=Path(__file__).resolve().parent

def check(lock):
    fusion.check_sources(lock['fusion_lock'])
    assert sha(fusion.TRIAL/'lock.json')==lock['fusion_lock_sha256']and sha(fusion.TRIAL/'sealed.json')==lock['fusion_seal_sha256']
    for rel,digest in lock['sources'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(ROOT/'outputs/interval_pool/confirmation_selection.json')==lock['confirmation_selection_sha256']

def prepare():
    source=fusion.read(fusion.TRIAL/'lock.json');refs=fusion.validate(source)
    selection=ROOT/'outputs/interval_pool/confirmation_selection.json';assert selection.exists(),'Freeze confirmation IDs before labels'
    config=read(HERE/'config.json');assert config['layers']==6 and config['offsets']==2 and config['no_sweeps']
    sources={str(p.relative_to(ROOT)):sha(p)for p in HERE.glob('*')if p.suffix in('.py','.json')}
    identity=dict(config=config,sources=sources,fusion_lock=source,fusion_lock_sha256=sha(fusion.TRIAL/'lock.json'),
        fusion_seal_sha256=sha(fusion.TRIAL/'sealed.json'),ids=source['ids'],manifest_sha256=source['manifest_sha256'],confirmation_selection_sha256=sha(selection),GT_read=False)
    lock={**identity,'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    assert not(TRIAL/'lock.json').exists(),'Never overwrite a previous round'
    atomic(TRIAL/'lock.json',lock)
    for rel in sources:
        p=TRIAL/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    return lock,refs

def validate(trial=None):
    assert trial is None or Path(trial).resolve()==TRIAL.resolve()
    lock=read(TRIAL/'lock.json');check(lock);seal=read(TRIAL/'sealed.json')
    assert seal['fingerprint']==lock['fingerprint']and seal['lock_sha256']==sha(TRIAL/'lock.json')and seal['GT_read']is False
    assert len(seal['files'])==64 and set(seal['files'])==set(lock['ids']);rows={}
    for key,entry in seal['files'].items():
        r=read(TRIAL/entry['receipt_path']);assert r['GT_read']is False and r['head_calls']==2 and r['network_calls_except_head']==0
        assert r['fingerprint']==lock['fingerprint']and r['old_Global6_exact']
        assert sha(TRIAL/entry['path'])==entry['sha256']and sha(TRIAL/entry['receipt_path'])==entry['receipt_sha256']
        assert sha(TRIAL/r['posterior_path'])==r['posterior_sha256']
        assert sha(tube.TRIAL/r['capture_path'])==r['capture_sha256']
        rows[key]=r
    return rows

@torch.no_grad()
def execute():
    lock,refs=prepare()
    from stvg.runtime import setup,lease
    setup();gpu=lease();files={};begin=time.perf_counter()
    try:
        head,meta=load_head(ROOT,ROOT/lock['fusion_lock']['checkpoint_path']);head.cuda();original=tensor_digest(head.state_dict())
        old_lock=read(tube.TRIAL/'lock.json')
        for i,key in enumerate(lock['ids']):
            rr=read(tube.TRIAL/'receipts'/f'{key}.json');start=time.perf_counter();c=torch.load(tube.TRIAL/rr['capture_path'],map_location='cpu',weights_only=False)
            hidden=[o['hidden']for o in c['temporal_cache']['offsets']];saved=[o['native_logits']for o in c['temporal_cache']['offsets']]
            logits,cost=head_logits(head,hidden,saved);ids=c['frame_ids'];pack=global_distributions(c['records'],logits,ids)
            fp=torch.load(fusion.TRIAL/refs[key]['posterior_path'],map_location='cpu',weights_only=False)
            assert np.array_equal(pack['hull'],fp['final']['Global6'])and np.array_equal(pack['hull_layers'],fp['layers']['G'])
            with np.load(fusion.TRIAL/read(fusion.TRIAL/'sealed.json')['files'][key]['path'],allow_pickle=False)as p:
                interval=choose(ids,pack['hull']);assert np.array_equal(interval,p['Global6_interval'])
                values=dict(Global6=p['Global6'],Global6_interval=p['Global6_interval'],Source=p['Source_Native224'],Source_interval=p['Source_Native224_interval'],FixedSpatial=p['Strong_Reference'],frame_ids=p['frame_ids'],positions8=p['positions8'])
            candidates=[];names=[]
            for layer,views in enumerate(pack['views']):
                for offset,g in enumerate(views):
                    candidates.append(choose(g['ids'],map_view(g,g['ids'])));names.append(f'L{layer+1}O{offset}')
            pack.update(ids=ids,logits=logits,candidate_intervals=np.asarray(candidates+[interval]),candidate_names=names+['Global6'],GT_read=False)
            pp=TRIAL/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez(pp,**values)
            qp=TRIAL/'posteriors'/f'{key}.pt';savept(qp,pack)
            r=dict(fingerprint=lock['fingerprint'],GT_read=False,old_Global6_exact=True,head_calls=2,network_calls_except_head=0,
                head_cost=cost,posterior_construct=pack['timing'],seconds=time.perf_counter()-start,
                posterior_path=str(qp.relative_to(TRIAL)),posterior_sha256=sha(qp),capture_path=rr['capture_path'],capture_sha256=rr['capture_sha256'])
            rp=TRIAL/'receipts'/f'{key}.json';atomic(rp,r);files[key]=dict(path=str(pp.relative_to(TRIAL)),sha256=sha(pp),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp))
            print(json.dumps(dict(completed=i+1,total=64)),flush=True)
        assert tensor_digest(head.state_dict())==original
        atomic(TRIAL/'sealed.json',dict(files=files,queries=64,fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),head=meta,GT_read=False))
        validate();atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,seconds=time.perf_counter()-begin,GT_read=False))
    finally:gpu.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('Explicit --run required')
    execute()
