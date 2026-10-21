"""Round2 HC64 fixed 2x2 intervention, after the completed round1 diagnosis."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.interval_pool import development as dev
from scripts.interval_pool.core import choose
from scripts.tube_fusion.core import restore_geometry
from scripts.tube_temporal import run as tube
from stvg.spatial import evidence_dp
read=tube.read;sha=tube.sha;atomic=tube.atomic;savept=tube.savept
TRIAL=ROOT/'outputs/interval_pool/intervention';METHODS=('Source_Native224','Hull_MBR','Hull_MAP','Mix_MBR','Mix_MAP')

def check(lock):
    dev.check(lock['development_lock'])
    assert sha(dev.TRIAL/'lock.json')==lock['development_lock_sha256']and sha(dev.TRIAL/'sealed.json')==lock['development_seal_sha256']
    for path,digest in lock['sources'].items():assert sha(ROOT/path)==digest,path

def validate():
    lock=read(TRIAL/'lock.json');check(lock);s=read(TRIAL/'sealed.json')
    assert s['GT_read']is False and s['fingerprint']==lock['fingerprint']and s['lock_sha256']==sha(TRIAL/'lock.json')
    assert set(s['files'])==set(lock['ids']);rs={}
    for key,item in s['files'].items():
        assert sha(TRIAL/item['path'])==item['sha256']and sha(TRIAL/item['receipt_path'])==item['receipt_sha256']
        r=read(TRIAL/item['receipt_path']);assert r['GT_read']is False and r['old_hull_exact']and r['network_calls']==0
        assert sha(TRIAL/r['spatial_path'])==r['spatial_sha256'];rs[key]=r
    return rs

def execute():
    old=dev.validate();assert(dev.TRIAL/'diagnosis_receipt.json').exists(),'First round must finish before intervention'
    assert not(TRIAL/'lock.json').exists(),'No overwrite'
    sources={str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*')if p.suffix in('.py','.json')}
    identity=dict(config=read(Path(__file__).with_name('config.json')),sources=sources,ids=read(dev.TRIAL/'lock.json')['ids'],
        development_lock=read(dev.TRIAL/'lock.json'),development_lock_sha256=sha(dev.TRIAL/'lock.json'),development_seal_sha256=sha(dev.TRIAL/'sealed.json'),GT_read=False)
    lock={**identity,'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()};atomic(TRIAL/'lock.json',lock)
    files={};begin=time.perf_counter();ds=read(dev.TRIAL/'sealed.json')
    for i,key in enumerate(lock['ids']):
        tick=time.perf_counter();r=old[key];pack=torch.load(dev.TRIAL/r['posterior_path'],map_location='cpu',weights_only=False)
        c=torch.load(tube.TRIAL/r['capture_path'],map_location='cpu',weights_only=False);geometry=restore_geometry(c['geometry'])
        with np.load(dev.TRIAL/ds['files'][key]['path'],allow_pickle=False)as p:before={k:p[k].copy()for k in p.files}
        input_seconds=time.perf_counter()-tick;values=dict(Source_Native224=before['Source'],Source_Native224_interval=before['Source_interval'],FixedSpatial=before['FixedSpatial'],frame_ids=before['frame_ids'],positions8=before['positions8']);paths={};cost={}
        for name,q,rule in [('Hull_MBR',pack['hull'],'mbr'),('Hull_MAP',pack['hull'],'map'),('Mix_MBR',pack['mix'],'mbr'),('Mix_MAP',pack['mix'],'map')]:
            tick=time.perf_counter();interval=choose(pack['ids'],q,rule);decode_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            out=evidence_dp(geometry.for_interval(interval));dp_seconds=time.perf_counter()-tick
            values[name]=out['final'].numpy();values[name+'_interval']=interval;paths[name]=out;cost[name]=dict(decode_seconds=decode_seconds,final_DP_seconds=dp_seconds)
        assert np.array_equal(values['Hull_MBR'],before['Global6'])and np.array_equal(values['Hull_MBR_interval'],before['Global6_interval'])
        pp=TRIAL/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez(pp,**values);sp=TRIAL/'spatial'/f'{key}.pt';savept(sp,paths)
        receipt=dict(fingerprint=lock['fingerprint'],GT_read=False,network_calls=0,old_hull_exact=True,cost=cost,input_IO_seconds=input_seconds,spatial_path=str(sp.relative_to(TRIAL)),spatial_sha256=sha(sp))
        rp=TRIAL/'receipts'/f'{key}.json';atomic(rp,receipt);files[key]=dict(path=str(pp.relative_to(TRIAL)),sha256=sha(pp),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp));print(json.dumps(dict(completed=i+1,total=64)),flush=True)
    atomic(TRIAL/'sealed.json',dict(queries=64,files=files,GT_read=False,fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json')));validate()
    atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,seconds=time.perf_counter()-begin,GT_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('--run required')
    execute()
