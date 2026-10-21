"""Sealed CPU-only parameter trial on shared frozen development captures."""
import argparse,json,sys,time,hashlib,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.boxfixed_tuning.capture import OUT
from scripts.boxfixed_tuning.core import BASE,check_params,temporal,spatial
from scripts.tube_temporal.run import read,sha,atomic,savept

def inputs(ds):
    lock=read(OUT/'capture_lock.json')
    for p,h in lock['code'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'configs/default.json')==lock['default_sha256'] and sha(ROOT/'outputs/formal/state.json')==lock['formal_sha256']
    rows=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['dataset']==ds][:64];assert [r['id']for r in rows]==lock['selection'][ds]
    if ds=='hc2':files=read(OUT/'hc2_inputs.json')['files']
    else:
        seal=read(OUT/'captures'/ds/'sealed.json');assert seal['queries']==64 and not seal['GT_read']
        files={k:dict(path=str((OUT/'captures'/ds/v['path']).relative_to(ROOT)),sha256=v['sha256'])for k,v in seal['files'].items()}
    for r in rows:assert sha(ROOT/files[r['id']]['path'])==files[r['id']]['sha256']
    return rows,files

def run(ds,name,p):
    torch.set_num_threads(1);check_params(p);plan=read(OUT/'plan.json')
    for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
    assert len(name)<80 and name.replace('_','').isalnum()
    rows,files=inputs(ds);dest=OUT/'trials'/ds/name;assert not(dest/'lock.json').exists(),'No overwrite'
    previous=list((OUT/'trials'/ds).glob('*/sealed.json'));assert len(previous)<18
    lock=dict(dataset=ds,name=name,params=p,plan_sha256=sha(OUT/'plan.json'),inputs=files,GT_read=False)
    lock['fingerprint']=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest();atomic(dest/'lock.json',lock)
    begin=time.perf_counter();entries={};baseline=OUT/'trials'/ds/'base'
    for i,r in enumerate(rows):
        key=r['id'];cap=torch.load(ROOT/files[key]['path'],map_location='cpu',weights_only=False);assert not cap['GT_read']
        cap['fps']=float(r['metadata']['fps']);td=OUT/'posterior_cache'/ds/key/(str(p['temperature'])+'.pt')
        if td.exists():
            tem=torch.load(td,map_location='cpu',weights_only=False);assert tem['input_sha256']==files[key]['sha256']
        else:
            tem={**temporal(cap,p['temperature']),'input_sha256':files[key]['sha256'],'GT_read':False};savept(td,tem)
        out=spatial(cap,tem['interval'],p);boxes=out['final'].numpy();action=out['actions'].numpy()
        if name=='base' and ds=='hc2':
            with np.load(ROOT/f'outputs/interval_pool/intervention/predictions/{key}.npz',allow_pickle=False)as old:
                assert np.array_equal(boxes,old['Hull_MBR'])and np.array_equal(tem['interval'],old['Hull_MBR_interval'])
        q=dest/'predictions'/f'{key}.npz';q.parent.mkdir(parents=True,exist_ok=True)
        np.savez(q,Candidate=boxes,Candidate_interval=tem['interval'],Source=np.asarray(cap['Source']),Source_interval=cap['native_interval'],
            frame_ids=cap['frame_ids'],positions8=cap['positions8'],actions=action)
        obs=np.zeros(len(action),bool);obs[cap['positions8']]=True
        stat=dict(entropy=tem['entropy'],expected_length_seconds=tem['expected_length_seconds'],
            observed_replacements=int(((action!=0)&obs).sum()),unobserved_replacements=int(((action!=0)&~obs).sum()),
            DP_seconds=out['costs']['seconds'],posterior_original_seconds=tem['seconds'])
        if name!='base':
            with np.load(baseline/'predictions'/f'{key}.npz',allow_pickle=False)as old:
                stat['changed_observed']=int(((action!=old['actions'])&obs).sum());stat['changed_unobserved']=int(((action!=old['actions'])&~obs).sum())
        entries[key]=dict(path=str(q.relative_to(dest)),sha256=sha(q),statistics=stat)
    assert not torch.cuda.is_initialized()
    atomic(dest/'sealed.json',dict(status='sealed_before_GT',files=entries,queries=64,params=p,GT_read=False,lock_sha256=sha(dest/'lock.json'),seconds=time.perf_counter()-begin))
    print(json.dumps(dict(status='sealed',dataset=ds,name=name,params=p,seconds=time.perf_counter()-begin)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True);p.add_argument('--name',required=True);p.add_argument('--params',required=True);a=p.parse_args();run(a.dataset,a.name,json.loads(a.params))
