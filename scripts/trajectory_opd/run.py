"""Isolated HC64 trajectory-evidence development experiment; explicit --run."""
import os,sys,json,hashlib,time,gc,traceback,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.state_opd.run import prepare,read,sha,atomic,savept,OLD
from scripts.trajectory_opd.policy import fit,ARMS
from scripts.trajectory_opd.evidence import TrajectoryEvidence
HERE=Path(__file__).resolve().parent;TR=ROOT/'outputs/trajectory_opd_hc64'


def main(run=False,resume=False):
    _,records,items,oldlock=prepare();records=records[:64];items=items[:64]
    cfg=read(HERE/'config.json')
    assert cfg['queries']==64 and cfg['arms']==list(ARMS)
    paths=sorted(p for p in HERE.glob('*.py') if not p.name.startswith('test_'))+[HERE/'config.json']
    identity=dict(config=cfg,parent_fingerprint=oldlock['fingerprint'],
                  sources={str(p.relative_to(ROOT)):sha(p)for p in paths},
                  frozen_sources=oldlock['sources'],loader_identity=oldlock['loader_identity'],
                  formal_state_sha256=oldlock['formal_state_sha256'],
                  registered_config_sha256=oldlock['registered_config_sha256'],
                  manifest_sha256=oldlock['manifest_sha256'],
                  ordered_input_digest=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest(),
                  ids=[r['id']for r in records],GT_read=False)
    fp=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();lock={**identity,'fingerprint':fp}
    if (TR/'lock.json').exists():assert read(TR/'lock.json')==lock,'Do not mix execution identities'
    else:
        atomic(TR/'lock.json',lock)
        for p in paths:
            dest=TR/'source_snapshot'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    if not run:
        print(json.dumps(dict(status='prepared_not_run',fingerprint=fp,queries=64,arms=ARMS)));return
    assert not (TR/'sealed.json').exists(),'Already sealed, refuse overwrite'
    assert resume or not (TR/'run.json').exists(),'Existing execution requires explicit identical --resume'
    from stvg.runtime import setup,lease,load_source
    setup();leasefile=lease();model=None;files={};begin=time.perf_counter()
    atomic(TR/'run.json',dict(status='running',PID=os.getpid(),fingerprint=fp,GT_read=False))
    sc=read(OLD/'config.json')['datasets']['hc2']
    try:
        for rec,item in zip(records,items):
            key=rec['id'];dest=TR/'predictions'/f'{key}.npz';receipt=dest.with_suffix('.json')
            if receipt.exists():
                rr=read(receipt);assert rr['fingerprint']==fp and sha(dest)==rr['sha256'] and sha(TR/rr['trace_path'])==rr['trace_sha256'];files[key]=rr;continue
            assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
            start=time.perf_counter();cp=OLD/'student_cache'/f'{key}.pt';mp=OLD/'research/native_masks'/item['mask_path']
            assert sha(cp)==item['cache_sha256'] and sha(mp)==item['native_mask_sha256']
            cache=torch.load(cp,map_location='cpu',weights_only=False);pack=torch.load(mp,map_location='cpu',weights_only=False)
            assert cache['GT_read'] is pack['GT_read'] is False
            assert sha(OLD/item['reference_path'])==item['reference_sha256']
            with np.load(OLD/item['reference_path'])as f:gold={k:f[k].copy()for k in f.files}
            assert np.array_equal(gold['E_Ref'],cache['E']) and np.array_equal(gold['Source'],cache['Source'])
            if model is None:
                load_begin=time.perf_counter()
                model=load_source(sc);model.cpu().eval().requires_grad_(False);gc.collect();torch.cuda.empty_cache()
                model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
                atomic(TR/'load.json',dict(seconds=time.perf_counter()-load_begin,encoder_forwards=0,expert_forwards=0,RGB_reads=0,GT_read=False))
            evidence_begin=time.perf_counter();evidence=TrajectoryEvidence(pack,cache,device='cuda')
            evidence_seconds=time.perf_counter()-evidence_begin
            result={k:gold[k]for k in ['frame_ids','positions8','Strong_interval']};traces={};costs={}
            for arm in ARMS:
                out=fit(model,cache,evidence,item['key'],arm,cfg)
                if arm=='TrajectoryOPD':result['InitialModes']=out['initial_final'].numpy()
                result[arm]=out['final'].numpy();costs[arm]=out['costs'];traces[arm]=out
                atomic(TR/'progress.json',dict(completed=len(files),total=64,id=key,arm=arm,GT_read=False))
                print(json.dumps(dict(completed=len(files),arm=arm,cost=out['costs'])),flush=True)
            ep=TR/'evidence'/f'{key}.pt';savept(ep,evidence.private_state())
            tp=TR/'traces'/f'{key}.pt';savept(tp,traces);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.npz.tmp')
            with tmp.open('wb')as h:np.savez(h,**result)
            os.replace(tmp,dest)
            rr=dict(fingerprint=fp,sha256=sha(dest),trace_path=str(tp.relative_to(TR)),trace_sha256=sha(tp),evidence_path=str(ep.relative_to(TR)),evidence_sha256=sha(ep),evidence_summary=evidence.metadata,evidence_seconds=evidence_seconds,reference_sha256=item['reference_sha256'],cache_sha256=item['cache_sha256'],native_mask_sha256=item['native_mask_sha256'],costs=costs,total_seconds=time.perf_counter()-start,GT_read=False)
            atomic(receipt,rr);files[key]=rr
            atomic(TR/'progress.json',dict(completed=len(files),total=64,GT_read=False))
            print(json.dumps(dict(completed=len(files),seconds=rr['total_seconds'])),flush=True)
            del traces,out,result,cache,pack,evidence;gc.collect();torch.cuda.empty_cache()
        assert len(files)==64
        for p,h in {**lock['sources'],**lock['frozen_sources']}.items():assert sha(ROOT/p)==h
        assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']
        assert sha(ROOT/'configs/default.json')==lock['registered_config_sha256']
        atomic(TR/'sealed.json',dict(fingerprint=fp,queries=64,files={k:dict(path='predictions/'+k+'.npz',sha256=v['sha256'],receipt_sha256=sha(TR/'predictions'/f'{k}.json'))for k,v in files.items()},lock_sha256=sha(TR/'lock.json'),GT_read=False))
        atomic(TR/'run.json',dict(status='completed_sealed',queries=64,seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(TR/'failure.json',dict(completed=len(files),traceback=traceback.format_exc(),GT_read=False));raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache();leasefile.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.run,a.resume)
