"""One fixed cached HC64 readout trial. All predictions seal before GT."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal import run as old
from scripts.tube_fusion.core import load_head,head_logits,layer_posteriors,fuse,restore_geometry,final_prediction,tensor_digest
from stvg.temporal import probabilities,mass_table,final_mbr
read=old.read;sha=old.sha;atomic=old.atomic
HERE=Path(__file__).resolve().parent;TRIAL=ROOT/'outputs/tube_fusion_hc64';METHODS=('Source_Native224','Global6','GlobalTube6','GlobalPermuted6','Strong_Reference')

def check_sources(lock):
    old.check_sources(lock['old_lock'])
    assert sha(old.TRIAL/'lock.json')==lock['old_lock_sha256']and sha(old.TRIAL/'sealed.json')==lock['old_seal_sha256']
    for p,d in lock['sources'].items():assert sha(ROOT/p)==d,p
    assert sha(ROOT/lock['checkpoint_path'])==lock['checkpoint_sha256']

def prepare():
    cfg=read(HERE/'config.json');assert cfg['queries']==64 and cfg['layers']==6 and cfg['mix']==.5 and cfg['methods']==list(METHODS)
    previous=read(old.TRIAL/'lock.json');receipts=old.validate();seal=read(old.TRIAL/'sealed.json')
    manifest=ROOT/'data/evaluation/development.jsonl';assert sha(manifest)==previous['manifest_sha256']
    records=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()];records=[r for r in records if r['dataset']=='hc2'][:64]
    assert [r['id']for r in records]==previous['ids']
    sources={str(p.relative_to(ROOT)):sha(p)for p in HERE.glob('*')if p.suffix in('.py','.json')}
    sources['external/TA-STVG/models/net_utils.py']=sha(ROOT/'external/TA-STVG/models/net_utils.py')
    cfg_source=read(ROOT/'configs/default.json')['datasets']['hc2'];checkpoint=cfg_source['source_checkpoint']
    inputs=[dict(id=k,receipt_sha256=sha(old.TRIAL/'receipts'/f'{k}.json'),prediction_sha256=receipts[k]['prediction_sha256'],trace_sha256=receipts[k]['trace_sha256'],capture_sha256=receipts[k]['capture_sha256'])for k in previous['ids']]
    identity=dict(config=cfg,sources=sources,old_lock=previous,old_lock_sha256=sha(old.TRIAL/'lock.json'),old_seal_sha256=sha(old.TRIAL/'sealed.json'),
        inputs=inputs,ids=previous['ids'],manifest_sha256=sha(manifest),checkpoint_path=checkpoint,checkpoint_sha256=sha(ROOT/checkpoint),GT_read=False)
    lock={**identity,'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    assert not(TRIAL/'lock.json').exists(),'Independent trial; do not overwrite or restart silently'
    atomic(TRIAL/'lock.json',lock)
    for rel in sources:
        dest=TRIAL/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    return cfg,lock,records,receipts,seal

def validate(lock=None):
    lock=read(TRIAL/'lock.json')if lock is None else lock;check_sources(lock);s=read(TRIAL/'sealed.json')
    assert s['GT_read']is False and s['queries']==64 and s['lock_sha256']==sha(TRIAL/'lock.json')and s['fingerprint']==lock['fingerprint']
    assert set(s['files'])==set(lock['ids']);receipts={}
    for item in lock['inputs']:
        key=item['id'];entry=s['files'][key];r=read(TRIAL/entry['receipt_path'])
        assert sha(TRIAL/entry['path'])==entry['sha256']and sha(TRIAL/entry['receipt_path'])==entry['receipt_sha256']
        assert r['fingerprint']==lock['fingerprint']and r['GT_read']is False and r['last_layer_logits_exact']and r['last_posterior_interval_exact']and r['old_spatial_exact']
        assert r['new_RGB_encoder_decoder_expert_updates']==0 and r['head_calls']==6
        assert sha(old.TRIAL/'receipts'/f'{key}.json')==item['receipt_sha256']
        rr=read(old.TRIAL/'receipts'/f'{key}.json')
        for stem in('prediction','trace','capture'):assert rr[stem+'_sha256']==item[stem+'_sha256']and sha(old.TRIAL/rr[stem+'_path'])==item[stem+'_sha256']
        for stem in('posterior','spatial_trace'):assert sha(TRIAL/r[stem+'_path'])==r[stem+'_sha256']
        with np.load(TRIAL/entry['path'],allow_pickle=False)as p,np.load(old.TRIAL/rr['prediction_path'],allow_pickle=False)as before:
            assert set(p.files)==set(METHODS)|{m+'_interval'for m in METHODS}|{'frame_ids','positions8'}
            assert np.array_equal(p['Source_Native224'],before['Source'])and np.array_equal(p['Source_Native224_interval'],before['B_Native224_interval'])
            assert np.array_equal(p['Strong_Reference'],before['A_Strong'])and np.array_equal(p['Strong_Reference_interval'],before['A_Strong_interval'])
            for name in METHODS:assert p[name].shape==before['Source'].shape and np.isfinite(p[name]).all()and(p[name][:,2:]>0).all()
        receipts[key]=r
    return receipts

@torch.no_grad()
def execute(cfg,lock,records,receipts,seal):
    from stvg.runtime import setup,lease
    setup();gpu=lease();begin=time.perf_counter();files={}
    atomic(TRIAL/'run.json',dict(status='running',PID=os.getpid(),GT_read=False,fingerprint=lock['fingerprint']))
    try:
        tick=time.perf_counter();head,head_meta=load_head(ROOT,ROOT/lock['checkpoint_path']);head.cuda();torch.cuda.synchronize();head_load=time.perf_counter()-tick
        original=tensor_digest(head.state_dict());assert original==head_meta['weights_sha256']
        for i,record in enumerate(records):
            tick=time.perf_counter();key=record['id'];r=receipts[key]
            c=torch.load(old.TRIAL/r['capture_path'],map_location='cpu',weights_only=False);t=torch.load(old.TRIAL/r['trace_path'],map_location='cpu',weights_only=False)
            with np.load(old.TRIAL/r['prediction_path'],allow_pickle=False)as before:previous={k:before[k].copy()for k in before.files}
            read_seconds=time.perf_counter()-tick;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            ids=c['frame_ids'];records_old=c['records'];geometry=restore_geometry(c['geometry'])
            modes=[('G',[o['hidden']for o in c['temporal_cache']['offsets']],[o['native_logits']for o in c['temporal_cache']['offsets']],'B_Native224'),
                   ('T',t['C_TubeReplay']['temporal_audit']['hidden'],t['C_TubeReplay']['temporal_audit']['logits'],'C_TubeReplay'),
                   ('P',t['D_Permuted']['temporal_audit']['hidden'],t['D_Permuted']['temporal_audit']['logits'],'D_Permuted')]
            qs={};all_layers={};costs={}
            for name,hidden,saved,baseline in modes:
                z,cost=head_logits(head,hidden,saved);costs[name]=cost
                U,last=mass_table(probabilities(records_old,[v[-1]for v in z]));assert U==ids
                assert np.array_equal(last,t[baseline]['posterior'])and final_mbr(U,last)['interval']==list(t[baseline]['interval'])
                qs[name],all_layers[name]=layer_posteriors(records_old,z,ids)
            # Prove deserializing geometry changes no original B/C/D readout.
            for baseline in('B_Native224','C_TubeReplay','D_Permuted'):
                from stvg.spatial import evidence_dp
                recovered=evidence_dp(geometry.for_interval(previous[baseline+'_interval']))
                assert np.array_equal(recovered['final'].numpy(),previous[baseline])and torch.equal(recovered['actions'],t[baseline]['spatial']['actions'])and recovered['score']==t[baseline]['spatial']['score']
            values=dict(Source_Native224=previous['Source'],Source_Native224_interval=previous['B_Native224_interval'],Strong_Reference=previous['A_Strong'],Strong_Reference_interval=previous['A_Strong_interval'],frame_ids=previous['frame_ids'],positions8=previous['positions8'])
            final_q=dict(Global6=qs['G'],GlobalTube6=fuse(qs['G'],qs['T']),GlobalPermuted6=fuse(qs['G'],qs['P']));spatial={};post={}
            tick=time.perf_counter()
            for name,q in final_q.items():
                boxes,interval,path=final_prediction(geometry,ids,q);values[name]=boxes;values[name+'_interval']=interval;spatial[name]=path;post[name]=q
            post_DP_seconds=time.perf_counter()-tick;compute_seconds=time.perf_counter()-start;tick=time.perf_counter()
            pp=TRIAL/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez(pp,**values)
            qp=TRIAL/'posteriors'/f'{key}.pt';old.savept(qp,dict(means=qs,layers=all_layers,final=post,GT_read=False))
            sp=TRIAL/'spatial'/f'{key}.pt';old.savept(sp,dict(outputs=spatial,GT_read=False));write_seconds=time.perf_counter()-tick
            rr=dict(fingerprint=lock['fingerprint'],GT_read=False,last_layer_logits_exact=True,last_posterior_interval_exact=True,old_spatial_exact=True,
                head_calls=6,new_RGB_encoder_decoder_expert_updates=0,head_costs=costs,input_IO_seconds=read_seconds,head_and_posterior_spatial_seconds=compute_seconds,
                temporal_postprocess_final_DP_seconds=post_DP_seconds,output_IO_seconds=write_seconds,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                posterior_path=str(qp.relative_to(TRIAL)),posterior_sha256=sha(qp),spatial_trace_path=str(sp.relative_to(TRIAL)),spatial_trace_sha256=sha(sp))
            rp=TRIAL/'receipts'/f'{key}.json';atomic(rp,rr);files[key]=dict(path=str(pp.relative_to(TRIAL)),sha256=sha(pp),receipt_path=str(rp.relative_to(TRIAL)),receipt_sha256=sha(rp))
            atomic(TRIAL/'progress.json',dict(completed=i+1,total=64,GT_read=False,seconds=time.perf_counter()-begin));print(json.dumps(dict(completed=i+1,total=64)),flush=True)
        assert tensor_digest(head.state_dict())==original and not any(p.requires_grad or p.grad is not None for p in head.parameters())
        atomic(TRIAL/'sealed.json',dict(queries=64,files=files,GT_read=False,fingerprint=lock['fingerprint'],lock_sha256=sha(TRIAL/'lock.json'),head=head_meta,head_load_seconds=head_load))
        validate(lock);atomic(TRIAL/'run.json',dict(status='completed_sealed',queries=64,GT_read=False,seconds=time.perf_counter()-begin,head_load_seconds=head_load,fingerprint=lock['fingerprint']))
    except BaseException:
        atomic(TRIAL/'failure.json',dict(completed=len(files),GT_read=False,traceback=traceback.format_exc()));raise
    finally:gpu.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not a.run:p.error('Explicit --run required; independent outputs, no overwrite')
    execute(*prepare())
