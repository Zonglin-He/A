"""Fixed parent-isolated 32+32 development recheck. No formal job resume."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal import run as tube
from scripts.tube_fusion.core import head_logits,restore_geometry
from scripts.interval_pool.core import global_distributions,choose
from stvg.spatial import evidence_dp
read=tube.read;sha=tube.sha;atomic=tube.atomic;savept=tube.savept
BASE=ROOT/'outputs/interval_pool';METHODS=('Source_Native224','Hull_MBR','Mix_MBR')

def prepare():
    selection=read(BASE/'confirmation_selection.json');cfg=read(Path(__file__).with_name('config.json'))
    from stvg.worker import runtime_fingerprint
    fingerprint,proof=runtime_fingerprint('tastvg');sources={**proof['code'],**{str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*')if p.suffix in('.py','.json')}}
    rows=selection['development_recheck_selection'];dev=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if s.strip()]
    for ds,items in rows.items():
        assert len(items)==len({r['parent']for r in items})==len({r['video_sha256']for r in items})==32
        assert not{r['parent']for r in items}&{r['metadata']['source']for r in dev if r['dataset']==ds}
        assert not{r['video_sha256']for r in items}&{r['metadata']['video_sha256']for r in dev if r['dataset']==ds}
        manifests={r['manifest_path']for r in items};assert len(manifests)==1
        assert sha(ROOT/next(iter(manifests)))==selection['summary'][ds]['input_manifest_sha256']
    identity=dict(config=cfg,selection_sha256=sha(BASE/'confirmation_selection.json'),runtime_fingerprint=fingerprint,sources=sources,
        formal_state_sha256=sha(ROOT/'outputs/formal/state.json'),default_sha256=sha(ROOT/'configs/default.json'),
        stage='frozen_parent_separated_development_recheck_not_independent_confirmation',GT_read=False)
    lock={**identity,'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    p=BASE/'confirmation_lock.json'
    if p.exists():assert read(p)==lock,'Do not mix source/config/selection identities'
    else:
        atomic(p,lock);atomic(BASE/'confirmation_runtime.json',proof)
        for rel in sources:
            dest=BASE/'confirmation_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    return lock,rows

def check(lock):
    for rel,d in lock['sources'].items():assert sha(ROOT/rel)==d,rel
    assert sha(BASE/'confirmation_selection.json')==lock['selection_sha256']
    assert sha(ROOT/'outputs/formal/state.json')==lock['formal_state_sha256']and sha(ROOT/'configs/default.json')==lock['default_sha256']

def validate(dataset):
    lock=read(BASE/'confirmation_lock.json');check(lock);tr=BASE/('recheck_'+dataset);seal=read(tr/'sealed.json')
    selected=read(BASE/'confirmation_selection.json')['development_recheck_selection'][dataset]
    assert seal['GT_read']is False and seal['queries']==32 and seal['confirmation_lock_sha256']==sha(BASE/'confirmation_lock.json')and seal['fingerprint']==lock['fingerprint']
    assert set(seal['files'])=={r['query_id']for r in selected};out={}
    for key,item in seal['files'].items():
        assert sha(tr/item['path'])==item['sha256']and sha(tr/item['receipt_path'])==item['receipt_sha256']
        r=read(tr/item['receipt_path']);assert r['GT_read']is False and r['fingerprint']==lock['fingerprint']
        assert r['new_Source448_calls']==r['extra_temporal_decoder_calls']==r['updates']==0 and r['Source224_captures']==1
        assert r['ReferDINO_query_forwards']in(0,1)and r['head_final_layer_exact']
        for kind in('capture','posterior','spatial'):assert sha(tr/r[kind+'_path'])==r[kind+'_sha256']
        out[key]=r
    return out

@torch.no_grad()
def execute(dataset):
    lock,rows=prepare();items=rows[dataset];tr=BASE/('recheck_'+dataset);assert not(tr/'run.json').exists(),'No implicit restart/overwrite'
    from stvg.runtime import setup,lease
    from stvg.predictor import Predictor
    from stvg.decode import decode
    from stvg.inputs import validate_input
    from stvg.worker import resource_guard
    from stvg.teacher_bridge import ResidentTeacher
    setup();gpu=lease();predictor=Predictor(dataset,check=resource_guard);begin=time.perf_counter();files={}
    atomic(tr/'run.json',dict(status='running',dataset=dataset,GT_read=False,fingerprint=lock['fingerprint']))
    teacher=ResidentTeacher(BASE/'teacher_cache',lock['runtime_fingerprint']);predictor.teacher_service=teacher
    try:
        for i,item in enumerate(items):
            resource_guard();start=time.perf_counter();record=item['record'];key=record['id'];metadata=record['metadata']
            tick=time.perf_counter();frames,ids=decode(metadata,dataset);validate_input(frames,ids,metadata);rgb_seconds=time.perf_counter()-tick
            torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats()
            capture=predictor.capture(frames,ids,metadata,temporal_mode='native224',export_replay=True);del frames
            assert capture['cost']['Source448_calls']==0
            tc=capture['temporal_cache'];hidden=[r['hidden']for r in tc['offsets']];saved=[r['native_logits']for r in tc['offsets']]
            head=predictor.source.temp_embed;tick=time.perf_counter();head.cuda();torch.cuda.synchronize();head_to_gpu=time.perf_counter()-tick
            try:logits,head_cost=head_logits(head,hidden,saved)
            finally:head.cpu()
            pack=global_distributions(capture['records'],logits,ids);geometry=capture['geometry'];paths={};cost={}
            values=dict(Source_Native224=np.asarray(capture['Source']),Source_Native224_interval=capture['intervals']['Native'],frame_ids=np.asarray(ids),positions8=np.asarray(capture['planned_positions']))
            for name,q in [('Hull_MBR',pack['hull']),('Mix_MBR',pack['mix'])]:
                tick=time.perf_counter();interval=choose(ids,q);mb=time.perf_counter()-tick;tick=time.perf_counter();out=evidence_dp(geometry.for_interval(interval));dp=time.perf_counter()-tick
                values[name]=out['final'].numpy();values[name+'_interval']=interval;paths[name]=out;cost[name]=dict(MBR_seconds=mb,final_DP_seconds=dp)
            values['FixedSpatial']=values['Hull_MBR'].copy()
            reference=item['cache']['Source_E_prediction_only'];compat=None
            if reference is not None:
                assert sha(ROOT/reference['path'])==reference['sha256']
                with np.load(ROOT/reference['path'],allow_pickle=False)as old:
                    compat=dict(Source_bit_exact=np.array_equal(old['Source'],values['Source_Native224']),Source_max_abs=float(np.max(np.abs(old['Source']-values['Source_Native224']))),positions_bit_exact=np.array_equal(old['positions8'],values['positions8']))
            tick=time.perf_counter();pp=tr/'predictions'/f'{key}.npz';pp.parent.mkdir(parents=True,exist_ok=True);np.savez(pp,**values)
            cp=tr/'capture'/f'{key}.pt';savept(cp,dict(Source=capture['Source'],E=capture['E'],frame_ids=ids,planned_positions=capture['planned_positions'],geometry=geometry.state_dict(),temporal_cache=tc,records=capture['records'],teacher_native=capture['teacher_native'],cost=capture['cost'],GT_read=False))
            qp=tr/'posteriors'/f'{key}.pt';savept(qp,dict(pack,ids=ids,logits=logits,GT_read=False));sp=tr/'spatial'/f'{key}.pt';savept(sp,paths);io=time.perf_counter()-tick
            r=dict(fingerprint=lock['fingerprint'],dataset=dataset,GT_read=False,Source224_captures=1,new_Source448_calls=0,extra_temporal_decoder_calls=0,updates=0,head_final_layer_exact=True,
                ReferDINO_query_forwards=capture['cost']['ReferDINO_query_forwards'],capture_cost=capture['cost'],head_cost=head_cost,head_to_GPU_seconds=head_to_gpu,posterior_cost=pack['timing'],branch_cost=cost,
                RGB_decode_seconds=rgb_seconds,output_IO_seconds=io,query_wall_seconds=time.perf_counter()-start,Source_process_peak_GPU_bytes=torch.cuda.max_memory_allocated(),old_formal_compatibility=compat,
                capture_path=str(cp.relative_to(tr)),capture_sha256=sha(cp),posterior_path=str(qp.relative_to(tr)),posterior_sha256=sha(qp),spatial_path=str(sp.relative_to(tr)),spatial_sha256=sha(sp))
            rp=tr/'receipts'/f'{key}.json';atomic(rp,r);files[key]=dict(path=str(pp.relative_to(tr)),sha256=sha(pp),receipt_path=str(rp.relative_to(tr)),receipt_sha256=sha(rp))
            atomic(tr/'progress.json',dict(completed=i+1,total=32,dataset=dataset,seconds=time.perf_counter()-begin,GT_read=False));print(json.dumps(dict(dataset=dataset,completed=i+1,total=32)),flush=True)
            assert not any(p.requires_grad or p.grad is not None for p in predictor.source.parameters())
            del capture,tc,hidden,saved,pack,geometry,paths,values,logits;gc.collect();torch.cuda.empty_cache()
        atomic(tr/'sealed.json',dict(queries=32,dataset=dataset,files=files,fingerprint=lock['fingerprint'],confirmation_lock_sha256=sha(BASE/'confirmation_lock.json'),GT_read=False));validate(dataset)
        atomic(tr/'run.json',dict(status='completed_sealed',queries=32,dataset=dataset,seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(tr/'failure.json',dict(completed=len(files),dataset=dataset,GT_read=False,traceback=traceback.format_exc()));raise
    finally:teacher.close();gpu.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['hc2','vidstg']);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:
        lock,_=prepare();print(json.dumps(dict(status='frozen_not_run',fingerprint=lock['fingerprint'])))
    elif a.dataset:execute(a.dataset)
    else:p.error('--prepare or --dataset required')
