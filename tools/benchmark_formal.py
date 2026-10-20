"""Post-score, label-free Source versus complete-method efficiency benchmark.

This command never selects parameters or opens annotations. It must be launched
in a fresh interpreter for one completed TA-STVG target at a time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as handle:
        for chunk in iter(lambda:handle.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def parent_key(row):
    meta=row['metadata']
    return str(meta.get('source')or meta.get('original_video_id')or meta['video_sha256'])


def select_records(records,maximum=16):
    """Uniform ranks of fixed input size, one query per distinct source parent."""
    if maximum<1:raise ValueError('Sample maximum must be positive')
    ordered=sorted(records,key=lambda r:(len(r['metadata']['frame_ids']),r['metadata']['video_sha256'],r['id']))
    unique=[];seen=set()
    for row in ordered:
        parent=parent_key(row)
        if parent not in seen:unique.append(row);seen.add(parent)
    k=min(maximum,len(unique))
    if k<=1:return unique[:k]
    indices=[round(i*(len(unique)-1)/(k-1))for i in range(k)]
    return [unique[i]for i in indices]


def completed_job(root,backbone,dataset):
    """Validate the entire completed prediction job before loading any model."""
    from tools.run_formal import canonical,validate_worker
    if backbone!='tastvg':raise NotImplementedError('CG-STVG efficiency benchmarking is deferred until the TA-STVG stage finishes')
    if dataset not in ['hc2','vidstg']:raise ValueError('Unsupported target')
    root=Path(root).resolve();folder=root/'outputs/formal'
    plan=json.loads((folder/'plan.json').read_text());state=json.loads((folder/'state.json').read_text())
    if state.get('plan_sha256')!=hashlib.sha256(canonical(plan)).hexdigest():raise ValueError('Formal state is not bound to its plan')
    key=backbone+'-'+dataset;jobs=[j for j in plan['jobs']if j['id']==key]
    if len(jobs)!=1:raise ValueError('Requested job is absent from the locked plan')
    job=jobs[0];progress=state['jobs'][key]
    if progress.get('status')!='scored':raise ValueError('Benchmark requires the whole job sealed and scored')
    output=root/job['output'];manifest=root/job['manifest']
    if digest(manifest)!=job['manifest_sha256']:raise ValueError('Manifest changed after the formal job was locked')
    validate_worker(root,job,require_seal=True)
    for filename,keyname in [('metrics.json','metrics_sha256'),('sealed.json','seal_sha256')]:
        if digest(output/filename)!=progress.get(keyname):raise ValueError('Completed '+filename+' hash is invalid')
    # Metrics bytes are hashed, never parsed or used for sample selection.
    records=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]
    if any(r['dataset']!=dataset for r in records):raise ValueError('Mixed target in manifest')
    seal=json.loads((output/'sealed.json').read_text());run=json.loads((output/'run.json').read_text())
    return records,output,seal,run


def require_equal(actual,expected,name):
    import numpy as np
    if not np.array_equal(np.asarray(actual),np.asarray(expected)):
        raise AssertionError('Bitwise sealed-prediction regression failed: '+name)


def verify_outputs(source,full,ids,reference):
    import numpy as np
    with np.load(reference,allow_pickle=False)as saved:
        for name,value in [('Source',source['Source']),('Native_interval',source['Native_interval']),
                           ('Source',full['Source']),('Native_interval',full['intervals']['Native']),
                           ('Live',full['Live']),('Strong_interval',full['intervals']['Strong']),('frame_ids',ids)]:
            require_equal(value,saved[name],name)
    return True


def source_only(predictor,record):
    """Actual standalone native Source: RGB, query parse, frozen 224, physical MBR."""
    import gc
    import numpy as np
    import torch
    from stvg.decode import decode
    from stvg.inputs import validate_metadata,validate_input
    from stvg.source import frozen_forward
    from stvg.readout import prediction
    from stvg.temporal import probabilities,mass_table,final_mbr
    from stvg.tensors import detached
    from stvg.query import QuerySubjectParser
    meta=record['metadata'];validate_metadata(meta)
    torch.manual_seed(20260920);np.random.seed(20260920)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    frames,ids=decode(meta,record['dataset']);validate_input(frames,ids,meta)
    # Keep the same query-only parser used by the complete predictor.
    parser=getattr(predictor,'_subject_parser',None)or getattr(predictor,'parser',None)
    if parser is None:parser=QuerySubjectParser(ROOT/'checkpoints/stanza');predictor.parser=parser
    row=dict(input=meta,frame_ids=list(ids),parses=dict(subject=parser(meta['caption'])['subject']))
    model=predictor.source.cuda()
    try:
        batch,recs,base=frozen_forward(model,frames,row)
        native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids)
        U,q=mass_table(probabilities(recs,detached(base.zero['logits'],'cpu')))
        out=dict(Source=native['boxes'].detach().cpu().numpy(),Native_interval=final_mbr(U,q)['interval'])
        del batch,recs,base,native,frames
    finally:model.cpu();gc.collect();torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return out,ids,dict(wall_seconds=time.perf_counter()-start,peak_allocated_cuda_bytes=int(torch.cuda.max_memory_allocated()))


def aggregate(rows):
    if not rows:raise ValueError('No benchmark rows')
    result={}
    for arm in ['Source','Full']:
        values=[r[arm]['wall_seconds']for r in rows];peaks=[r[arm]['peak_allocated_cuda_bytes']for r in rows]
        result[arm]=dict(queries=len(rows),wall_seconds_mean=sum(values)/len(values),wall_seconds_min=min(values),wall_seconds_max=max(values),peak_allocated_cuda_bytes_max=max(peaks),peak_allocated_cuda_bytes_mean=sum(peaks)/len(peaks))
    result['Full_to_Source_wall_ratio']=result['Full']['wall_seconds_mean']/result['Source']['wall_seconds_mean']
    return result


def benchmark(root,backbone,dataset):
    # This gate runs before setup/lease/CUDA/model imports, and reads no labels.
    records,output,seal,run=completed_job(root,backbone,dataset)
    path=output/'benchmark.json'
    if path.exists():raise FileExistsError('A benchmark receipt already exists; never overwrite it')
    selected=select_records(records)
    if not selected:raise ValueError('Cannot benchmark an empty cohort')
    if 'models'in sys.modules:raise RuntimeError('Start each target benchmark in a fresh Python process')
    import numpy as np
    import torch
    from stvg.runtime import setup,lease,load_source
    from stvg.predictor import Predictor
    from stvg.worker import ResidentSession,runtime_fingerprint,atomic_json
    setup();lock=lease()
    try:
        fingerprint,proof=runtime_fingerprint(backbone)
        if fingerprint!=run['runtime_fingerprint']:raise ValueError('Current frozen runtime differs from the sealed job')
        with ResidentSession(fingerprint,cache_dir=None,visual_cache=None,backbone=backbone)as session:
            tick=time.perf_counter();predictor=Predictor(dataset);predictor.source=load_source(predictor.config);predictor.source.cpu();torch.cuda.synchronize()
            source_load_seconds=time.perf_counter()-tick
            predictor.teacher_service=session.teacher;session.predictors[dataset]=predictor
            def measure(record):
                session.decoded=None
                src,ids,source_cost=source_only(predictor,record)
                session.decoded=None
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                full,full_ids=session.predict(record)
                torch.cuda.synchronize();wall=time.perf_counter()-tick
                if session.visual_cache is not None or session.teacher.cache_dir is not None:raise AssertionError('Benchmark caching unexpectedly enabled')
                if full.get('engineering',{}).get('decode_cache_hit'):raise AssertionError('Benchmark reused decoded RGB')
                if full['capture_cost'].get('ReferDINO_query_forwards')!=1:raise AssertionError('Benchmark did not execute the full eight-frame expert forward')
                worker_peak=int(torch.cuda.max_memory_allocated());teacher_peak=int(full['capture_cost']['peak_GPU_bytes'])
                require_equal(ids,full_ids,'Source/Full physical frame grid')
                reference=output/seal['files'][record['id']]['path'];verify_outputs(src,full,ids,reference)
                frozen=all(not p.requires_grad and p.grad is None for p in predictor.source.parameters())
                if not frozen:raise AssertionError('Frozen Source accumulated gradients')
                return dict(id=record['id'],parent=parent_key(record),physical_nodes=len(ids),video_sha256=record['metadata']['video_sha256'],Source=source_cost,
                            Full=dict(wall_seconds=wall,peak_allocated_cuda_bytes=max(worker_peak,teacher_peak)),
                            model_load_seconds=dict(source=0.,teacher=float(full['capture_cost'].get('ReferDINO_load_seconds',0.))),
                            bitwise_sealed_regression=True,GT_read=False)
            # Independently run first; then rerun it as part of the fixed measured cohort.
            warmup=measure(selected[0]);measured=[measure(r)for r in selected]
            result=dict(status='passed',backbone=backbone,dataset=dataset,manifest_sha256=run['manifest_sha256'],runtime_fingerprint=fingerprint,
                        selection='sort len(frame_ids),video_sha256,id; first query per parent; uniform ranks, at most 16; no outcome or labels used',
                        sample_parents=len(selected),warmup_excluded_from_means=True,warmup=warmup,
                        model_load_seconds=dict(source=source_load_seconds,teacher=warmup['model_load_seconds']['teacher']),rows=measured,summary=aggregate(measured),
                        caches=dict(RGB=False,visual=False,teacher_results=False,model_weights_resident=True),
                        timing_scope='CUDA-synchronized wall from decode through parse and final Source MBR / complete eight-frame ReferDINO, Strong and OPD; weight loads reported separately; warmup excluded',
                        memory_scope='max allocated CUDA bytes in worker versus teacher; not sum, not device-reserved memory or nvidia-smi',GT_read=False)
            atomic_json(path,result);return result
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone',choices=['tastvg','cgstvg'],required=True)
    parser.add_argument('--dataset',choices=['hc2','vidstg'],required=True)
    args=parser.parse_args()
    result=benchmark(ROOT,args.backbone,args.dataset)
    print(json.dumps(dict(status=result['status'],backbone=args.backbone,dataset=args.dataset,parents=result['sample_parents'],summary=result['summary']),indent=2))

if __name__=='__main__':main()
