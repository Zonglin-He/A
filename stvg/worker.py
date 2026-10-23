"""Label-free sequential inference with frozen models resident on CPU.

No optimizer or content adaptation is part of this main flow. Evidence cache
hits require exact RGB, raw text, physical positions and a verified runtime key.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import shutil

import numpy as np
import torch

from .runtime import ROOT, load_source, setup, lease, sha
from .predictor import Predictor
from .teacher_bridge import ResidentTeacher
from .inputs import validate_metadata, validate_input
from .decode import decode


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def runtime_fingerprint(backbone="tastvg"):
    """Hash complete frozen assets once at startup, never once per query."""
    assets=json.loads((ROOT/'docs/assets.json').read_text())['required_files']
    checked={}
    for relative,expected in sorted(assets.items()):
        actual=sha(ROOT/relative)
        if actual!=expected:raise ValueError('Frozen asset checksum changed: '+relative)
        checked[relative]=actual
    if backbone!='tastvg':raise ValueError('Only the TA-STVG main method is supported')
    paths=set()
    for directory in ['stvg','external/TA-STVG','external/ReferDINO','runtime/referdino/python']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not {'.git','__pycache__','build'}.intersection(path.parts) and path.suffix in {'.py','.yaml','.yml','.json','.cpp','.cu','.cuh','.h'}:
                paths.add(path)
    paths.update((ROOT/relative)for relative in ['configs/default.json','requirements.txt','requirements-teacher.txt','docs/assets.json'])
    paths.add(ROOT/'runtime/referdino/cuda_build/MultiScaleDeformableAttention.so')
    code={str(p.relative_to(ROOT)):sha(p)for p in sorted(paths)}
    versions={}
    for package in ['torch','torchvision','transformers','numpy','scipy','stanza','Pillow','einops','timm','tokenizers','ruamel.yaml','easydict','addict','yacs','supervision']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='runtime-target-or-not-installed'
    proof=dict(schema=1,backbone=backbone,assets=checked,code=code,packages=versions,python=sys.version,
               torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    return hashlib.sha256(canonical(proof)).hexdigest(),proof


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(canonical(value)+b'\n');os.replace(tmp,path)


def resource_guard():
    # Fail the owned job explicitly; never kill or signal unrelated services.
    mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))}
    if mem.get('MemAvailable',0)<1536*2**20:raise RuntimeError('Resource guard: less than 1.5 GiB available RAM')
    if shutil.disk_usage(ROOT).free<5*2**30:raise RuntimeError('Resource guard: less than 5 GiB free disk')


class ResidentSession:
    """An ordered set of independent queries sharing only frozen resources."""
    def __init__(self, fingerprint, cache_dir=None, visual_cache=None, backbone="tastvg",temporal_mode="native2d"):
        self.temporal_mode=temporal_mode
        self.fingerprint=fingerprint;self.backbone=backbone
        if backbone!='tastvg':raise NotImplementedError('Only TA-STVG BoxFixed+Native2D is supported')
        self.teacher=ResidentTeacher(cache_dir=cache_dir,fingerprint=fingerprint)
        self.predictors={};self.visual_cache=visual_cache;self.decoded=None;self.decode_hits=0;self.decode_misses=0
    def predict(self,record):
        resource_guard()
        dataset=record['dataset'];meta=record['metadata'];validate_metadata(meta)
        if dataset not in ['hc2','vidstg']:raise ValueError('Unsupported dataset')
        decode_start=time.perf_counter()
        decode_key=canonical(dict(dataset=dataset,video_path=str(Path(meta['video_path']).resolve()),video_sha256=meta['video_sha256'],width=meta['width'],height=meta['height'],frame_ids=meta['frame_ids']))
        decode_hit=self.decoded is not None and self.decoded[0]==decode_key
        if decode_hit:
            if sha(meta['video_path'])!=meta['video_sha256']:raise ValueError('Cached video bytes changed')
            _,frames,ids=self.decoded;self.decode_hits+=1
        else:
            frames,ids=decode(meta,dataset);self.decoded=(decode_key,frames,ids);self.decode_misses+=1
        validate_input(frames,ids,meta);decode_seconds=time.perf_counter()-decode_start
        torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats()
        source_load_start=time.perf_counter();source_first_load=dataset not in self.predictors
        if dataset not in self.predictors:
            predictor=Predictor(dataset,check=resource_guard);predictor.source=load_source(predictor.config)
            predictor.teacher_service=self.teacher;self.predictors[dataset]=predictor
        source_load_seconds=time.perf_counter()-source_load_start if source_first_load else 0.
        predictor=self.predictors[dataset]
        # Optional pure-visual context supplied by separately verified engineering code.
        ctx=self.visual_cache.context(predictor.source,frames,ids)if self.visual_cache else contextlib.nullcontext()
        try:
            with ctx:out=predictor.predict(frames,ids,meta,record['key'],temporal_mode=self.temporal_mode)
        finally:
            predictor.source.cpu()
            if any(p.requires_grad or p.grad is not None for p in predictor.source.parameters()):
                raise AssertionError('A frozen network acquired trainable state or gradients')
            torch.cuda.empty_cache()
        out['engineering']=dict(worker_peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),worker_peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),source_model_load_wall_seconds=source_load_seconds,decode_seconds=decode_seconds,decode_cache_hit=decode_hit,decode_hits=self.decode_hits,decode_misses=self.decode_misses,
                                visual_cache=self.visual_cache.stats()if self.visual_cache else None)
        return out,ids
    def close(self):self.teacher.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def prediction_arrays(out,ids):
    arrays=dict(Source=out['Source'],E_Ref=out['E_Ref'],BoxFixed=out['BoxFixed'],frame_ids=np.asarray(ids),positions8=np.asarray(out['positions8']),Native_interval=np.asarray(out['intervals']['Native']),Selected_interval=np.asarray(out['interval']))
    arrays['Global6_interval']=np.asarray(out['interval'])
    return arrays


def request_hash(record):
    # Input key is included: shared evidence never changes per-query output binding.
    return hashlib.sha256(canonical(record)).hexdigest()


def run(manifest,output,cache_dir=None,resume_hash=None,visual_cache=None,backbone="tastvg",temporal_mode="native2d"):
    """One manifest; no labels argument, imports, scoring or target selection."""
    rows=[json.loads(s)for s in Path(manifest).read_text().splitlines()if s.strip()]
    if len({r['id']for r in rows})!=len(rows):raise ValueError('Repeated evaluation IDs')
    for r in rows:
        if set(r)!={'id','dataset','key','metadata'}:raise ValueError('Expected label-free id/dataset/key/metadata records')
        if not r['id']or not all(c.isalnum()or c in '_-'for c in r['id']):raise ValueError('Unsafe query ID')
        validate_metadata(r['metadata'])
    fingerprint,proof=runtime_fingerprint(backbone)
    method_hash=hashlib.sha256(canonical(dict(manifest_sha256=sha(manifest),runtime_fingerprint=fingerprint,backbone=backbone,temporal_mode=temporal_mode))).hexdigest()
    if resume_hash is not None and resume_hash!=method_hash:raise ValueError('Requested resume hash does not match this run')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    run_path=root/'run.json'
    run_info=dict(schema=2,temporal_mode=temporal_mode,backbone=backbone,manifest_sha256=sha(manifest),runtime_fingerprint=fingerprint,resume_hash=method_hash,labels_read=False,queries=len(rows))
    if run_path.exists():
        if json.loads(run_path.read_text())!=run_info:raise ValueError('Existing outputs belong to another run')
    else:atomic_json(run_path,run_info);atomic_json(root/'runtime_fingerprint.json',proof)
    files={}
    with ResidentSession(fingerprint,cache_dir,visual_cache,backbone,temporal_mode)as session:
        for record in rows:
            key=record['id'];dest=root/'predictions'/(key+'.npz');receipt=dest.with_suffix('.json');binding=request_hash(record)
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old.get('resume_hash')!=method_hash or old.get('request_sha256')!=binding or not dest.exists() or sha(dest)!=old.get('prediction_sha256'):
                    raise ValueError('Saved query is not bound to this exact run: '+key)
            else:
                if dest.exists():raise ValueError('Unbound prediction exists; refusing overwrite: '+key)
                start=time.perf_counter();out,ids=session.predict(record);dest.parent.mkdir(parents=True,exist_ok=True)
                tmp=dest.with_suffix('.npz.tmp')
                with tmp.open('wb')as handle:
                    np.savez(handle,**prediction_arrays(out,ids))
                os.replace(tmp,dest)
                atomic_json(receipt,dict(resume_hash=method_hash,request_sha256=binding,prediction_sha256=sha(dest),configuration=session.predictors[record['dataset']].config,labels_read=False,total_seconds=time.perf_counter()-start,capture=out['capture_cost'],spatial_and_temporal=out['cost'],temporal_mode=out['temporal_mode'],engineering=out.get('engineering',{})))
            files[key]=dict(path=str(dest.relative_to(root)),sha256=sha(dest))
            progress=dict(completed=len(files),total=len(rows),id=key,resume_hash=method_hash)
            atomic_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(root/'sealed.json',dict(status='sealed_before_offline_scoring',schema=2,temporal_mode=temporal_mode,queries=len(rows),manifest_sha256=sha(manifest),files=files,labels_read=False,resume_hash=method_hash,runtime_fingerprint=fingerprint))
    return method_hash


def main():
    p=argparse.ArgumentParser(description='Resident STVG inference; no labels or scoring')
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backbone',choices=['tastvg'],default='tastvg')
    p.add_argument('--temporal',choices=['native2d'],default='native2d')
    p.add_argument('--teacher-cache');p.add_argument('--resume-hash')
    p.add_argument('--visual-cache-mib',type=int,default=4096)
    args=p.parse_args();setup();lock=lease()
    from .visual_cache import VisualCache
    try:run(args.manifest,args.output,args.teacher_cache,args.resume_hash,VisualCache(args.visual_cache_mib*2**20)if args.visual_cache_mib else None,args.backbone,args.temporal)
    finally:lock.close()

if __name__=='__main__':main()
