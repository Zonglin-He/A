"""One shared native224/eight-frame capture per development query, before tuning."""
import argparse, hashlib, json, sys, time, gc, traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal.run import read,sha,atomic,savept
from scripts.tube_fusion.core import head_logits
OUT=ROOT/'outputs/boxfixed_tuning_v1'

def prepare():
    manifest=ROOT/'data/evaluation/development.jsonl'
    records=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip()]
    selected={ds:[r for r in records if r['dataset']==ds][:64]for ds in ('hc2','vidstg')}
    assert all(len(x)==len({r['metadata']['source']for r in x})==64 for x in selected.values())
    from stvg.worker import runtime_fingerprint
    fp,proof=runtime_fingerprint('tastvg')
    identity=dict(runtime_fingerprint=fp,code=proof['code'],manifest_sha256=sha(manifest),
        formal_sha256=sha(ROOT/'outputs/formal/state.json'),default_sha256=sha(ROOT/'configs/default.json'),
        selection={ds:[r['id']for r in rows]for ds,rows in selected.items()},
        purpose='new user authorized bounded paired tuning then new full queue; no old formal resume',GT_read=False)
    identity['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    p=OUT/'capture_lock.json'
    if p.exists():assert read(p)==identity
    else:
        atomic(p,identity);atomic(OUT/'runtime.json',proof)
        for rel in proof['code']:
            q=OUT/'snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
    return identity,selected

@torch.no_grad()
def run(ds):
    lock,selected=prepare();rows=selected[ds];dest=OUT/'captures'/ds
    assert not(dest/'run.json').exists(),'Never overwrite captures'
    from stvg.runtime import setup,lease
    from stvg.predictor import Predictor
    from stvg.teacher_bridge import ResidentTeacher
    from stvg.decode import decode
    from stvg.inputs import validate_input
    from stvg.worker import resource_guard
    setup();gpu=lease();model=Predictor(ds,check=resource_guard)
    expert=ResidentTeacher(OUT/'teacher_cache',lock['runtime_fingerprint']);model.teacher_service=expert
    atomic(dest/'run.json',dict(status='running',GT_read=False));files={};begin=time.perf_counter()
    try:
        for i,r in enumerate(rows):
            start=time.perf_counter();m=r['metadata'];frames,ids=decode(m,ds);validate_input(frames,ids,m)
            decode_seconds=time.perf_counter()-start
            torch.manual_seed(20260920);np.random.seed(20260920)
            cap=model.capture(frames,ids,m,temporal_mode='native224',export_replay=True);del frames
            tc=cap['temporal_cache'];head=model.source.temp_embed.cuda()
            logits,cost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu()
            q=dest/(r['id']+'.pt')
            savept(q,dict(Source=cap['Source'],E=cap['E'],frame_ids=ids,positions8=cap['planned_positions'],
                geometry=cap['geometry'].state_dict(),records=cap['records'],logits=logits,
                native_interval=cap['intervals']['Native'],GT_read=False))
            rec=dict(fingerprint=lock['fingerprint'],path=q.name,sha256=sha(q),GT_read=False,
                Source224_captures=1,expert_forwards=cap['cost']['ReferDINO_query_forwards'],
                Source448_calls=cap['cost']['Source448_calls'],extra_decoder_calls=0,updates=0,
                final_head_exact=True,capture_cost=cap['cost'],head_cost=cost,RGB_seconds=decode_seconds,
                query_wall_seconds=time.perf_counter()-start)
            atomic(dest/(r['id']+'.json'),rec);files[r['id']]=rec
            atomic(dest/'progress.json',dict(completed=i+1,total=64,seconds=time.perf_counter()-begin,GT_read=False))
            print(json.dumps(dict(dataset=ds,completed=i+1,total=64)),flush=True)
            del cap,tc,logits;gc.collect();torch.cuda.empty_cache()
        atomic(dest/'sealed.json',dict(fingerprint=lock['fingerprint'],files=files,queries=64,GT_read=False))
        atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-begin,GT_read=False))
    except BaseException:
        atomic(dest/'failure.json',dict(completed=len(files),traceback=traceback.format_exc()));raise
    finally:expert.close();gpu.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg']);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:lock,_=prepare();print(lock['fingerprint'])
    elif a.dataset:run(a.dataset)
    else:p.error('dataset or prepare required')
