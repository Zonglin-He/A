"""Bounded frozen feature VJP capture, before any diagnostic GT access."""
import argparse,gc,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,load,save,sha
OUT=ROOT/'artifacts/actionness_spatial_attribution_v1'
PARENT=ROOT/'artifacts/spatial_evidence_retention_v1'
CODE=['scripts/run_actionness_spatial_attribution_v1.py','vg_tta/actionness_spatial_attribution_v1.py',
      'protocols/actionness_spatial_attribution_v1.md','tests/test_actionness_spatial_attribution_v1.py']


def prepare():
    from vg_tta.actionness_spatial_attribution_v1 import select_donors
    p=read(PARENT/'LOCK.json');inputs=[]
    for r in p['rows']:
        assert sha(r['path'])==r['sha256'];x=load(r['path'])
        inputs.append(dict(key=r['key'],source=r['source'],cohort=r['cohort'],
                           caption=x['input']['caption'],subject=x['parses']['subject']))
    donors=select_donors(inputs)
    write(OUT/'LOCK.json',dict(rows=p['rows'],donors=donors,production_pins=p['production_pins'],
        parent_lock_sha256=sha(PARENT/'LOCK.json'),parent_barrier_sha256=sha(PARENT/'PREDICTION_BARRIER.json'),
        parent_frame_results_sha256=sha(PARENT/'analysis_v2/FRAME_RESULTS.pt'),
        code={f:sha(ROOT/f) for f in CODE},max_gpu_seconds=3600,historically_exposed=True,
        labels=p['labels'],labels_sha256=p['labels_sha256'],model='TA-STVG',TTA=False,experts=False,
        checkpoint_directions=['VidSTG -> HC1','HC2 -> VidSTG'],GT_donors=False,
        primary='gradcam',secondary='grad_times_activation',bootstrap_seed=20260918,
        bootstrap_replicates=10000,created=time.time()))
    print('LOCKED 64, same-noun wrong-query controls',sum(d['matched_query_noun'] for d in donors.values()),flush=True)


def verify():
    p=read(OUT/'LOCK.json')
    assert sha(PARENT/'LOCK.json')==p['parent_lock_sha256']
    assert sha(PARENT/'PREDICTION_BARRIER.json')==p['parent_barrier_sha256']
    code=dict(p['code'])
    for revision in sorted(OUT.glob('CAPTURE_CODE_REVISION_*.json')):
        for r in read(revision)['files']:
            assert code[r['file']]==r['old_sha256']
            code[r['file']]=r['new_sha256']
    for f,h in {**code,**p['production_pins']}.items():assert sha(ROOT/f)==h,f
    return p


def pause():
    base=ROOT/'artifacts/decota_paper_execution_20260917';s=read(base/'background_last/STATUS.json')
    if s.get('status')!='running':return None
    pid=s['pid'];proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists():return None
    assert 'scripts/run_decota_paper_background_last_20260917.py' in proc.read_bytes().replace(b'\0',b' ').decode()
    snap=dict(coordinator=s,worker=read(base/'vid_last/native/full/STATUS.json'),time=time.time())
    write(OUT/f'BASELINE_PAUSE_{int(time.time())}.json',snap);os.kill(pid,signal.SIGTERM)
    for _ in range(120):
        if not proc.exists() and not Path(f'/proc/{s["child_pid"]}').exists():break
        time.sleep(.25)
    else:raise RuntimeError('Baseline did not stop at receipt boundary')
    return snap


def resume(snapshot):
    if snapshot is None:return
    with (OUT/'baseline_resume.log').open('a') as log:
        p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-u','-B',
            str(ROOT/'scripts/run_decota_paper_background_last_20260917.py')],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/f'BASELINE_RESUME_{int(time.time())}.json',dict(pid=p.pid,previous=snapshot,
          new_automation=False,existing_finite_queue=True,time=time.time()))


def run(limit=0):
    import numpy as np,torch
    from scripts.run_final_simplification_v1 import lease
    from scripts.run_spatial_regression_alignment_v1 import model_load
    from scripts.run_spatial_ssl_gpu_v1 import frozen_forward
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.actionness_spatial_attribution_v1 import capture
    from methods.decota_final_simplified_v1.tensors import state_hash
    p=verify();snapshot=None;guard=None;model=None;current=None;mhash=None;done=0;begin=time.time()
    torch.set_num_threads(4);torch.manual_seed(20260910);np.random.seed(20260910)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:
        snapshot=pause();guard=lease()
        for r in p['rows']:
            if limit and done>=limit:break
            stem=r['key'].replace(':','_')
            finished=True
            for arm in ['correct','wrong']:
                f=OUT/'captures'/arm/f'{stem}.pt'
                if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256']
                else:finished=False
            if finished:done+=1;continue
            assert time.time()-begin<p['max_gpu_seconds']
            if r['cohort']!=current:
                if model is not None:
                    assert state_hash(model.state_dict())==mhash
                    del model;gc.collect();torch.cuda.empty_cache()
                model=model_load(r['cohort']);current=r['cohort'];mhash=state_hash(model.state_dict())
            assert sha(r['path'])==r['sha256'];parent=load(r['path'])
            x=dict(input=parent['input'],frame_ids=parent['frame_ids'],parses=dict(subject=parent['parses']['subject']))
            del parent
            frames,ids=decode(x['input']);assert ids==x['frame_ids']
            old=load(PARENT/'captures'/f'{stem}.pt')
            for arm in ['correct','wrong']:
                f=OUT/'captures'/arm/f'{stem}.pt'
                if f.with_suffix('.json').exists():continue
                xx=dict(x)
                if arm=='wrong':
                    d=p['donors'][r['key']]
                    xx.update(input={**x['input'],'caption':d['caption']},parses=dict(subject=d['subject']))
                tick=time.time();torch.cuda.reset_peak_memory_stats()
                status(OUT/'STATUS.json',dict(status='running',done=done,total=64,key=r['key'],arm=arm,
                                            pid=os.getpid(),updated=time.time()))
                batch,records,s=frozen_forward(model,frames,xx)
                versions={n:v._version for n,v in model.state_dict(keep_vars=True).items()}
                z=capture(s)
                assert versions=={n:v._version for n,v in model.state_dict(keep_vars=True).items()}
                assert not any(v.requires_grad or v.grad is not None for v in model.parameters())
                if arm=='correct':
                    assert torch.equal(z['boxes'],old['boxes'][:,-1]),'parent native boxes differ'
                    assert z['gates']==old['gates'],'parent gates differ'
                z.update(key=r['key'],source=r['source'],cohort=r['cohort'],arm=arm,input=xx['input'],
                    subject=xx['parses']['subject'],frame_ids=ids,records=records,checkpoint_state_hash=mhash,
                    query_donor=p['donors'][r['key']] if arm=='wrong' else None,model_unchanged=True,
                    seconds=time.time()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated())
                save(f,z);write(f.with_suffix('.json'),dict(sha256=sha(f),key=r['key'],arm=arm,
                    lock_sha256=sha(OUT/'LOCK.json'),seconds=z['seconds']))
                print('CAPTURE',done+1,arm,r['key'],round(z['seconds'],2),'s',flush=True)
                del z,s,batch;gc.collect();torch.cuda.empty_cache()
            done+=1;del old,x,xx,frames
        if model is not None:assert state_hash(model.state_dict())==mhash
        verify();status(OUT/'STATUS.json',dict(status='completed' if done==64 else 'smoke_complete',
            done=done,total=64,seconds=time.time()-begin,pid=os.getpid(),updated=time.time()))
        if done==64:
            files={str(f.relative_to(ROOT)):sha(f) for f in sorted((OUT/'captures').glob('*/*.pt'))}
            assert len(files)==128
            write(OUT/'PREDICTION_BARRIER.json',dict(files=files,queries=64,forward_arms=128,GT_access=False,
                  parameter_updates=0,experts=0,lock_sha256=sha(OUT/'LOCK.json'),time=time.time()))
    except BaseException as exc:
        write(OUT/f'FAILURE_{int(time.time())}.json',dict(error=repr(exc),done=done,time=time.time()));raise
    finally:
        if model is not None:del model;gc.collect();torch.cuda.empty_cache()
        if guard is not None:guard.close()
        resume(snapshot)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();prepare() if a.stage=='prepare' else run(a.limit)
