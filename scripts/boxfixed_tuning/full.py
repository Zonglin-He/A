"""NEW immutable full queue for tuned output-only BoxFixed; never resume old formal."""
import argparse,fcntl,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.tube_temporal.run import read,sha,atomic,savept
from scripts.boxfixed_tuning.core import BASE,temporal,spatial
from scripts.boxfixed_tuning.cache import binding,RGBCache,ReusingTeacher,from_native
from scripts.tube_fusion.core import head_logits
RUN=ROOT/'outputs/boxfixed_full_v1';TUNE=ROOT/'outputs/boxfixed_tuning_v1'

def identity(pid=None):
    pid=os.getpid()if pid is None else int(pid)
    return dict(pid=pid,startticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]))

def alive(item):
    try:return identity(item['pid'])=={k:item[k]for k in ('pid','startticks')}
    except (FileNotFoundError,ProcessLookupError,KeyError):return False

def validate_plan():
    p=read(RUN/'plan.json')
    assert sha(ROOT/'configs/tuned_boxfixed_v1.json')==p['profile_sha256']
    assert sha(ROOT/'configs/default.json')==p['default_sha256']
    assert sha(ROOT/'outputs/formal/state.json')==p['old_formal_state_sha256']
    for rel,h in p['sources'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in p['assets'].items():assert sha(ROOT/rel)==h,rel
    assert sha(RUN/'capture_index.json')==p['capture_index_sha256']
    for job in p['jobs'].values():assert sha(ROOT/job['manifest'])==job['manifest_sha256']
    return p

def prepare():
    if(RUN/'plan.json').exists():return validate_plan()
    from stvg.worker import runtime_fingerprint
    fp,proof=runtime_fingerprint('tastvg');profile=read(ROOT/'configs/tuned_boxfixed_v1.json')
    assert read(ROOT/'results/boxfixed_tuning_v1/cache_regression.json')['status']=='pass'
    old=read(ROOT/'outputs/formal/plan.json');jobs={}
    for ds,n in [('hc2',1902),('vidstg',10303)]:
        manifest=ROOT/f'data/evaluation/{ds}.jsonl';rows=[json.loads(s)for s in manifest.read_text().splitlines()]
        assert len(rows)==len({r['id']for r in rows})==n and all(r['dataset']==ds for r in rows)
        j=next(j for j in old['jobs']if j['id']=='tastvg-'+ds)
        jobs[ds]=dict(queries=n,manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest),labels=j['labels'],labels_sha256=j['labels_sha256'])
    sources={**proof['code'],**{str(p.relative_to(ROOT)):sha(p)for p in Path(__file__).parent.glob('*.py')}}
    p=dict(runtime_fingerprint=fp,profile=profile,profile_sha256=sha(ROOT/'configs/tuned_boxfixed_v1.json'),
        default_sha256=sha(ROOT/'configs/default.json'),old_formal_state_sha256=sha(ROOT/'outputs/formal/state.json'),
        sources=sources,assets=proof['assets'],jobs=jobs,methods=['Source','Untuned','Final'],
        numerical_cache_proof_sha256=sha(ROOT/'results/boxfixed_tuning_v1/cache_regression.json'),
        cache_policy='512MiB exact RGB and4GiB actual-tensor visual LRU. No text/fused/decoder sharing across queries. Complete-capture reuse keyed by all metadata except location plus runtime; per-file hash required.',GT_read=False)
    build_index(p);p['capture_index_sha256']=sha(RUN/'capture_index.json')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    atomic(RUN/'plan.json',p);atomic(RUN/'runtime.json',proof)
    for rel in sources:
        q=RUN/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/rel).read_bytes())
    print(json.dumps(dict(status='prepared',fingerprint=p['fingerprint'],queries=jobs)))
    return p

def build_index(plan):
    dev=[json.loads(s)for s in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()];index={}
    hc=read(TUNE/'hc2_inputs.json')['files'];vs=read(TUNE/'captures/vidstg/sealed.json')['files']
    for ds,rows in [('hc2',[r for r in dev if r['dataset']=='hc2'][:64]),('vidstg',[r for r in dev if r['dataset']=='vidstg'][:64])]:
        for r in rows:
            if ds=='hc2':item=hc[r['id']]
            else:
                v=vs[r['id']];item=dict(path=str((TUNE/'captures'/ds/v['path']).relative_to(ROOT)),sha256=v['sha256'])
            assert sha(ROOT/item['path'])==item['sha256'];index[binding(r,plan['runtime_fingerprint'])]=item
    selected=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection']
    for ds,rows in selected.items():
        tr=ROOT/'outputs/interval_pool'/('recheck_'+ds)
        for item in rows:
            key=item['query_id'];r=item['record'];ref=read(tr/'receipts'/f'{key}.json')
            assert sha(tr/ref['capture_path'])==ref['capture_sha256']and sha(tr/ref['posterior_path'])==ref['posterior_sha256']
            c=torch.load(tr/ref['capture_path'],map_location='cpu',weights_only=False);post=torch.load(tr/ref['posterior_path'],map_location='cpu',weights_only=False)
            with np.load(tr/'predictions'/f'{key}.npz',allow_pickle=False)as z:native=z['Source_Native224_interval'].copy()
            cap=dict(Source=c['Source'],E=c['E'],frame_ids=c['frame_ids'],positions8=c['planned_positions'],records=c['records'],geometry=c['geometry'],logits=post['logits'],native_interval=native,GT_read=False)
            dest=RUN/'imported_inputs'/ds/f'{key}.pt';savept(dest,cap)
            index[binding(r,plan['runtime_fingerprint'])]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest))
    atomic(RUN/'capture_index.json',dict(files=index,GT_read=False,entries=len(index)))

@torch.no_grad()
def worker(ds):
    plan=validate_plan();job=plan['jobs'][ds];params=plan['profile']['datasets'][ds]
    from stvg.runtime import setup,lease,load_source
    from stvg.predictor import Predictor
    from stvg.visual_cache import VisualCache
    from stvg.inputs import validate_input
    from stvg.worker import resource_guard
    setup();gpu=lease();dest=RUN/ds;index=read(RUN/'capture_index.json')['files']
    model=Predictor(ds,check=resource_guard);expert=ReusingTeacher(RUN/'teacher_cache',plan['runtime_fingerprint'],
        previous=[TUNE/'teacher_cache',ROOT/'outputs/interval_pool/teacher_cache']);model.teacher_service=expert
    visual=VisualCache(4*2**30);rgb=RGBCache();rows=[json.loads(s)for s in (ROOT/job['manifest']).read_text().splitlines()]
    entries={};begin=time.perf_counter();recent=[]
    atomic(dest/'worker.json',dict(**identity(),fingerprint=plan['fingerprint'],GT_read=False))
    try:
        for i,r in enumerate(rows):
            resource_guard();key=r['id'];rp=dest/'receipts'/f'{key}.json';pp=dest/'predictions'/f'{key}.npz'
            if rp.exists():
                rec=read(rp);assert rec['fingerprint']==plan['fingerprint']and rec['binding']==binding(r,plan['runtime_fingerprint'])and sha(pp)==rec['prediction_sha256']
                entries[key]=dict(path=str(pp.relative_to(dest)),sha256=sha(pp),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp));continue
            if pp.exists():
                archived=dest/'orphaned'/f'{key}.{time.time_ns()}.npz';archived.parent.mkdir(parents=True,exist_ok=True);pp.rename(archived)
            start=time.perf_counter();b=binding(r,plan['runtime_fingerprint']);capture_cost={};vc=visual.stats();rc={};source_load=0.;source_peak=expert_peak=None;load_peak=0
            if b in index:
                item=index[b];assert sha(ROOT/item['path'])==item['sha256']
                assert sha(r['metadata']['video_path'])==r['metadata']['video_sha256']
                cap=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);reuse=dict(complete_capture_hit=True,path=item['path'],sha256=item['sha256'])
            else:
                frames,ids,rc=rgb.get(r['metadata'],ds);validate_input(frames,ids,r['metadata'])
                if model.source is None:
                    tick=time.perf_counter();model.source=load_source(model.config);source_load=time.perf_counter()-tick;load_peak=torch.cuda.max_memory_allocated()
                torch.manual_seed(20260920);np.random.seed(20260920);torch.cuda.reset_peak_memory_stats()
                with visual.context(model.source,frames,ids):native=model.capture(frames,ids,r['metadata'],temporal_mode='native224',export_replay=True)
                tc=native['temporal_cache'];head=model.source.temp_embed.cuda()
                logits,hcost=head_logits(head,[v['hidden']for v in tc['offsets']],[v['native_logits']for v in tc['offsets']]);head.cpu()
                capture_cost=dict(native['cost']);capture_cost['six_layer_head']=hcost
                cap=from_native(native,logits);reuse=dict(complete_capture_hit=False)
                source_peak=max(load_peak,torch.cuda.max_memory_allocated());expert_peak=native['teacher_native'].get('peak_GPU_bytes')
                del native,tc,logits,frames
            assert cap['GT_read']is False and list(cap['frame_ids'])==r['metadata']['frame_ids']
            cap['fps']=float(r['metadata']['fps']);tb=temporal(cap,BASE['temperature']);untuned=spatial(cap,tb['interval'],BASE)
            tt=tb if params['temperature']==BASE['temperature'] else temporal(cap,params['temperature']);tuned=untuned if params==BASE else spatial(cap,tt['interval'],params)
            pp.parent.mkdir(parents=True,exist_ok=True);np.savez(pp,Source=cap['Source'],Source_interval=cap['native_interval'],
                Untuned=untuned['final'].numpy(),Untuned_interval=tb['interval'],Final=tuned['final'].numpy(),Final_interval=tt['interval'],
                E_Ref=cap['E'],frame_ids=cap['frame_ids'],positions8=cap['positions8'])
            after=visual.stats();rec=dict(fingerprint=plan['fingerprint'],binding=b,GT_read=False,prediction_sha256=sha(pp),
                reused=reuse,capture_cost=capture_cost,RGB=rc,Source_initial_load_seconds=source_load,
                visual_cache_before=vc,visual_cache_after=after,visual_hits=after['hits']-vc['hits'],visual_misses=after['misses']-vc['misses'],
                posterior_seconds=dict(Untuned=tb['seconds'],Final=0. if tt is tb else tt['seconds']),DP_seconds=dict(Untuned=untuned['costs']['seconds'],Final=0. if params==BASE else tuned['costs']['seconds']),
                Final_aliases_Untuned=params==BASE,
                query_wall_seconds=time.perf_counter()-start,params=params,updates=0,Source448_calls=0,extra_decoder_calls=0)
            rec.update(Source_process_peak_allocated_GPU_bytes=source_peak,expert_process_peak_allocated_GPU_bytes=expert_peak)
            atomic(rp,rec);entries[key]=dict(path=str(pp.relative_to(dest)),sha256=sha(pp),receipt_path=str(rp.relative_to(dest)),receipt_sha256=sha(rp))
            recent.append(rec['query_wall_seconds']);recent=recent[-100:]
            atomic(dest/'progress.json',dict(completed=i+1,total=len(rows),dataset=ds,**identity(),fingerprint=plan['fingerprint'],
                seconds=time.perf_counter()-begin,recent_query_seconds=float(np.mean(recent)),remaining_estimate_seconds=float(np.mean(recent))*(len(rows)-i-1),
                visual=after,RGB_hits=rgb.hits,RGB_misses=rgb.misses,GT_read=False))
            if(i+1)%10==0:print(json.dumps(dict(dataset=ds,completed=i+1,total=len(rows),recent_seconds=float(np.mean(recent)))),flush=True)
            del cap,untuned,tuned,tb,tt;gc.collect();torch.cuda.empty_cache()
        validate_plan();assert len(entries)==job['queries']
        atomic(dest/'sealed.json',dict(files=entries,queries=len(entries),fingerprint=plan['fingerprint'],GT_read=False))
        atomic(dest/'run.json',dict(status='completed_sealed',seconds=time.perf_counter()-begin,queries=len(entries)))
    except BaseException:
        atomic(dest/'failure.json',dict(completed=len(entries),traceback=traceback.format_exc(),**identity()));raise
    finally:expert.close();gpu.close()

def controller(resume):
    plan=validate_plan();lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=RUN/'state.json'
    if statepath.exists():
        assert resume,'Existing queue requires explicit --resume';old=read(statepath)
        assert old['fingerprint']==plan['fingerprint']
        for item in old.get('children',[]):assert not alive(item),'An old owned child is still alive; do not duplicate'
    state=dict(status='running',fingerprint=plan['fingerprint'],controller=identity(),children=[],jobs={})
    atomic(statepath,state)
    try:
        for ds in ('hc2','vidstg'):
            with(RUN/(ds+'.log')).open('ab')as log:
                child=subprocess.Popen([sys.executable,'-B',__file__,'--worker',ds],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['children']=[dict(**identity(child.pid),role='worker',dataset=ds)];state['jobs'][ds]='running';atomic(statepath,state)
                code=child.wait();assert code==0,f'{ds} worker exit{code}; preserve logs/outputs'
            state['children']=[];state['jobs'][ds]='sealed';atomic(statepath,state)
            with(RUN/(ds+'-score.log')).open('ab')as log:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('full_score.py')),'--dataset',ds],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
                state['children']=[dict(**identity(child.pid),role='CPU_scorer',dataset=ds)];state['jobs'][ds]='scoring';atomic(statepath,state)
                code=child.wait();assert code==0,f'{ds} scorer exit{code}; predictions remain sealed'
            state['children']=[];state['jobs'][ds]='scored';atomic(statepath,state)
        state['status']='awaiting_publication';atomic(statepath,state)
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();atomic(statepath,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--worker',choices=['hc2','vidstg']);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
    elif a.run:controller(a.resume)
    else:p.error('prepare/run/worker required')
