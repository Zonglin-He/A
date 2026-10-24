"""Complete HC1902/Vid10303 CG evaluation, immutable profiles and domain seals."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
import sys, json, time, hashlib, argparse, subprocess, traceback, shutil, gc
from pathlib import Path
S = Path(__file__).resolve().parent; R = S.parents[1]
sys.path[:0] = [str(S), str(R), str(R/'tools'), str(R/'studies/paper_followup_v1'),
               str(R/'studies/paper_review_suite_v1')]
from full import read, sha, atomic, identity, check, RUN, PROVIDER
from core import PROFILES, ARMS
D = R/'outputs/cgstvg_full_v1'; P = R/'results/cgstvg_full_v1'
OLD = R/'outputs/paper_review_suite_v1'; FOLLOW = R/'outputs/paper_followup_v1'
TOTAL = dict(hc2=1902, vidstg=10303)

def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
def row_binding(row):
    return digest(dict(id=row['id'], dataset=row['dataset'],
        metadata={k:v for k,v in row['metadata'].items() if k!='frame_ids'}))
def rows(ds):
    job=read(RUN/'plan.json')['jobs'][ds]; path=PROVIDER/job['manifest']
    assert sha(path)==job['manifest_sha256']
    data=list(map(json.loads,path.read_text().splitlines()))
    assert len(data)==TOTAL[ds] and len({r['id'] for r in data})==len(data)
    return data

def prepare():
    assert not (D/'plan.json').exists(); check(True)
    from cgstvg_interfaces_v9 import frame_grid
    from cgstvg_runtime_v9 import verify_vendor
    verify_vendor(); index={ds:{} for ds in TOTAL}; bindings={}
    oldsel=read(OLD/'selection.json'); newsel=read(FOLLOW/'cg_selection.json')
    for root, tag, selection, sealname in [
        (OLD,'old_screen',{ds:g['screen'] for ds,g in oldsel.items()},'cgstvg_v9_screen_sealed.json'),
        (OLD,'old_confirm',{ds:g['confirm'] for ds,g in oldsel.items()},'cgstvg_v9_confirm_sealed.json'),
        (FOLLOW,'followup_confirm',{ds:g['confirm'] for ds,g in newsel.items()},'cg_confirm_sealed.json')]:
        sp=root/sealname; bindings[str(sp)]=sha(sp); seal=read(sp)
        for ds,rr in selection.items():
            current={r['id']:r for r in rows(ds)}
            for row in rr:
                key=row['id']; assert key not in index[ds] and row_binding(row)==row_binding(current[key])
                e=seal['files'][ds][key]
                index[ds][key]=dict(root=str(root),origin=tag,entry=e,row_binding=row_binding(row))
    assert {ds:len(v) for ds,v in index.items()}==dict(hc2=227,vidstg=320)
    native={ds:dict(queries=len(rr:=rows(ds)),parents=len({r['metadata']['source'] for r in rr}),
        min_frames=min(len(frame_grid(r['metadata'],ds)) for r in rr),
        max_frames=max(len(frame_grid(r['metadata'],ds)) for r in rr)) for ds in TOTAL}
    code=[S/'run.py',S/'core.py',S/'audit.py',R/'studies/paper_followup_v1/common.py',
        R/'studies/paper_followup_v1/cg_calibrate.py',R/'studies/paper_review_suite_v1/cgstvg_runtime_v9.py',
        R/'studies/paper_review_suite_v1/cgstvg_interfaces_v9.py',
        R/'studies/paper_review_suite_v1/cgstvg_memory_stream_v8.py',R/'studies/paper_review_suite_v1/cgstvg_score.py',
        *sorted((R/'stvg').glob('*.py')),*sorted((R/'tools').glob('full*.py'))]
    bindingpaths=[R/'outputs/paper_hc_refine_v2/plan.json',R/'results/paper_hc_refine_v2/selection.json',
        FOLLOW/'cg_confirm_binding.json',FOLLOW/'cg_calibrations.json',R/'configs/default.json',RUN/'plan.json']
    bindings.update({str(p):sha(p) for p in bindingpaths})
    atomic(D/'reuse_index.json',index)
    p=dict(schema='cgstvg_full_v1',authorization='Direct user requests full HC1902 with latest HC profile and full Vid10303; explicitly chooses prior Vid-specific profile.',
        profiles=PROFILES,domains=TOTAL,native_input=native,arms=ARMS,
        Source='Same CG-STVG native boxes/native last-layer physical MBR; SourceNative original MAP is a separate readout control. Never compare against TA Source as if it were CG.',
        inference='Original CG420 FP32 cross-source EMA/single-grid/native boundaries; existing audited stream64 attention storage schedule, no video/expert context chunking. SAME ReferDINO8 from nativeMAP positions. No optimizer/new model/sweep.',
        freeze='All domain profiles fixed before any full labels; no selection or retuning from these full outcomes.',
        exposure='Both official sets historically TA-exposed. HC all227parents already used in CG development/CV; one selected query/parent. Vid64CG development queries selected its profile. Full evaluation is not fresh unseen confirmation. Preserve development-query and remaining-query strata without claiming parent independence.',
        reuse='547 prior CG query captures/evidence only with exact manifest/native frame grid/input bindings and SHA; recompute selected-profile pooling/geometry/DP/posterior. No TA grids/logits/teacher evidence masquerading as CG.',
        reused_queries={ds:len(v) for ds,v in index.items()},reuse_index_sha256=sha(D/'reuse_index.json'),
        code_sha256={str(f.relative_to(R)):sha(f) for f in code},bindings=bindings,
        teacher_runtime_fingerprint=read(OLD/'budget_plan.json')['teacher_runtime_fingerprint'],
        scoring='Whole domain seal before independent CPU labels; HC scorer may run alongside Vid GPU. All signed t/v/s, severe t/v/union once (>5pp loss), rescues, short/long, parentclusterCI, queryequal means and real cost segments. Pointwise CIs; no false Source+.9 qualification.',
        storage='Within THIS new output root only: save each paid raw Source before teacher, retain raw scratch while incomplete. Commit prediction + lossless compact geometry/aux/pooled FP64 features/q0/logits + teacher cache + checksums/receipt, replay all arrays exactly, then dispose only that query raw scratch with original SHA and disposal receipt. Old raw caches never deleted. Full raw grids not retained for arbitrary future repooling; compact replays this frozen profile exactly.',
        budget=dict(max_new_Source_queries=11658,extra_native_reference_forwards=4,max_new_same_ReferDINO8_queries=11658,
            max_stage_seconds=259200,max_private_bytes=160*2**30,minimum_disk_free_bytes=80*2**30),
        prior_results='All old positives/negatives immutable; production TA/Fig1/third benchmark untouched')
    p['fingerprint']=digest(p); atomic(D/'plan.json',p)
    public={k:v for k,v in p.items() if k not in ['bindings']};atomic(P/'protocol.json',public)
    for path in code:
        dest=D/'source_snapshot'/path.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    atomic(D/'state.json',dict(status='prepared',fingerprint=p['fingerprint']))
    print(p['fingerprint'],native,flush=True)

def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for n,h in p['code_sha256'].items():assert sha(R/n)==h,n
    for n,h in p['bindings'].items():assert sha(n)==h,n
    assert sha(D/'reuse_index.json')==p['reuse_index_sha256'];check();return p

def torch_save(path, value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)

def load_legacy(e, row, ds):
    import torch
    from cgstvg_interfaces_v9 import frame_grid
    root=Path(e['root']);rec=e['entry'];assert row_binding(row)==e['row_binding']
    cp=root/rec['capture_path']; assert sha(cp)==rec['capture_sha256']
    assert sha(root/rec['path'])==rec['sha256'] and sha(root/rec['receipt_path'])==rec['receipt_sha256']
    cap=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
    assert cap['frame_ids']==frame_grid(row['metadata'],ds) and not cap['GT_read']
    if root==OLD:
        key=cap['teacher_evidence_key'];tp=OLD/'cgstvg_teacher_cache'/(key+'.pt')
    else:
        ap=(root/rec['path']).with_suffix('.acquired.pt');assert sha(ap)==rec['acquired_sha256']
        aq=torch.load(ap,map_location='cpu',weights_only=False,mmap=True);key=aq['teacher']['cache_key'];tp=FOLLOW/'cg_teacher_cache'/(key+'.pt')
    meta=read(tp.with_suffix('.json'));assert sha(tp)==meta['sha256']
    assert meta['key']==key and meta['context']['dataset']==ds
    assert meta['context']['frame_ids']==cap['frame_ids'] and meta['context']['positions']==cap['positions8']
    assert meta['context']['caption']==row['metadata']['caption']
    native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);native['positions']=cap['positions8']
    return cap,native,dict(capture_path=str(cp),capture_sha256=rec['capture_sha256'],
        teacher_path=str(tp),teacher_sha256=meta['sha256'],teacher_key=key,
        historical_receipt_path=str(root/rec['receipt_path']),historical_receipt_sha256=rec['receipt_sha256'],
        historical_origin=e['origin'],historical_cost=rec,acquisition_reused=True)

def committed(ds,key):
    rp=D/ds/'receipts'/(key+'.json')
    if not rp.exists():return None
    rec=read(rp);assert rec['fingerprint']==read(D/'plan.json')['fingerprint']
    for f,h in [('prediction_path','prediction_sha256'),('compact_path','compact_sha256')]:assert sha(D/rec[f])==rec[h]
    assert sha(rec['acquisition']['teacher_path'])==rec['acquisition']['teacher_sha256']
    return dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))

def predict(ds):
    import torch,numpy as np
    from stvg.runtime import setup,lease
    from stvg.decode import decode
    from stvg.positions import observation_positions
    from stvg.teacher_bridge import ResidentTeacher,evidence_key
    from cgstvg_runtime_v9 import load_model,capture
    from cgstvg_interfaces_v9 import frame_grid
    from core import feedback,replay
    p=validate();assert read(D/'preflight_receipt.json')['status']=='passed'
    setup();gpu=lease();teacher=ResidentTeacher(cache_dir=D/'teacher_cache',fingerprint=p['teacher_runtime_fingerprint'])
    started=time.perf_counter();who=identity();atomic(D/f'{ds}_worker.json',who)
    prior_bridges=sum(read(f)['Source_forwards']-1 for f in (D/'active'/ds).glob('*/Source.json'))
    assert 0<=prior_bridges<=2
    files={};reuse=read(D/'reuse_index.json')[ds];model=None;load=None
    counts=dict(queries=0,reused_legacy=0,resumed_committed=0,new_Source_queries=0,new_Source_forwards=0,new_expert=0,native_bridges=0,recovered_paid_Source=0)
    try:
        for i,row in enumerate(rows(ds)):
            assert time.perf_counter()-started<p['budget']['max_stage_seconds']
            assert shutil.disk_usage(R).free>p['budget']['minimum_disk_free_bytes'],'Preserve paid acquisitions; insufficient free disk'
            if i%64==0:
                assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['budget']['max_private_bytes'],'Private storage limit; preserve state'
            key=row['id'];old=committed(ds,key)
            if old:
                assert old['input_binding']==row_binding(row);files[key]=old;counts['queries']+=1;counts['resumed_committed']+=1;continue
            tick=time.perf_counter();scratch=D/'active'/ds/key;scratch.mkdir(parents=True,exist_ok=True)
            sp=scratch/'Source.pt';sj=scratch/'Source.json';attempt=scratch/'attempt.json'
            if key in reuse:
                cap,native,origin=load_legacy(reuse[key],row,ds);counts['reused_legacy']+=1
            else:
                assert not attempt.exists() or sj.exists(),'Unresolved prior Source attempt: preserve failure; no blind rerun'
                if sj.exists():
                    acquisition=read(sj);assert sha(sp)==acquisition['sha256'] and acquisition['input_binding']==row_binding(row)
                    cap=torch.load(sp,map_location='cpu',weights_only=False,mmap=True);counts['recovered_paid_Source']+=1
                else:
                    if model is None:
                        startload=time.perf_counter();model,load=load_model(ds);load['model_load_seconds']=time.perf_counter()-startload
                        atomic(D/f'{ds}_load_{who["pid"]}.json',dict(load,worker=who))
                    metadata=dict(row['metadata']);metadata['frame_ids']=frame_grid(metadata,ds)
                    frames,ids=decode(metadata,ds);assert ids==metadata['frame_ids']
                    atomic(attempt,dict(status='before_Source',input_binding=row_binding(row),worker=who,time=time.time(),GT_read=False))
                    cap=capture(model,frames,ids,metadata['caption'],noop=prior_bridges+counts['native_bridges']<2)
                    pos=observation_positions(ids,[ids.index(cap['native_map'][0]),ids.index(cap['native_map'][1]-1)])
                    cap['positions8']=pos;torch_save(sp,cap)
                    atomic(sj,dict(sha256=sha(sp),input_binding=row_binding(row),worker=who,
                        Source_forwards=cap['source_forwards'],Source_seconds=cap['source_seconds'],load=load,GT_read=False))
                    counts['new_Source_queries']+=1;counts['new_Source_forwards']+=cap['source_forwards'];counts['native_bridges']+=int(cap['native_noop'])
                if 'frames'not in locals():
                    metadata=dict(row['metadata']);metadata['frame_ids']=cap['frame_ids'];frames,ids=decode(metadata,ds)
                pos=cap['positions8'];ids=cap['frame_ids'];native=teacher(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,dataset=ds)
                native['positions']=pos;counts['new_expert']+=native['query_forwards'];tp=D/'teacher_cache'/(native['cache_key']+'.pt');tm=read(tp.with_suffix('.json'));assert sha(tp)==tm['sha256']
                atomic(scratch/'teacher.json',dict(key=native['cache_key'],sha256=tm['sha256'],path=str(tp),worker=who,new_expert=native['query_forwards']))
                acquisition=read(sj);origin=dict(capture_path=str(sp),capture_sha256=acquisition['sha256'],
                    teacher_path=str(tp),teacher_sha256=tm['sha256'],teacher_key=native['cache_key'],
                    acquisition_reused=False,Source=acquisition,expert_new=native['query_forwards'],
                    expert_cache_hit=native['cache_hit'],expert_original_cost=native.get('evidence_original_cost',native['cost']),
                    expert_current_cost=native['cost'],expert_load_seconds=native['model_load_seconds'])
            feedback_start=time.perf_counter();arrays,compact=feedback(cap,native,ds,p['profiles'][ds]);feedback_seconds=time.perf_counter()-feedback_start
            assert np.array_equal(arrays['Source'],cap['Source']) and arrays['Source_interval'].tolist()==cap['native_mbr']
            compact['input_binding']=row_binding(row);compact['acquisition']=origin
            out=D/ds/'predictions'/(key+'.npz');cp=D/ds/'compact'/(key+'.pt');out.parent.mkdir(parents=True,exist_ok=True)
            assert not out.exists() and not cp.exists(),'Uncommitted outputs: preserve and inspect before recovery'
            torch_save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
            again=replay(cold,p['profiles'][ds]);assert all(np.array_equal(arrays[k],again[k]) for k in arrays)
            with out.with_suffix('.npz.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
            os.replace(out.with_suffix('.npz.tmp'),out)
            with np.load(out)as z:assert all(np.array_equal(arrays[k],z[k]) for k in arrays)
            rec=dict(fingerprint=p['fingerprint'],input_binding=row_binding(row),profile=p['profiles'][ds],worker=who,
                prediction_path=str(out.relative_to(D)),prediction_sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),
                acquisition=origin,full_raw_capture_retained=bool(key in reuse),raw_new_scratch_retention='Only until committed exact sufficient-state replay',
                feedback_seconds=feedback_seconds,pipeline_seconds=time.perf_counter()-tick,GT_read=False,all_arrays_cold_replay_exact=True)
            rp=D/ds/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1
            # Only THIS study's completed scratch is disposed. Never touch historical raw caches.
            if sp.exists():
                assert sp.resolve().is_relative_to((D/'active').resolve()) and sha(sp)==origin['capture_sha256']
                sp.unlink();atomic(scratch/'disposed_after_commit.json',dict(raw_sha256=origin['capture_sha256'],
                    compact_sha256=rec['compact_sha256'],receipt_sha256=sha(rp),exact_replay_passed=True))
            atomic(D/f'{ds}_progress.json',dict(completed=i+1,total=TOTAL[ds],counts=counts,seconds=time.perf_counter()-started,worker=who,GT_read=False))
            print(ds,i+1,TOTAL[ds],counts,flush=True)
            del cap,native,arrays,compact,cold,again
            if 'frames'in locals():del frames
            gc.collect();torch.cuda.empty_cache()
        assert len(files)==TOTAL[ds]
        assert sum(e['acquisition']['acquisition_reused'] for e in files.values())==p['reused_queries'][ds]
        assert sum(e['acquisition']['Source']['Source_forwards']-1 for e in files.values()if not e['acquisition']['acquisition_reused'])==2
        atomic(D/f'{ds}_sealed.json',dict(dataset=ds,queries=len(files),files=files,fingerprint=p['fingerprint'],
            counts=counts,GT_read=False,worker=who,worker_seconds=time.perf_counter()-started))
    finally:
        teacher.close();gpu.close()

def score(ds):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np,csv
    from full_score import metric
    from cgstvg_score import native_official_metric
    from common import labels,report
    p=validate();torch.set_num_threads(1);seal=read(D/f'{ds}_sealed.json')
    assert seal['queries']==TOTAL[ds] and not seal['GT_read'] and seal['fingerprint']==p['fingerprint']
    truth=labels(ds);rr=[];checks=0;maxerr=0.;reuse=read(D/'reuse_index.json')[ds]
    for row in rows(ds):
        e=committed(ds,row['id']);assert e==seal['files'][row['id']]
        g=truth[row['id']];gt={int(k):v for k,v in g['boxes'].items()};vals={}
        with np.load(D/e['prediction_path'])as z:
            r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for a in ARMS:
                iv=z[a+'_interval'].tolist();vals[a],err=metric(z[a],r,gt,g['interval'],iv,z['positions8'])
                ref=native_official_metric(z[a],z['frame_ids'].tolist(),r['metadata'],gt,g['interval'],iv,ds)
                err=max(err,max(abs(ref[k]-vals[a][k]) for k in ['t','v','s']));assert err<2e-10;maxerr=max(maxerr,err);checks+=1
        origin=reuse.get(row['id'],{}).get('origin','not_previously_CG_evaluated_query')
        rr.append(dict(id=row['id'],parent=str(g['parent']),origin=origin,
            selection_query=(ds=='hc2'and row['id']in reuse)or(ds=='vidstg'and origin=='old_screen'),
            duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
    pairs=[('Full','Source'),('Full','Space'),('Source','SourceNative')]
    result=report(rr,ARMS,pairs,P/ds);atomic(D/ds/'metrics.json',dict(private_rows=rr,report=result))
    for tag,subset in [('selection_queries',[r for r in rr if r['selection_query']]),('remaining_queries',[r for r in rr if not r['selection_query']])]:
        report(subset,ARMS,pairs,P/ds/tag)
    receipt=dict(status='scored',dataset=ds,queries=len(rr),parents=len({r['parent']for r in rr}),
        checks=checks,max_error=maxerr,seal_sha256=sha(D/f'{ds}_sealed.json'),whole_domain_sealed_before_GT=True,fingerprint=p['fingerprint'])
    atomic(D/f'{ds}_score_receipt.json',receipt);atomic(P/ds/'score_receipt.json',receipt)

def exited(w):
    try:v=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return v[0]=='Z'or int(v[19])!=w['startticks']
    except FileNotFoundError:return True

def run(resume=False):
    p=validate();assert read(D/'preflight_receipt.json')['status']=='passed'
    for path in [D/'controller.json',*D.glob('*_predict_child.json'),*D.glob('*_score_child.json')]:
        if path.exists():assert exited(read(path)),f'Owned process still active: {path}'
    if not resume:assert not(D/'controller.json').exists(),'Use verified same-plan resume only'
    who=identity();atomic(D/'controller.json',who);scorers=[]
    try:
        for ds in TOTAL:
            if not(D/f'{ds}_sealed.json').exists():
                atomic(D/'state.json',dict(status=ds+'_predict',fingerprint=p['fingerprint'],controller=who))
                with(D/f'{ds}_predict.log').open('a')as log:
                    c=subprocess.Popen([sys.executable,'-B',__file__,'predict','--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                    atomic(D/f'{ds}_predict_child.json',identity(c.pid));code=c.wait()
                atomic(D/f'{ds}_predict_exit.json',dict(code=code));assert code==0,'Preserve acquired evidence and failure; never blind retry'
            if not(D/f'{ds}_score_receipt.json').exists():
                log=(D/f'{ds}_score.log').open('a');c=subprocess.Popen([sys.executable,'-B',__file__,'score','--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/f'{ds}_score_child.json',identity(c.pid));scorers.append((ds,c,log))
        atomic(D/'state.json',dict(status='CPU_scoring',controller=who,fingerprint=p['fingerprint']))
        for ds,c,log in scorers:
            code=c.wait();log.close();atomic(D/f'{ds}_score_exit.json',dict(code=code));assert code==0,'Preserve scoring failure without inference rerun'
        atomic(D/'state.json',dict(status='complete_scored_pending_reporting_publication',fingerprint=p['fingerprint'],controller=who))
    except BaseException as e:
        atomic(D/'state.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint'],controller=who));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','predict','score','run']);a.add_argument('--dataset',choices=list(TOTAL));a.add_argument('--resume',action='store_true');z=a.parse_args()
    {'prepare':prepare,'predict':lambda:predict(z.dataset),'score':lambda:score(z.dataset),'run':lambda:run(z.resume)}[z.command]()
