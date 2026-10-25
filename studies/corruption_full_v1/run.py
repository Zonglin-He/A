"""Finite complete-target current-kappa0 corruption, per-condition seals."""
from common import *
import argparse, traceback, copy

def prepare():
    assert not (D/'plan.json').exists()
    code = [*S.glob('*.py'), *S.joinpath('reference').glob('*'), R/'stvg/current.py',
            R/'stvg/predictor.py', R/'stvg/teacher_bridge.py', R/'stvg/decode.py', R/'stvg/runtime.py',
            R/'studies/map_paper_completion_v1/corruptions.py', R/'studies/no_shrink_main_v1/core.py',
            R/'studies/no_shrink_main_v1/ns_common.py', R/'tools/full_score.py',
            R/'studies/paper_followup_v1/common.py']
    inputs = [R/'CURRENT_METHOD.json', R/'profiles/no_shrink_map_v1.json', RUN/'plan.json',
              ns.D/'plan.json', R/'outputs/tta_full_v1/plan.json',
              R/'outputs/tta_full_v1/execution_v2_plan.json', ns.OLD/'shifts_plan.json']
    for ds in COUNTS:
        inputs += [RUN/ds/'sealed.json', ns.D/f'corruption_{ds}_sealed.json',
                   ns.D/f'corruption_{ds}_reporting_receipt.json',
                   ns.PREV/f'temporal_v2_{ds}_sealed.json', ns.PREV/f'temporal_v2_{ds}_reporting_receipt.json']
    p = dict(schema='current_kappa0_full_physical_corruptions_v1',
        authorization='Direct user requests complete targets and DECOTA-like realistic corruption families/severity. Default five archived deployment bursts at2.5/5/10percent plus all three previous fixed corruption controls.',
        datasets=COUNTS, conditions=conditions(), views=sum(COUNTS.values())*len(conditions()),
        arms=['Source', 'FullMAP'], main='TA current kappa0/beta1/eight observations/q1jointMAP/newinterval exactDP; all other settings unchanged.',
        severity='Burst severity is physical affected-frame coverage, not equal magnitude across families. Original kernel/occluder/exposure magnitude fixed; ceil coverage and no-sampled-hit retained. No GT/query-dependent placement or forced event/observation hit.',
        roster='Existing maintained HC1902/227parents and Vid10303/732parents; complete current target manifests, not old DECOTA HC3482 or one-query/video panels. Historical full exposure disclosed.',
        readout='Native Source per-offset legalMAP+physicalhull; no new MBR comparison. Only same-corrupted-input Source versus currentFull plus matched clean scalar references.',
        reuse='Exact prior768views/128query panels after RGB/inputbinding/parent/tripleSHA and current kappa0 replay. Exact same-input no-hit clean NoShrink reuse after pixel identity proof; new Source and expert only for genuinely new input evidence. Old acquisition cost nonzero.',
        upper_bound_new_Source=sum(COUNTS.values())*len(conditions())-768,
        upper_bound_new_expert=sum(COUNTS.values())*len(conditions())-768,
        raw_checkpoint='Paid Source.pt+SHA before teacher. New raw removed only after committed prediction/receipt/compact, all-array cold replay/readback. Attempt without validSource checkpoint forbids blind rerun.',
        storage='At most one condition/domain resident. After entireunit seal+officialscore+all-array/teacherSHA audit+signed report, remove ONLY this new unit geometry/pooledfeature compacts and ownedteacher payloads, recording every originalSHA/disposal. Retain full predictions, costs, metrics, input/teacher/Source hashes and exact-replay receipt. This does NOT preserve arbitrary future geometry/repool/optimizer replay. Old roots/cache never removed. Failed/uncommitted raw preserved.',
        schedule='Wait existing tta_full_v1 controller1924421/start245882503 fully exits successfully and all its TTA/current recalculation audits exist; do not signal or alter it or currentSource. One GPUworker, sequential full condition/domain predict→independentCPUscore→audit/root scalar export→own temporary disposal.',
        evaluation='Full1902or10303 unit seal before that unitGT. All signed positive/negative/severe>5pp uniononce/rescues, short<=5s/long parentpointwiseCI, clean-corrupt and sampled-hit strata, requested/actualcoverage, perPIDstart costs. No condition/severity/method selection from labels.',
        limits=dict(GPUworkers=1, CPUjobs=1, CPUaffinity=[14,15], CPUnice=19, CPUthreads=1,
                    Source_numeric_threads=4, minimum_RAM=8*2**30, minimum_disk=80*2**30,
                    maximum_private=160*2**30, unit_seconds=259200),
        failure='Preserve code/plan/log/allpaidraw/attempts/teacherSHA/predictions. No autoretry/skip/fallback/hashwaiver. Independent versioned compatible engineering fix if necessary.',
        code_sha256={str(f.relative_to(R)):sha(f) for f in code},
        input_sha256={str(f):sha(f) for f in inputs})
    p['fingerprint'] = digest(p)
    atomic(D/'plan.json', p)
    atomic(P/'protocol.json', {k:v for k,v in p.items() if k != 'input_sha256'})
    for f in code:
        out = D/'source_snapshot'/f.relative_to(R)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f.read_bytes())
    print(p['fingerprint'], flush=True)

def preflight():
    import numpy as np, torch
    from operators import apply, FAMILIES, burst_spec, corrupt_local
    from stvg.decode import decode
    from stvg.runtime import guard
    validate()
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    checks = []
    # Physical ceil and subinteger percentages; original canonical functions,
    # not a new integer-truncated approximation. All no-hit inputs retained.
    for n in [1, 11, 101, 600]:
        for p in [2.5, 5., 10.]:
            spec = burst_spec('fixed_contract_source', n, list(range(n)), p)
            assert spec['length'] == max(1, __import__('math').ceil(n*p/100))
            assert 0 <= spec['physical_start'] < spec['physical_end'] <= n
            checks.append(dict(frames=n, coverage=p, length=spec['length']))
    synthetic=np.random.default_rng(7).integers(0,256,(8,32,40,3),dtype=np.uint8)
    ids=[0,10,20,30,40,50,60,70]
    for family in ['black','motion_blur','occlusion','overexposure']:
        a=corrupt_local(synthetic,[2,3],family,'fixed_contract_source')
        assert np.array_equal(a,corrupt_local(synthetic,[2,3],family,'fixed_contract_source'))
        assert np.array_equal(a[[0,1,4,5,6,7]],synthetic[[0,1,4,5,6,7]])
    real=[]
    for ds in COUNTS:
        # Two real decoded inputs/domain. No labels, models or CUDA.
        for row in records(ds)[:2]:
            frames, ids=decode(row['metadata'],ds)
            for c in conditions():
                a, meta=apply(frames,ids,row['metadata'],ds,c)
                b, again=apply(frames,ids,row['metadata'],ds,c)
                assert np.array_equal(a,b) and meta==again
                if c['kind']=='burst':
                    untouched=np.ones(len(ids),bool);untouched[meta['positions']]=False
                    assert np.array_equal(a[untouched],frames[untouched])
                real.append(dict(dataset=ds,condition=c['id'],pixels_SHA=hashlib.sha256(a.tobytes()).hexdigest(),
                                 repeat_exact=True,sampled_hit_count=len(meta.get('positions',ids))))
            arr, proof=clean(ds,row)
            assert len(arr['Source'])==len(ids) and np.array_equal(arr['frame_ids'],ids)
            del frames,arr
    assert not torch.cuda.is_initialized()
    r=dict(status='passed',physical_contracts=checks,real_transform_checks=real,
           GT_read=False,new_Source=0,new_expert=0,new_optimizer=0,new_GPU=0,
           first_paid_model_native_checkpoint_coldreplay_still_required=True)
    atomic(D/'preflight_receipt.json',r);atomic(P/'preflight_receipt.json',r)
    print('CPU transforms and real inputs passed; zero model/GT work',flush=True)

class Teacher:
    def __init__(self, root):
        from stvg.teacher_bridge import ResidentTeacher
        self.root=root
        self.fingerprint=fixed(ns.OLD/'shifts_plan.json')['runtime_fingerprint']
        self.live=ResidentTeacher(cache_dir=root/'teacher_cache',fingerprint=self.fingerprint)
        self.last=None
    def __call__(self, frames, caption, **kw):
        import torch
        from stvg.teacher_bridge import evidence_key
        key,context=evidence_key(frames,caption,fingerprint=self.fingerprint,**kw)
        old=ns.OLD/'shift_teacher_cache'
        rp,pp=old/(key+'.json'),old/(key+'.pt')
        if rp.exists():
            rc=read(rp);assert rc['key']==key and rc['context']==context and sha(pp)==rc['sha256']
            z=copy.deepcopy(torch.load(pp,map_location='cpu',weights_only=False))
            cost=copy.deepcopy(z['cost']);z['evidence_original_cost']=cost
            z['cost']={k:0. for k in z['cost']}
            z.update(cache_hit=True,cache_key=key,model_load_seconds=0.,process_wall_seconds=0.,query_forwards=0,peak_GPU_bytes=0)
            self.last=dict(path=str(pp),sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp),owned=False,original_cost=cost)
            return z
        z=self.live(frames,caption,**kw)
        pp=self.root/'teacher_cache'/(key+'.pt');rp=pp.with_suffix('.json')
        self.last=dict(path=str(pp),sha256=sha(pp),receipt=str(rp),receipt_sha256=sha(rp),owned=True,
                       original_cost=z.get('evidence_original_cost',z['cost']))
        return z
    def close(self):
        self.live.close()

def predict(ds,name):
    import torch,numpy as np
    from operators import apply
    from stvg.runtime import setup,lease,load_source
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from stvg.current import finish
    from stvg.readout import prediction
    p=validate();assert read(D/'preflight_receipt.json')['status']=='passed'
    root=unit(ds,name);assert not(root/'worker.json').exists(),'Preserve paid attempts, no blind retry'
    setup();lock=lease();who=identity();atomic(root/'worker.json',who)
    parent=ns.load_old('_full_corrupt_checkpoint_parent',R/'studies/map_paper_completion_v1/corruptions.py',ns.suite())
    Klass=parent.klass();model=None;teacher=None;parser=None;files={}
    start=time.perf_counter();counts=dict(queries=0,new_Source=0,new_expert=0,legacy_queries=0,clean_identity_queries=0)
    try:
        for i,row in enumerate(records(ds),1):
            resources();assert time.perf_counter()-start<259200
            key=row['id'];out=root/'predictions'/(key+'.npz');rp=root/'receipts'/(key+'.json');cp=root/'compact'/(key+'.pt')
            assert not out.exists() and not rp.exists() and not cp.exists()
            tick=time.perf_counter();frames,ids=decode(row['metadata'],ds)
            view,meta=apply(frames,ids,row['metadata'],ds,condition(name))
            rgbsha=hashlib.sha256(view.tobytes()).hexdigest();originalsha=hashlib.sha256(frames.tobytes()).hexdigest()
            old=legacy(ds,key,name);raw=None;teacherproof=None
            if old is not None:
                acq=old['provenance']['original_acquisition']
                assert acq['transformed_RGB_sha256']==rgbsha and acq['original_RGB_sha256']==originalsha
                c=dict(kind='legacy_reused',dataset=ds,query=key,view=condition(name)['name'],entry=old,GT_read=False)
                arrays=replay(c);counts['legacy_queries']+=1
                cost=dict(new_Source=0,new_expert=0,inherited_nonzero_acquisition=acq)
            elif np.array_equal(view,frames):
                arrays,proof=clean(ds,row)
                c=dict(kind='clean_identity',dataset=ds,query=key,provenance=proof,GT_read=False)
                assert rgbsha==originalsha
                counts['clean_identity_queries']+=1
                cost=dict(new_Source=0,new_expert=0,inherited_nonzero_acquisition=proof)
            else:
                if model is None:
                    loadtick=time.perf_counter();model=Klass(ds);model.source=load_source(model.config);model.source.cpu()
                    teacher=Teacher(root);model.teacher_service=teacher
                    parser=QuerySubjectParser(R/'checkpoints/stanza')
                    atomic(root/'model_load.json',dict(worker=who,seconds=time.perf_counter()-loadtick,scope='Source/parser construction; resident expert loads on first actual miss'))
                scratch=root/'active'/key;attempt=scratch/'attempt.json';raw=scratch/'Source.pt';rawmeta=scratch/'Source.json'
                assert not attempt.exists(),'Paid attempt exists; preserve and investigate'
                atomic(attempt,dict(worker=who,before_Source=True,GT_read=False,input_RGB_SHA=rgbsha))
                def checkpoint(z):
                    save(raw,z);atomic(rawmeta,dict(path=str(raw),sha256=sha(raw),worker=who,Source_completed=True,GT_read=False,RGB_sha256=rgbsha));counts['new_Source']+=1
                model.acquisition_checkpoint=checkpoint
                parses=dict(subject=parser(row['metadata']['caption'])['subject'])
                torch.manual_seed(20260920);np.random.seed(20260920)
                cap=model.capture(view,ids,row['metadata'],parses=parses,collect_mask_diagnostics=True)
                result=finish(cap)
                native=prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],ids)['physical_interval']
                c=dict(kind='acquired',Source=np.asarray(cap['Source']),Source_interval=native,frame_ids=ids,
                       positions8=cap['planned_positions'],geometry=cap['geometry'].state_dict(),
                       content_features=result['content_features'],native_prior=cap['posterior']['Native'],GT_read=False)
                arrays=dict(Source=c['Source'],Source_interval=np.asarray(native),FullMAP=result['boxes'],
                            FullMAP_interval=np.asarray(result['interval']),frame_ids=np.asarray(ids),positions8=np.asarray(c['positions8']))
                teacherproof=teacher.last;counts['new_expert']+=cap['cost']['ReferDINO_query_forwards']
                cost=dict(new_Source=1,new_expert=cap['cost']['ReferDINO_query_forwards'],Source=cap['cost'],finish=result['cost'],teacher=teacherproof)
                del cap,result
            save(cp,c);again=replay(torch.load(cp,map_location='cpu',weights_only=False,mmap=True))
            assert all(np.array_equal(arrays[k],again[k]) for k in arrays),'Cold replay mismatch'
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
            os.replace(out.with_suffix('.tmp'),out)
            with np.load(out) as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays)
            e=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),
                   worker=who,input_binding=binding(row,fixed(RUN/'plan.json')['provider_runtime_fingerprint']),
                   original_RGB_sha256=originalsha,transformed_RGB_sha256=rgbsha,transform=meta,cost=cost,
                   teacher=teacherproof,query_seconds=time.perf_counter()-tick,kind=c['kind'],
                   raw_Source_original_sha256=sha(raw)if raw else None,all_arrays_coldreplay_exact=True,GT_read=False)
            atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
            if raw:
                assert sha(raw)==read(rawmeta)['sha256'];raw.unlink()
                atomic(raw.parent/'disposed.json',dict(raw_sha256=e['raw_Source_original_sha256'],committed_receipt_sha256=sha(rp),all_arrays_coldreplay_exact=True))
            counts['queries']=i;atomic(root/'progress.json',dict(counts=counts,total=COUNTS[ds],worker=who,GT_read=False))
            print(ds,name,i,counts,flush=True)
            del frames,view,arrays,c,again;gc.collect();torch.cuda.empty_cache()
        assert len(files)==COUNTS[ds]
        atomic(root/'sealed.json',dict(files=files,queries=len(files),counts=counts,seconds=time.perf_counter()-start,worker=who,fingerprint=p['fingerprint'],GT_read=False))
    except BaseException:
        atomic(root/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),worker=who));raise
    finally:
        if teacher is not None:teacher.close()
        lock.close()

def score(ds,name):
    import torch,numpy as np
    from full_score import metric
    torch.set_num_threads(1);validate();root=unit(ds,name);seal=read(root/'sealed.json')
    assert seal['queries']==COUNTS[ds] and not seal['GT_read']
    assert set(seal['files'])=={r['id']for r in records(ds)}
    truth=labels(ds);cleanrows={r['id']:r for r in read(ns.PREV/'temporal_v2_v2'/ds/'metrics.json')['private_rows']}
    rr=[];checks=0;maxerr=0.
    for i,row in enumerate(records(ds),1):
        key=row['id'];e=seal['files'][key];assert sha(D/e['path'])==e['sha256'];g=truth[key];v={}
        with np.load(D/e['path'])as z:
            assert np.array_equal(z['frame_ids'],row['metadata']['frame_ids'])
            for a in ['Source','FullMAP']:
                v[a],err=metric(z[a],row,{int(k):b for k,b in g['boxes'].items()},g['interval'],z[a+'_interval'].tolist(),z['positions8'])
                maxerr=max(maxerr,err);checks+=1
        cl=cleanrows[key];assert cl['parent']==str(g['parent'])
        v.update(CleanSource=cl['values']['Source'],CleanFullMAP=cl['values']['NoShrink'])
        rr.append(dict(id=key,parent=str(g['parent']),duration_group=cl['duration_group'],values=v,
                       sampled_hit=not e['transform'].get('no_sampled_hit',False)))
        if i%100==0:atomic(root/'score_progress.json',dict(completed=i,total=COUNTS[ds]))
    folder=P/name/ds;pairs=[('FullMAP','Source'),('Source','CleanSource'),('FullMAP','CleanFullMAP')]
    report=ns.suite().parent_report(rr,['Source','FullMAP','CleanSource','CleanFullMAP'],pairs,folder)
    ns.suite().signed_tails(rr,pairs,folder)
    atomic(root/'metrics.json',dict(private_rows=rr,report=report))
    for hit in [False,True]:
        sub=[r for r in rr if r['sampled_hit']==hit]
        if sub:ns.suite().parent_report(sub,['Source','FullMAP','CleanSource','CleanFullMAP'],pairs,folder/('sampled_hit'if hit else'no_sampled_hit'))
    rec=dict(status='scored',checks=checks,max_error=maxerr,seal_sha256=sha(root/'sealed.json'),whole_unit_sealed_before_GT=True)
    atomic(root/'score_receipt.json',rec);atomic(folder/'score_receipt.json',rec)

def audit(ds,name):
    import torch,numpy as np
    torch.set_num_threads(1);validate();root=unit(ds,name);seal=read(root/'sealed.json');sr=read(root/'score_receipt.json')
    assert sr['seal_sha256']==sha(root/'sealed.json');checks=0;segments={}
    for key,e in seal['files'].items():
        for f,h in [('path','sha256'),('receipt_path','receipt_sha256'),('compact_path','compact_sha256')]:assert sha(D/e[f])==e[h];checks+=1
        if e['teacher']:
            t=e['teacher'];assert sha(t['path'])==t['sha256']and sha(t['receipt'])==t['receipt_sha256']
        c=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=replay(c)
        with np.load(D/e['path'])as z:assert all(np.array_equal(a[k],z[k])for k in a)
        w=f"{e['worker']['pid']}:{e['worker']['startticks']}";s=segments.setdefault(w,dict(queries=0,new_Source=0,new_expert=0,query_wall_seconds=0.))
        s['queries']+=1;s['new_Source']+=e['cost']['new_Source'];s['new_expert']+=e['cost']['new_expert'];s['query_wall_seconds']+=e['query_seconds']
    rr=read(root/'metrics.json')['private_rows'];report=read(P/name/ds/'report.json')
    for a,fields in report['means'].items():
        for f,v in fields.items():
            values=[r['values'][a][f]for r in rr if r['values'][a][f]is not None]
            assert (v is None and not values)or abs(v-float(np.mean(values)))<1e-12
    tails={}
    for key in report['paired']:
        a,b=key.split('-');dt=[100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr];dv=[100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]
        tails[key]=dict(severe_t=sum(x< -5 for x in dt),severe_v=sum(x< -5 for x in dv),severe_union_once=sum(x< -5 or y< -5 for x,y in zip(dt,dv)),rescue_union_once=sum(x>5 or y>5 for x,y in zip(dt,dv)))
    folder=P/name/ds;atomic(folder/'all_paired_tails.json',tails);atomic(folder/'cost_segments.json',dict(segments=segments,whole_worker_seconds=seal['seconds'],counts=seal['counts'],scope='Actual acquisition and conditional reused time by PID/start. Nested Source/teacher/finish/query times not summed; cachehits preserve nonzero original acquisitions. Not isolated deployment latency.'))
    rec=dict(status='verified_complete_unit',queries=COUNTS[ds],SHA_checks=checks,all_arrays_coldreplay_exact=True,all_means_recomputed=True,score_sha256=sha(root/'score_receipt.json'),new_GT_scoring=0)
    atomic(root/'reporting_receipt.json',rec);atomic(folder/'reporting_receipt.json',rec)
    atomic(root/'state.json',dict(status='complete_scored_audited_pending_root_review_publication'))

def cleanup(ds,name):
    validate();root=unit(ds,name);audit=read(root/'reporting_receipt.json');assert audit['status']=='verified_complete_unit'
    seal=read(root/'sealed.json');deleted=[]
    for e in seal['files'].values():
        cp=D/e['compact_path'];assert sha(cp)==e['compact_sha256'];cp.unlink();deleted.append(dict(path=str(cp.relative_to(D)),sha256=e['compact_sha256'],reason='Whole unit exact replay audited; immutable predictions retained'))
    cache=root/'teacher_cache'
    if cache.exists():
        for pp in cache.glob('*.pt'):
            rp=pp.with_suffix('.json');h=sha(pp);assert read(rp)['sha256']==h;pp.unlink();deleted.append(dict(path=str(pp.relative_to(D)),sha256=h,reason='Owned teacher temporary payload after whole-unit audited Source/geometry replay; context receipt retained'))
    rec=dict(status='only_new_verified_temporary_payloads_disposed',files=deleted,reporting_receipt_sha256=sha(root/'reporting_receipt.json'),old_roots_unchanged=True,arbitrary_future_geometry_replay_available=False)
    atomic(root/'temporary_disposal_receipt.json',rec)
    atomic(P/name/ds/'storage_receipt.json',dict(status=rec['status'],disposed_files=len(deleted),private_disposal_receipt_sha256=sha(root/'temporary_disposal_receipt.json'),predictions_metrics_all_signed_retained=True,arbitrary_future_geometry_replay_available=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','predict','score','audit','cleanup'])
    parser.add_argument('--dataset',choices=COUNTS);parser.add_argument('--condition',choices=[c['id']for c in conditions()]);a=parser.parse_args()
    globals()[a.command]()if a.command in ['prepare','preflight']else globals()[a.command](a.dataset,a.condition)
