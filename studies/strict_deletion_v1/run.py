"""Deferred TA full strict-deletion study; no experiment work before both CG fulls finish."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,argparse,hashlib,subprocess,traceback,gc,shutil
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(S),str(R),str(R/'tools'),str(R/'studies/paper_followup_v1')]
from full import read,sha,atomic,identity,check,RUN,PROVIDER,binding
D=R/'outputs/strict_deletion_v1';P=R/'results/strict_deletion_v1'
CG=R/'outputs/cgstvg_full_v1';REVIEW=R/'outputs/paper_review_suite_v1'
COUNTS=dict(hc2=1902,vidstg=10303)
ARMS=['Full','NoSharedCorrection','NoSpatialTrajectory','NoTemporalInference']

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def exited(w):
    try:a=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return a[0]=='Z'or int(a[19])!=w['startticks']
    except FileNotFoundError:return True

def gate():
    """Check receipts/owned exit before importing any model or opening grid caches."""
    for ds in COUNTS:
        names=[CG/f'{ds}_sealed.json',CG/f'{ds}_score_receipt.json',CG/f'{ds}_reporting_receipt.json',
               R/'results/cgstvg_full_v1'/ds/'named_source_comparison/semantics_receipt.json']
        if not all(p.exists()for p in names):return False
        seal,score,report,semantic=map(read,names)
        assert seal['queries']==score['queries']==COUNTS[ds] and score['whole_domain_sealed_before_GT']
        assert score['seal_sha256']==sha(names[0]) and report['source_seal_sha256']==sha(names[0])
        assert report['score_receipt_sha256']==sha(names[1]) and semantic['score_sha256']==sha(names[1])
    for p in [CG/'controller.json',*CG.glob('*_predict_child.json'),*CG.glob('*_score_child.json')]:
        if p.exists()and not exited(read(p)):return False
    return True

def require_gate():
    if not gate():raise RuntimeError('Deferred strict deletion: both CG full domains must finish scoring/reporting and owned exit first')

def rows(ds):
    j=read(RUN/'plan.json')['jobs'][ds];p=PROVIDER/j['manifest'];assert sha(p)==j['manifest_sha256']
    rr=list(map(json.loads,p.read_text().splitlines()));assert len(rr)==COUNTS[ds];return rr

def prepare():
    assert not(D/'plan.json').exists();check();support=read(REVIEW/'supports_sealed.json');shared={ds:{}for ds in COUNTS}
    for ds,stages in support['files'].items():
        for stage,files in stages.items():
            for key,e in files.items():shared[ds][key]=dict(e,origin_stage=stage)
    assert sum(map(len,shared.values()))==384
    atomic(D/'shared_index.json',shared)
    code=[S/'run.py',S/'strict_core.py',S/'reporting.py',*sorted((R/'stvg').glob('*.py')),
          *sorted((R/'tools').glob('full*.py')),R/'studies/paper_followup_v1/common.py']
    inputs=[RUN/'plan.json',RUN/'reuse_index.json',REVIEW/'supports_sealed.json',
            R/'configs/default.json',CG/'plan.json',*[RUN/ds/'sealed.json'for ds in COUNTS]]
    p=dict(schema='strict_deletion_v1',authorization='User explicitly approves the four strict deletion arms, queued AFTER both current CG HC/Vid fulls finish. Reuse compatible caches. No interrupt or ablation predictions/scoring before gate.',
        backbone='TA-STVG primary method; NOT CG-specific HC conservative profile',domains=COUNTS,
        arms=ARMS,definitions=dict(Full='Frozen production gamma/domain k64 beta1 context1.5, shared correction + both-anchor exactDP + posteriorupdate/MBR; every output must match prior full bitwise.',
            NoSharedCorrection='Bypass shared correction E=Source. Rebuild every E-dependent candidate/unary/link, fulltime aux, actual ROI pool, posterior and finalDP.',
            NoSpatialTrajectory='Remove propagated candidates/path score/DP entirely. Use unchanged upstream E as output and ROI trajectory; repool actual pretext grids, recompute time posterior/MBR. NO interpolation/greedy/direct expert-overwrite replacement.',
            NoTemporalInference='Remove posterior update AND MBR; use already-produced native per-offset MAP physical hull and recompute spatialDP for it. NOT old OursS, which retained MBR.'),
        controls=['Source(originalnativeMAPhull)','SourcePlusMBR'],no_grid_search=True,new_expert_queries=0,optimizer_steps=0,
        cache='Read original full Source/E/nativeFP32logits/q0/geometry/eightmaskseeds with SHA. Known384 shared original pretext+postfusion grids reused. Also inspect bound legacy raw grid candidates AFTER gate; use exact-compatible raw inputs only. Missing postfusion requires original Source224 capture; postfusion-only cache needs actual appearance-prefix only. No reuse of pooled Full features for changed trajectories. Full fixed-profile control may reuse matching saved features only with output bitwise proof.',
        timing='No real ablation cache materialization/Source/prefix/teacher/video/GT/prediction before both CG domains seal, score, audit, named source reports and ownedexit. Queue does not holdgpu.lock. CPU-only protocol/hash/gate checks now.',
        exposure='All full sets historically exposed. Fixed configurations; no parameter/sample selection. Same source weights/eight observations/native224/twooffsets/FP16visualFP32readout as original TA full.',
        publication='All four complete arms, signed both signs/severe uniononce/shortlong/parentCI, raw Source versusSource+MBR separated, real worker/cache/prefixcost; never relabel prior replacement controls as deletions.',
        budget=dict(max_full_Source_queries=11821,max_appearance_prefix_queries=11821,combined_missing_grid_queries_max=11821,new_expert_queries=0,max_domain_seconds=259200,max_private_bytes=120*2**30,min_disk_free_bytes=60*2**30),
        known_shared_queries={ds:len(e)for ds,e in shared.items()},shared_index_sha256=sha(D/'shared_index.json'),
        code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f):sha(f)for f in inputs})
    p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
    for f in code:
        target=D/'source_snapshot'/f.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
    atomic(D/'state.json',dict(status='prepared_deferred_until_both_CG_fulls',fingerprint=p['fingerprint']));print(p['fingerprint'])

def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for n,h in p['code_sha256'].items():assert sha(R/n)==h,n
    for n,h in p['input_sha256'].items():assert sha(n)==h,n
    assert sha(D/'shared_index.json')==p['shared_index_sha256'];check();return p

def save_torch(path,value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)

def preflight():
    require_gate();p=validate()
    import torch,numpy as np
    from strict_core import derive,replay
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.representation import extract
    torch.set_num_threads(1);shared=read(D/'shared_index.json');proof=[];started=time.perf_counter()
    for ds in COUNTS:
        seal=read(RUN/ds/'sealed.json')
        for key in list(shared[ds])[:2]:
            e=seal['files'][key];s=shared[ds][key]
            assert sha(RUN/ds/e['capture_path'])==e['capture_sha256']and sha(REVIEW/s['shared_path'])==s['shared_sha256']
            cap=torch.load(RUN/ds/e['capture_path'],map_location='cpu',weights_only=False,mmap=True)
            raw=torch.load(REVIEW/s['shared_path'],map_location='cpu',weights_only=False,mmap=True)
            geo=SpatialGeometry.from_cached_geometry(cap['geometry']);ids=cap['frame_ids'];aux=evidence_dp(geo.for_interval([ids[0],ids[-1]+1]))['final'].numpy()
            assert np.array_equal(extract(raw,aux)[0],cap['content_features'])
            arrays,compact=derive(cap,raw);again=replay(compact)
            with np.load(RUN/ds/e['path'])as z:assert np.array_equal(arrays['Full'],z['Full'])and np.array_equal(arrays['Full_interval'],z['Full_interval'])
            assert all(np.array_equal(arrays[k],again[k])for k in arrays)
            proof.append(dict(dataset=ds,Full_and_pool_bitwise=True,all_compact_replay_exact=True,
                NoSpatial_is_exact_E=np.array_equal(arrays['NoSpatialTrajectory'],cap['E']),
                NoShared_E_exact_Source=torch.equal(compact['uncorrected_geometry']['E'],torch.as_tensor(cap['Source'],dtype=torch.float64)),
                NoTemporal_uses_nativeMAP=True))
    receipt=dict(status='passed_after_both_CG_fulls',queries=4,checks=proof,seconds=time.perf_counter()-started,
        new_Source=0,new_expert=0,new_model=0,GT_read=False,fingerprint=p['fingerprint'])
    atomic(D/'preflight_receipt.json',receipt);atomic(P/'preflight_receipt.json',receipt);print(receipt)

def predict(ds):
    require_gate();p=validate()
    import torch,numpy as np
    from stvg.runtime import setup,lease,load_source
    from stvg.native import frozen_native_forward
    from stvg.evidence import extract_visual_grids
    from stvg.representation import observe_native_grid,extract
    from stvg.visual_prefix import native_appearance_grid
    from stvg.visual_cache import VisualCache
    from stvg.decode import decode
    from stvg.query import QuerySubjectParser
    from stvg.spatial import SpatialGeometry,evidence_dp
    from strict_core import derive,replay
    setup();gpu=lease();who=identity();atomic(D/f'{ds}_worker.json',who)
    fullseal=read(RUN/ds/'sealed.json');shared=read(D/'shared_index.json')[ds];legacy=read(RUN/'reuse_index.json')[ds]
    cfg=read(R/'configs/default.json')['datasets'][ds];runtime=read(RUN/'plan.json')['provider_runtime_fingerprint']
    model=None;parser=None;visual=VisualCache(2*2**30);files={};start=time.perf_counter();model_load_seconds=0.
    counts=dict(queries=0,shared_raw_grid_hits=0,legacy_raw_grid_hits=0,full_Source_queries=0,appearance_prefix_queries=0,new_expert=0)
    try:
        for i,row in enumerate(rows(ds),1):
            assert time.perf_counter()-start<p['budget']['max_domain_seconds'];assert shutil.disk_usage(R).free>p['budget']['min_disk_free_bytes']
            if i%64==1:assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['budget']['max_private_bytes']
            key=row['id'];e=fullseal['files'][key];oldcp=RUN/ds/e['capture_path'];oldpred=RUN/ds/e['path'];oldrec=RUN/ds/e['receipt_path']
            assert sha(oldcp)==e['capture_sha256']and sha(oldpred)==e['sha256']and sha(oldrec)==e['receipt_sha256']
            assert read(oldrec)['input_binding']==binding(row,runtime)
            cap=torch.load(oldcp,map_location='cpu',weights_only=False,mmap=True);assert not cap['GT_read']
            dest=D/ds;out=dest/'predictions'/(key+'.npz');compactpath=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json')
            if rp.exists():
                rec=read(rp);assert rec['fingerprint']==p['fingerprint']and sha(out)==rec['sha256']and sha(compactpath)==rec['compact_sha256']
                files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1;continue
            assert not out.exists()and not compactpath.exists(),'Preserve incomplete unbound outputs, no blind retry'
            tick=time.perf_counter();scratch=D/'active'/ds/key;rawfile=scratch/'raw.pt';rawmeta=scratch/'raw.json';attempt=scratch/'attempt.json'
            raw=None;origin={};source_seconds=0.;mode=None
            if key in shared:
                proof=shared[key];path=REVIEW/proof['shared_path'];assert sha(path)==proof['shared_sha256']
                raw=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                assert raw['parent_capture_sha256']==e['capture_sha256']and raw['frame_ids']==cap['frame_ids']and not raw['GT_read']
                mode='shared_raw';origin=dict(path=str(path),sha256=proof['shared_sha256']);counts['shared_raw_grid_hits']+=1
            elif rawmeta.exists():
                meta=read(rawmeta);assert sha(rawfile)==meta['sha256']and meta['parent_capture_sha256']==e['capture_sha256']
                raw=torch.load(rawfile,map_location='cpu',weights_only=False,mmap=True);mode='paid_raw_recovered';origin=meta
            else:
                assert not attempt.exists(),'Unresolved paid Source attempt: preserve, no blind rerun'
                # Only exact input-bound raw inputs may be reused. Compact candidate
                # features are not a full postfusion grid and cannot substitute.
                prior=legacy.get(key)
                if prior:
                    assert prior['input_binding']==binding(row,runtime)and sha(prior['path'])==prior['sha256']
                    old=torch.load(prior['path'],map_location='cpu',weights_only=False,mmap=True)
                    if 'spatial_inputs'in old and 'records'in old and ('native_logits'in old or 'logits'in old):
                        logs=old.get('native_logits',[z[-1]for z in old.get('logits',[])])
                        assert np.array_equal(old['Source'],cap['Source'])and old['frame_ids']==cap['frame_ids']
                        assert all(torch.equal(a,b)for a,b in zip(logs,cap['native_logits']))and len(logs)==2
                        spatial,valid,ids,shape,_=extract_visual_grids(old['spatial_inputs'],cap['frame_ids'])
                        raw=dict(frame_ids=ids,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        if 'native_local_grid'in old:raw['native_local_grid']=old['native_local_grid'];mode='legacy_both_raw'
                        else:mode='legacy_spatial_need_prefix'
                        origin=dict(path=prior['path'],sha256=prior['sha256'])
                    del old
                if raw is None or 'native_local_grid'not in raw:
                    if model is None:
                        load_tick=time.perf_counter();model=load_source(cfg);model_load_seconds=time.perf_counter()-load_tick
                        atomic(D/f'{ds}_model_load.json',dict(worker=who,seconds=model_load_seconds,loads=1,nested_in_first_query_seconds=True))
                    if parser is None:parser=QuerySubjectParser(R/'checkpoints/stanza')
                    frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids']
                    scratch.mkdir(parents=True,exist_ok=True);atomic(attempt,dict(worker=who,mode=mode or'need_full_Source',before_Source=True,GT_read=False))
                    st=time.perf_counter();torch.manual_seed(20260920);np.random.seed(20260920)
                    if raw is not None:
                        with visual.context(model,frames,ids):grid=native_appearance_grid(model,frames,ids,row['metadata'])
                        raw['native_local_grid']=grid;counts['appearance_prefix_queries']+=1;mode='appearance_prefix'
                    else:
                        model.cuda()
                        with observe_native_grid(model)as grid,visual.context(model,frames,ids):
                            batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses={'subject':parser(row['metadata']['caption'])['subject']}))
                        assert np.array_equal(base.zero['boxes'].cpu().numpy(),cap['Source'])
                        assert all(torch.equal(a.cpu(),b)for a,b in zip(base.zero['logits'],cap['native_logits']))
                        assert [r['frame_ids']for r in recs]==[r['frame_ids']for r in cap['records']]
                        spatial,valid,_,shape,_=extract_visual_grids(base.spatial_inputs,ids)
                        raw=dict(frame_ids=ids,native_local_grid=grid,spatial_features=spatial,spatial_valid=valid,spatial_grid_shape=shape)
                        del batch,base;model.cpu();counts['full_Source_queries']+=1;mode='full_Source'
                    source_seconds=time.perf_counter()-st;raw['GT_read']=False;save_torch(rawfile,raw)
                    origin=dict(sha256=sha(rawfile),parent_capture_sha256=e['capture_sha256'],mode=mode,Source_seconds=source_seconds,worker=who,GT_read=False)
                    atomic(rawmeta,origin);del frames;gc.collect();torch.cuda.empty_cache()
                else:counts['legacy_raw_grid_hits']+=1
            assert raw['frame_ids']==cap['frame_ids']
            # Full-control ROI pool must be exact for every materialized raw grid.
            geo=SpatialGeometry.from_cached_geometry(cap['geometry']);aux=evidence_dp(geo.for_interval([cap['frame_ids'][0],cap['frame_ids'][-1]+1]))['final'].numpy()
            pooled,_=extract(raw,aux);assert np.array_equal(pooled,cap['content_features'])
            arrays,compact=derive(cap,raw)
            with np.load(oldpred)as z:
                assert np.array_equal(arrays['Full'],z['Full'])and np.array_equal(arrays['Full_interval'],z['Full_interval'])
            compact.update(parent_capture_sha256=e['capture_sha256'],raw_origin=origin)
            save_torch(compactpath,compact);cold=torch.load(compactpath,map_location='cpu',weights_only=False,mmap=True)
            again=replay(cold);assert all(np.array_equal(arrays[k],again[k])for k in arrays)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.with_suffix('.npz.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
            os.replace(out.with_suffix('.npz.tmp'),out)
            rec=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(compactpath.relative_to(D)),compact_sha256=sha(compactpath),
                fingerprint=p['fingerprint'],input_binding=binding(row,runtime),parent_capture_sha256=e['capture_sha256'],raw_origin=origin,mode=mode,
                full_pool_and_outputs_bitwise=True,all_compact_replay_exact=True,worker=who,Source_seconds=source_seconds,
                query_seconds=time.perf_counter()-tick,new_expert=0,GT_read=False)
            atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1
            if rawfile.exists():
                assert sha(rawfile)==read(rawmeta)['sha256'];rawfile.unlink();atomic(scratch/'disposed.json',dict(raw_sha256=read(rawmeta)['sha256'],compact_sha256=rec['compact_sha256'],cold_replay_exact=True))
            atomic(D/f'{ds}_progress.json',dict(completed=i,total=COUNTS[ds],counts=counts,seconds=time.perf_counter()-start,worker=who,GT_read=False));print(ds,i,mode,flush=True)
            del cap,raw,arrays,compact,cold,again,geo,pooled;gc.collect()
        assert len(files)==COUNTS[ds]
        assert sum(e['mode']=='shared_raw'for e in files.values())==len(shared)
        assert counts['full_Source_queries']+counts['appearance_prefix_queries']<=COUNTS[ds]-len(shared)
        atomic(D/f'{ds}_sealed.json',dict(queries=len(files),files=files,counts=counts,worker=who,seconds=time.perf_counter()-start,model_load_seconds=model_load_seconds,GT_read=False,fingerprint=p['fingerprint']))
    finally:gpu.close()

def score(ds):
    require_gate();p=validate();os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np
    from full_score import metric
    from common import labels,report
    torch.set_num_threads(1);seal=read(D/f'{ds}_sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read']
    truth=labels(ds);rr=[];checks=0;maxerr=0.;names=['Source','SourcePlusMBR',*ARMS]
    for row in rows(ds):
        e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['compact_path'])==e['compact_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
        gt=truth[row['id']];boxes={int(k):v for k,v in gt['boxes'].items()};vals={}
        with np.load(D/e['path'])as z:
            r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for a in names:
                vals[a],err=metric(z[a],r,boxes,gt['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;maxerr=max(maxerr,err);checks+=1
        rr.append(dict(id=row['id'],parent=str(gt['parent']),duration_group='short_le5s'if(gt['interval'][1]-gt['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
    res=report(rr,names,[('Full',a)for a in names if a!='Full']+[('SourcePlusMBR','Source')],P/ds)
    atomic(D/ds/'metrics.json',dict(private_rows=rr,report=res));receipt=dict(status='scored',queries=len(rr),checks=checks,max_error=maxerr,all_domain_sealed_before_GT=True,seal_sha256=sha(D/f'{ds}_sealed.json'))
    atomic(D/f'{ds}_score_receipt.json',receipt);atomic(P/ds/'score_receipt.json',receipt)

def queue():
    p=validate()
    if(D/'controller.json').exists():assert exited(read(D/'controller.json')),'Do not duplicate queued controller'
    who=identity();atomic(D/'controller.json',who);atomic(D/'state.json',dict(status='waiting_both_CG_fulls_and_reports',controller=who,fingerprint=p['fingerprint'],ablation_predictions=0,new_Source=0,new_expert=0))
    start=time.time()
    while not gate():
        assert time.time()-start<7*86400,'Dependency wait exceeded7days; preserve and inspect'
        state=read(CG/'state.json')
        if state['status'].startswith('failed'):
            atomic(D/'state.json',dict(status='dependency_failed_preserved',dependency_state=state,ablation_predictions=0));raise RuntimeError('Current CG full failed; strict deletion remains unstarted')
        time.sleep(30)
    atomic(D/'dependency_receipt.json',dict(status='both_CG_fulls_complete_before_any_ablation',time=time.time(),
        receipts={str(CG/f'{ds}_{suffix}.json'):sha(CG/f'{ds}_{suffix}.json')for ds in COUNTS for suffix in ['sealed','score_receipt','reporting_receipt']}))
    try:
        if not(D/'preflight_receipt.json').exists():
            with(D/'preflight.log').open('a')as log:
                c=subprocess.Popen([sys.executable,'-B',__file__,'preflight'],stdout=log,stderr=subprocess.STDOUT)
                atomic(D/'preflight_child.json',identity(c.pid));assert c.wait()==0,'Preserve preflight failure; never skip compatibility checks'
        for ds in COUNTS:
            for command in ['predict','score','report']:
                suffix={'predict':'sealed','score':'score_receipt','report':'reporting_receipt'}[command]
                if(D/f'{ds}_{suffix}.json').exists():continue
                atomic(D/'state.json',dict(status=ds+'_'+command,controller=who,fingerprint=p['fingerprint']))
                with(D/f'{ds}_{command}.log').open('a')as log:
                    argv=[sys.executable,'-B',str(S/'reporting.py'),'--dataset',ds]if command=='report'else[sys.executable,'-B',__file__,command,'--dataset',ds]
                    c=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);atomic(D/f'{ds}_{command}_child.json',identity(c.pid));code=c.wait()
                assert code==0,'Preserve source/log/paid raw evidence; no automatic retry'
        atomic(D/'state.json',dict(status='complete_scored_reported_pending_publication',fingerprint=p['fingerprint']))
    except BaseException as e:
        atomic(D/'state.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint']));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','preflight','predict','score','gate']);a.add_argument('--dataset',choices=list(COUNTS));z=a.parse_args()
    {'prepare':prepare,'queue':queue,'preflight':preflight,'predict':lambda:predict(z.dataset),'score':lambda:score(z.dataset),'gate':require_gate}[z.command]()
