"""Deferred CPU-only complete MAP/DP evaluation on both frozen backbones."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,time,hashlib,argparse,subprocess,traceback,shutil,csv,gc
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path[:0]=[str(S),str(R),str(R/'tools'),str(R/'studies/paper_followup_v1')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
D=R/'outputs/map_full_v1';P=R/'results/map_full_v1';CG=R/'outputs/cgstvg_full_v1';STRICT=R/'outputs/strict_deletion_v1'
COUNTS=dict(hc2=1902,vidstg=10303);BACKBONES=['TA','CG']
ARMS=['Source','SourcePlusMBR','SourceJointMAP','FullMBR','FullMAPFixedBoxes','FullMAP']

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def exited(w):
    try:a=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return a[0]=='Z'or int(a[19])!=w['startticks']
    except FileNotFoundError:return True
def gate():
    for ds in COUNTS:
        names=[CG/f'{ds}_sealed.json',CG/f'{ds}_score_receipt.json',CG/f'{ds}_reporting_receipt.json',R/'results/cgstvg_full_v1'/ds/'named_source_comparison/semantics_receipt.json']
        if not all(p.exists()for p in names):return False
        seal,score,audit,semantic=map(read,names)
        assert seal['queries']==score['queries']==COUNTS[ds]and score['whole_domain_sealed_before_GT']
        assert score['seal_sha256']==audit['source_seal_sha256']==sha(names[0])
        assert audit['score_receipt_sha256']==semantic['score_sha256']==sha(names[1])
    return all(exited(read(p))for p in [CG/'controller.json',*CG.glob('*_predict_child.json'),*CG.glob('*_score_child.json')]if p.exists())
def require_gate():
    if not gate():raise RuntimeError('MAP full deferred until both CG fulls seal/score/report and all CG owned processes exit')
def rows(ds):
    j=read(RUN/'plan.json')['jobs'][ds];path=PROVIDER/j['manifest'];assert sha(path)==j['manifest_sha256']
    rr=list(map(json.loads,path.read_text().splitlines()));assert len(rr)==COUNTS[ds];return rr
def prepare():
    assert not(D/'plan.json').exists();check()
    code=[S/'run.py',S/'map_core.py',R/'studies/cgstvg_full_v1/core.py',R/'studies/strict_deletion_v1/run.py',
          R/'studies/paper_followup_v1/common.py',*sorted((R/'stvg').glob('*.py')),*sorted((R/'tools').glob('full*.py')),
          R/'studies/paper_review_suite_v1/cgstvg_runtime_v9.py',R/'studies/paper_review_suite_v1/cgstvg_interfaces_v9.py',R/'studies/paper_review_suite_v1/cgstvg_memory_stream_v8.py']
    inputs=[RUN/'plan.json',CG/'plan.json',STRICT/'plan.json',*[RUN/ds/'sealed.json'for ds in COUNTS]]
    p=dict(schema='map_full_v1',authorization='Latest direct user requests complete MAP decoding with spatial DP recomputed in the new MAP interval for both TA and CG, cache first, AFTER current Vid full. This cache-only study precedes the previously queued strict deletions without changing their frozen science.',
        backbones=BACKBONES,domains=COUNTS,arms=ARMS,
        change='Keep every upstream input/eight observation, auxiliary full-video trajectory, pooled features, posterior and domain-specific profile fixed. Replace ONLY final q1 MBR with joint physical-interval argmax; run unchanged spatial DP on that MAP interval. No ROI/prototype refit because upstream auxiliary path is unchanged.',
        MAP='numpy.argmax of full legal q1 matrix, C-order earliest index tie break, [frame_ids[start],frame_ids[end]+1). SourceJointMAP is q0 joint MAP with source boxes. Original TA Source remains per-offset MAP then physical hull; joint MAP is not asserted to be the original two-offset readout.',
        control='FullMBR must reproduce saved Full arrays/intervals bitwise on every query. FullMAPFixedBoxes isolates readout on fixed old Full boxes; FullMAP reruns space and is the requested complete output. Same upstream q1 for both.',
        profiles='TA frozen original k64/beta1/context1.5; CG HC gamma0/u0/link0/beta.5/k64/context1.5 and Vid gamma.5/u.5/link.25/beta1/k64/context1 unchanged. No outcome-based reselection.',
        acquisition='All required q0/features/geometry/native readouts are sufficient cached state. Zero new Source/prefix/expert/RGB/optimizer/GPU work. Cached study times are not deployment cost; original acquisition retained as provenance.',
        order='Wait for BOTH CG full domains complete seals/official scores/audit/named-source reports and actual CG owned exit. Never run before gate. Prior waiting strict controller preserved then safely stopped while0predictions/noGPU; original strict entrypoint launches only after all four MAP domain prediction/score/report stages succeed and MAP receipt gate verifies them.',
        evaluation='Each entire backbone/domain prediction set seals before separate CPU official labels. FullMAP vs FullMBR and vs Source/SourceJointMAP, paired fixed-box vs re-DP, all signed positives/negatives, severe t/v union counted once, rescues, shortlong parent-cluster pointwise CI, exact parent Source/MBR metrics alignment, CPU cost per PID/start. Never promise reviewer acceptance.',
        budget=dict(new_Source=0,new_prefix=0,new_expert=0,new_GPU=0,optimizer_steps=0,max_stage_seconds=259200,max_private_bytes=40*2**30,min_disk_free_bytes=60*2**30),
        exposure='All official sets historically exposed. HC CG all227parents and Vid64DEV selected profiles; this is fixed readout comparison, not fresh validation or permission for further tuning.',
        code_sha256={str(f.relative_to(R)):sha(f)for f in code},input_sha256={str(f):sha(f)for f in inputs})
    p['fingerprint']=digest(p);atomic(D/'plan.json',p);atomic(P/'protocol.json',{k:v for k,v in p.items()if k!='input_sha256'})
    for f in code:
        target=D/'source_snapshot'/f.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
    atomic(D/'state.json',dict(status='prepared_deferred',fingerprint=p['fingerprint']));print(p['fingerprint'])
def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for n,h in p['code_sha256'].items():assert sha(R/n)==h,n
    for n,h in p['input_sha256'].items():assert sha(n)==h,n
    check();return p
def parent(backbone,ds,key,seal):
    root=RUN/ds if backbone=='TA'else CG;e=seal['files'][key]
    cp=root/e['capture_path']if backbone=='TA'else root/e['compact_path']
    pp=root/e['path']if backbone=='TA'else root/e['prediction_path'];rp=root/e['receipt_path']
    ch=e['capture_sha256']if backbone=='TA'else e['compact_sha256'];ph=e['sha256']if backbone=='TA'else e['prediction_sha256']
    assert sha(cp)==ch and sha(pp)==ph and sha(rp)==e['receipt_sha256']
    return dict(capture_path=str(cp),capture_sha256=ch,prediction_path=str(pp),prediction_sha256=ph,receipt_path=str(rp),receipt_sha256=e['receipt_sha256'])
def calculate(backbone,ds,origin):
    import torch,numpy as np
    from map_core import derive
    c=torch.load(origin['capture_path'],map_location='cpu',weights_only=False,mmap=True);assert not c['GT_read']
    a=derive(backbone,ds,c)
    with np.load(origin['prediction_path'])as z:
        assert np.array_equal(a['FullMBR'],z['Full'])and np.array_equal(a['FullMBR_interval'],z['Full_interval'])
        assert np.array_equal(a['SourcePlusMBR'],z['Source'])and np.array_equal(a['SourcePlusMBR_interval'],z['Source_interval'])
        if backbone=='CG':assert np.array_equal(a['Source_interval'],z['SourceNative_interval'])
    assert not torch.cuda.is_initialized();return a
def preflight():
    require_gate();p=validate()
    import torch,numpy as np
    torch.set_num_threads(1);proof=[]
    for b in BACKBONES:
        for ds in COUNTS:
            sp=RUN/ds/'sealed.json'if b=='TA'else CG/f'{ds}_sealed.json';seal=read(sp)
            for row in rows(ds)[:2]:
                o=parent(b,ds,row['id'],seal);a=calculate(b,ds,o);again=calculate(b,ds,o)
                assert all(np.array_equal(a[k],again[k])for k in a)
                if b=='TA':
                    from cached import ARMS as ca
                    cp=R/'outputs/paper_review_suite_v1/cached'/ds/(row['id']+'.npz');cr=read(cp.with_suffix('.json'))
                    assert sha(cp)==cr['sha256']
                    with np.load(cp)as z:assert np.array_equal(a['FullMAP_interval'],z['intervals'][ca.index('P1MAP')])
                proof.append(dict(backbone=b,dataset=ds,FullMBR_parent_exact=True,MAP_DPinference_replay_exact=True,TA_MAP_matches_prior_interval=b=='TA'))
    atomic(D/'preflight_receipt.json',dict(status='passed',queries=8,proof=proof,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint']))
def predict(b,ds):
    require_gate();p=validate()
    import torch,numpy as np
    from stvg.runtime import guard
    sys.addaudithook(guard);torch.set_num_threads(1);who=identity();dest=D/b/ds;files={};started=time.perf_counter()
    sp=RUN/ds/'sealed.json'if b=='TA'else CG/f'{ds}_sealed.json';seal=read(sp);assert seal['queries']==COUNTS[ds]and not seal['GT_read']
    atomic(D/f'{b}_{ds}_worker.json',who)
    for i,row in enumerate(rows(ds),1):
        assert time.perf_counter()-started<p['budget']['max_stage_seconds']and shutil.disk_usage(R).free>p['budget']['min_disk_free_bytes']
        key=row['id'];rp=dest/'receipts'/(key+'.json');out=dest/'predictions'/(key+'.npz')
        if rp.exists():
            e=read(rp);assert e['fingerprint']==p['fingerprint']and sha(out)==e['sha256'];files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));continue
        assert not out.exists(),'Preserve unbound output, no overwrite'
        tick=time.perf_counter();origin=parent(b,ds,key,seal);arrays=calculate(b,ds,origin)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
        os.replace(out.with_suffix('.tmp'),out)
        with np.load(out)as z:assert all(np.array_equal(z[k],arrays[k])for k in arrays)
        e=dict(path=str(out.relative_to(D)),sha256=sha(out),parent=origin,worker=who,query_CPU_seconds=time.perf_counter()-tick,
            FullMBR_parent_bitwise=True,MAP_recomputed_spatialDP=True,new_Source=0,new_expert=0,new_GPU=0,GT_read=False,fingerprint=p['fingerprint'])
        atomic(rp,e);files[key]=dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
        atomic(D/f'{b}_{ds}_progress.json',dict(completed=i,total=COUNTS[ds],worker=who,seconds=time.perf_counter()-started,GT_read=False));print(b,ds,i,flush=True)
        del arrays;gc.collect()
    assert len(files)==COUNTS[ds]and not torch.cuda.is_initialized()
    atomic(D/f'{b}_{ds}_sealed.json',dict(queries=len(files),files=files,parent_seal_sha256=sha(sp),worker=who,seconds=time.perf_counter()-started,GT_read=False,fingerprint=p['fingerprint']))
def score(b,ds):
    require_gate();p=validate()
    import torch,numpy as np
    from full_score import metric
    from common import labels,report
    torch.set_num_threads(1);sp=D/f'{b}_{ds}_sealed.json';seal=read(sp);assert seal['queries']==COUNTS[ds]and not seal['GT_read']
    truth=labels(ds);old=read((RUN/ds/'metrics.json')if b=='TA'else CG/ds/'metrics.json')['private_rows'];old={x['id']:x for x in old}
    rr=[];checks=0;errmax=0.
    for row in rows(ds):
        e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256']
        g=truth[row['id']];gt={int(k):v for k,v in g['boxes'].items()};vals={}
        with np.load(D/e['path'])as z:
            r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
            for a in ARMS:
                vals[a],err=metric(z[a],r,gt,g['interval'],z[a+'_interval'].tolist(),z['positions8']);assert err<2e-10;errmax=max(errmax,err);checks+=1
        for a,ref in [('FullMBR','Full'),('SourcePlusMBR','Source')]:assert vals[a]==old[row['id']]['values'][ref]
        if b=='CG':assert vals['Source']==old[row['id']]['values']['SourceNative']
        rr.append(dict(id=row['id'],parent=str(g['parent']),duration_group=old[row['id']]['duration_group'],values=vals))
    pairs=[('FullMAP',x)for x in ARMS if x!='FullMAP']+[('SourceJointMAP','Source'),('FullMBR','SourcePlusMBR')]
    res=report(rr,ARMS,pairs,P/b/ds);atomic(D/b/ds/'metrics.json',dict(private_rows=rr,report=res))
    receipt=dict(status='scored',queries=len(rr),checks=checks,max_error=errmax,seal_sha256=sha(sp),whole_domain_sealed_before_GT=True,parent_MBR_metrics_exact=True,fingerprint=p['fingerprint'])
    atomic(D/f'{b}_{ds}_score_receipt.json',receipt);atomic(P/b/ds/'score_receipt.json',receipt)
def reporting(b,ds):
    require_gate();p=validate()
    import torch,numpy as np
    torch.set_num_threads(1);sp=D/f'{b}_{ds}_sealed.json';seal=read(sp);score=read(D/f'{b}_{ds}_score_receipt.json');assert score['seal_sha256']==sha(sp)and score['checks']==COUNTS[ds]*len(ARMS)
    rr=read(D/b/ds/'metrics.json')['private_rows'];res=read(P/b/ds/'report.json');assert len(rr)==COUNTS[ds]
    segments={};table=[];checks=0
    for i,row in enumerate(rr,1):
        e=seal['files'][row['id']];assert sha(D/e['path'])==e['sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];checks+=2
        o=e['parent'];assert sha(o['capture_path'])==o['capture_sha256']and sha(o['prediction_path'])==o['prediction_sha256']and sha(o['receipt_path'])==o['receipt_sha256'];checks+=3
        again=calculate(b,ds,o)
        with np.load(D/e['path'])as z:assert all(np.array_equal(z[k],again[k])for k in again)
        wk=f"{e['worker']['pid']}:{e['worker']['startticks']}";v=segments.setdefault(wk,dict(queries=0,query_CPU_seconds=0.));v['queries']+=1;v['query_CPU_seconds']+=e['query_CPU_seconds']
        for base in ['Source','SourceJointMAP','FullMBR','FullMAPFixedBoxes']:
            dt=100*(row['values']['FullMAP']['t']-row['values'][base]['t']);dv=100*(row['values']['FullMAP']['v']-row['values'][base]['v'])
            table.append([f'Q{i:05}',row['duration_group'],base,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,dt>5 or dv>5])
    for a,m in res['means'].items():
        for f,v in m.items():
            vv=[x['values'][a][f]for x in rr if x['values'][a][f]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
    folder=P/b/ds
    with(folder/'all_signed_severe_timegroups.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','timegroup','baseline','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_over5pp_union']);w.writerows(table)
    v=dict(status='verified_complete_MAP_DP_domain',queries=COUNTS[ds],SHA_checks=checks,all_MAP_and_MBR_arrays_replayed_exact=True,all_means_recomputed=True,
        worker_segments=segments,new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0,
        original_acquisition='Inherited parent Source/teacher receipts are bound and unchanged; CPU times here are cache-only research costs, not deployment latency.',
        seal_sha256=sha(sp),score_sha256=sha(D/f'{b}_{ds}_score_receipt.json'),fingerprint=p['fingerprint'])
    assert not torch.cuda.is_initialized();atomic(D/f'{b}_{ds}_reporting_receipt.json',v);atomic(folder/'reporting_receipt.json',v)
def queue():
    p=validate()
    if(D/'controller.json').exists():assert exited(read(D/'controller.json')),'Do not duplicate queue'
    who=identity();atomic(D/'controller.json',who);atomic(D/'state.json',dict(status='waiting_both_CG_fulls_and_reports',controller=who,fingerprint=p['fingerprint'],MAP_predictions=0,new_Source=0,new_expert=0))
    started=time.time()
    try:
        while not gate():
            assert time.time()-started<7*86400
            assert not read(CG/'state.json')['status'].startswith('failed'),'Preserve dependency failure; no automatic retry'
            time.sleep(30)
        if not(D/'preflight_receipt.json').exists():
            with(D/'preflight.log').open('a')as log:
                c=subprocess.Popen([sys.executable,'-B',__file__,'preflight'],stdout=log,stderr=subprocess.STDOUT);atomic(D/'preflight_child.json',identity(c.pid));assert c.wait()==0
        for b in BACKBONES:
            for ds in COUNTS:
                for command,suffix in [('predict','sealed'),('score','score_receipt'),('report','reporting_receipt')]:
                    if(D/f'{b}_{ds}_{suffix}.json').exists():continue
                    atomic(D/'state.json',dict(status=f'{b}_{ds}_{command}',controller=who,fingerprint=p['fingerprint']))
                    with(D/f'{b}_{ds}_{command}.log').open('a')as log:
                        c=subprocess.Popen([sys.executable,'-B',__file__,command,'--backbone',b,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT)
                        atomic(D/f'{b}_{ds}_{command}_child.json',identity(c.pid));assert c.wait()==0,'Preserve failure; no blind retry or missing-query suppression'
        for b in BACKBONES:
            for ds in COUNTS:
                score=read(D/f'{b}_{ds}_score_receipt.json');report=read(D/f'{b}_{ds}_reporting_receipt.json')
                assert score['queries']==COUNTS[ds]and report['score_sha256']==sha(D/f'{b}_{ds}_score_receipt.json')
        atomic(D/'state.json',dict(status='complete_scored_reported_pending_publication',controller=who,fingerprint=p['fingerprint']))
        assert exited(read(STRICT/'controller.json'))and read(STRICT/'state.json')['ablation_predictions']==0
        with(STRICT/'queue_after_MAP.log').open('a')as log:
            c=subprocess.Popen([sys.executable,'-B',str(R/'studies/strict_deletion_v1/run.py'),'queue'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(STRICT/'launch_after_MAP.json',dict(**identity(c.pid),MAP_plan_sha256=sha(D/'plan.json'),MAP_reporting_sha256={f'{b}_{ds}':sha(D/f'{b}_{ds}_reporting_receipt.json')for b in BACKBONES for ds in COUNTS}))
    except BaseException as e:
        atomic(D/'state.json',dict(status='failed_preserved',error=repr(e),traceback=traceback.format_exc(),fingerprint=p['fingerprint']));raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue','gate','preflight','predict','score','report']);a.add_argument('--backbone',choices=BACKBONES);a.add_argument('--dataset',choices=list(COUNTS));v=a.parse_args()
    dict(prepare=prepare,queue=queue,gate=require_gate,preflight=preflight,predict=lambda:predict(v.backbone,v.dataset),score=lambda:score(v.backbone,v.dataset),report=lambda:reporting(v.backbone,v.dataset))[v.command]()
