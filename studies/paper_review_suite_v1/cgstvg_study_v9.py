"""CG-STVG LAST: pinned native Source and fixed generic-feedback comparison."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
from cgstvg_runtime_v9 import ta_complete_gate,verify_vendor,prepare_asset_links
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
ARMS=['SourceNative','Source','SharedCorrection','Space','Full']
CODE=['cgstvg_study_v9.py','cgstvg_runtime_v9.py','cgstvg_score.py','cgstvg_interfaces_v9.py','cgstvg_memory_stream_v8.py']
MATH=['stvg/spatial.py','stvg/_spatial_math.py','stvg/evidence.py','stvg/initialization.py','stvg/consensus.py','stvg/positions.py','stvg/prototype.py','stvg/representation.py','stvg/temporal.py','stvg/predictor.py','stvg/teacher_bridge.py','stvg/teacher_worker.py','configs/default.json']


def prepare():
    assert not(D/'cgstvg_v9_plan.json').exists();check();verify_vendor();prepare_asset_links()
    audit=read(D/'cgstvg_v9_runtime_cpu_tests.json');assert audit['status']=='passed_no_CG_model'
    budget=read(D/'budget_plan.json')
    p=dict(schema='paper_review_CGSTVG_v9_native_segment_boundary',suite_plan_sha256=sha(D/'plan.json'),selection_sha256=sha(D/'selection.json'),seed=20261024,
           code_sha256={n:sha(Path(__file__).with_name(n))for n in CODE},math_sha256={n:sha(R/n)for n in MATH},
           cpu_tests_sha256=sha(D/'cgstvg_v9_runtime_cpu_tests.json'),official_interface_audit_sha256=sha(D/'cgstvg_cpu_interface_audit.json'),
           vendor_revision='f062d84b72aee8be4425a4a5ab20aa5b077ca142',
           native_input_boundary='Frozen runtime/source/vidstg manifest: input end_frame may be inclusive or one-past; convert to min(end_frame,frame_count-1) BEFORE official native128sampling. No postdecode truncation/padding, no change in available video or resolution/precision.',
           boundary_input_manifest_sha256=sha(R/'runtime/source/vidstg/manifest.json'),
           native_box_readout='Official negativexyxy clamp permits exactlyzero area. Accept nonnegative widths/heights; no epsilon, no coordinate changes, no case removal. Prior strictpositive assert was an adapter error.',
           acquisition_guard_sha256=sha(D/'cgstvg_v9_failure_acquisition_guard.json'),
           before_feedback_acquisition='Persist Source/teacher acquisition snapshot and SHA before CPUfeedback so numerical/postprocess failures need not rerun model/evidence.',
           mask_cache='Clear only upstream deterministiccompute_mask LRU before/after each capture; identical recomputation, no change to perwindow arithmetic.',
           exact_window_schedule='64independent existing attention windows per batch; complete perwindow tokens and original globalmask row mapping. QKV/output projections unchanged.16CPU/GPU126window/7mask tests+512window case bitwise. Every7previous successful nativecapture is revalidated with one new Source-only forward before byte reuse; no extra expert.',
           expert_lifetime='Original resident SAME expert restored; memorybounded independentwindow processing removes need for perquery reload. Seven completed v6 prediction/captures reused byteexact only after Source-only equality checks.',
           source_cost_origin='Current cohort reuses6 completed v5 Source forwards; prior v4 extra4 completed Source forwards and v3/v4/v5 three failed attempts remain separately charged. Prior expert evidence reused with original cost, never zero-cost deployment.',
           allocator='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; same native arithmetic; firsttwo previous successful HC cases must reproduce every head/grid/posterior/box bitwise.',
           historical_attempts=dict(v3_failed_Source_attempts=1,v4_completed_Source_forwards=4,v4_failed_Source_attempts=1,v4_expert_calls=2,v4_successful_predictions=2,reused_teacher_contexts=2,all_cost_retained_separately=True),
           memory_parity_sha256=sha(D/'cgstvg_memory_stream_parity_receipt.json'),
           allocation='Scoped eval-only exact per-window nativeFP32 attention scheduled64independentwindowrows at a time; fullN and allwindows retained in original order, globalmask modulo preserved. No video or expertcontext splitting. Previous bulk executableOOM preserved; native reference uses these same audited equations with bounded intermediate storage.',
           previous_failed_execution=dict(fingerprint='4d9f93321edb06201806dfd037b59375a03c5e8d7e303028684805f6725c7dde',native_Source_attempts=1,completed_Source_forwards=0,expert_calls=0,outputs=0,preserved=True),
           source_weights=dict(hc2='official VidSTG EMA checkpoint',vidstg='official HCSTVG2 EMA checkpoint'),
           native='Original CG-STVG FP32, official420-pixel height, native single test grid. No TA offset/replay/head or six-layer/hires fusion. Original internal CG context/actionness untouched.',
           arms=dict(SourceNative='Original native joint MAP and native official boxes.',Source='Same last-layer legal endpoint posterior, MBR, SAME native boxes; separates readout benefit.',SharedCorrection='Original eight observation rule on native MAP; transferred fixed shared correction and same MBR.',Space='Shared correction plus unchanged SimplePath native MBR interval.',Full='Same SimplePath auxiliary ROI, one all-video Native2D pair kappa64 beta1 context1.5, one update and MBR, finalSimplePath.'),
           box_readout='CG official negative xyxy clamp on BOTH datasets; no upper clamp. Every arm uses this same native output convention.',
           calibration='Transfer unchanged TA source-training scales/truncation/gamma, not CG recalibration. No CG target GT tuning. Gamma original source-only selection provenance unresolved; kappa had TA target development selection.',
           freeze=dict(kappa=64,beta=1,context=1.5,rounds=1,observed=.5,unobserved=.5,link=.25,expert_K=8,training=0,optimizer=0,new_expert_models=0),
           timing='Report actual native FP32 Source, same ReferDINO8, feature capture/CPUgeometry/prototype/DP and loading separately; no comparison of cache hits with uncached deployment.',
           teacher_runtime_fingerprint=budget['teacher_runtime_fingerprint'],
           screening='64fixed parents/domain; real-checkpoint no-op first2/domain (extra4 native forwards). Native decoder/feature/weights/source checks must pass. Failures preserve outputs/trace/source and do not silently change resolution/precision/sampling. No scoring partial cohorts.',
           confirmation='128fixed parent-disjoint/domain after complete screening and all4 real no-op/pooling bridges. NO accuracy selection or parameter search. Fullsets previously exposed, not unseen.',
           budget=dict(max_original_Source_forwards=396,max_new_same_expert8_joint_queries=384,screen_wall_seconds=14400,confirm_wall_seconds=28800,maximum_private_bytes=100*2**30),
           real_CG_model_start_gate='ALL TA consistency, budget and controlled-shift confirmation score receipts exist; TA owned processes exited; acquire sharedGPUlock. Preparing this plan never constructs a CG model.',
           all_stage_predictions_sealed_before_independent_CPU_official_GT=True,CG_GT_inference=False,Fig1='excluded')
    p['fingerprint']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    atomic(D/'cgstvg_v9_plan.json',p);atomic(P/'cgstvg_v9_protocol.json',p)
    for n in CODE:
        f=D/'source_snapshot'/('CGv9_'+n);assert not f.exists();f.write_bytes(Path(__file__).with_name(n).read_bytes())
    atomic(D/'cgstvg_v9_state.json',dict(status='prepared_CPU_only_waiting_all_TA_complete',fingerprint=p['fingerprint'],model_constructed=False))
    print(p['fingerprint'])


def frozen():
    p=read(D/'cgstvg_v9_plan.json');assert p['fingerprint']==hashlib.sha256(json.dumps({k:v for k,v in p.items()if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
    assert sha(D/'selection.json')==p['selection_sha256']
    for n,h in p['code_sha256'].items():assert sha(Path(__file__).with_name(n))==h
    for n,h in p['math_sha256'].items():assert sha(R/n)==h
    check();return p


def predict(stage):
    import numpy as np
    import torch
    from stvg.runtime import setup,lease
    from stvg.decode import decode
    from stvg.positions import observation_positions
    from stvg.teacher_bridge import ResidentTeacher
    from cgstvg_runtime_v9 import load_model,capture,feedback
    from cgstvg_interfaces_v9 import frame_grid
    p=frozen();dependencies=ta_complete_gate();setup();torch.manual_seed(p['seed']);np.random.seed(p['seed']);gpu=lease()
    atomic(D/('cgstvg_v9_'+stage+'_worker.json'),identity());selection=read(D/'selection.json');files={};started=time.perf_counter();counts=dict(queries=0,Source_forwards=0,new_expert=0,native_noop_queries=0,reused_Source_forwards=0,reused_queries=0)
    teacher=ResidentTeacher(cache_dir=D/'cgstvg_teacher_cache',fingerprint=p['teacher_runtime_fingerprint'])
    try:
        for ds,groups in selection.items():
            model,load=load_model(ds);files[ds]={}
            atomic(D/('cgstvg_v9_'+stage+'_'+ds+'_strict_load.json'),load)
            for i,row in enumerate(groups[stage]):
                assert time.perf_counter()-started<p['budget'][stage+'_wall_seconds']
                assert counts['new_expert']<p['budget']['max_new_same_expert8_joint_queries']
                dest=D/'cgstvg_v9'/stage/ds;dest.mkdir(parents=True,exist_ok=True);key=row['id'];out=dest/(key+'.npz');cp=dest/(key+'.capture.pt');rp=dest/(key+'.receipt.json')
                if rp.exists():
                    saved=read(rp);assert saved['fingerprint']==p['fingerprint']and saved['acquisition_origin']['previous_fingerprint']=='e5e9b950eab71603841d314c1928e84ab246ec52de8cc7cf37d86f6964b49416'
                    assert sha(out)==saved['sha256']and sha(cp)==saved['capture_sha256']
                    old_forwards=saved['source_forwards']
                    files[ds][key]=dict(saved,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
                    counts['queries']+=1;counts['reused_queries']+=1;counts['reused_Source_forwards']+=old_forwards;counts['native_noop_queries']+=int(saved['native_noop'])
                    atomic(D/'cgstvg_v9_progress.json',dict(stage=stage,dataset=ds,counts=counts,seconds=time.perf_counter()-started,**identity(),GT_read=False));continue
                assert not any(x.exists()for x in [out,cp,rp]),'Preserve prior/partial CG outputs, no blind retry'
                metadata=dict(row['metadata']);metadata['frame_ids']=frame_grid(metadata,ds)
                frames,ids=decode(metadata,ds);assert ids==metadata['frame_ids'];tick=time.perf_counter()
                atomic(dest/(key+'.attempt.json'),dict(status='before_native_Source_capture',worker=identity(),fingerprint=p['fingerprint'],time=time.time(),GT_read=False,source_forwards_completed_so_far=counts['Source_forwards'],expert_completed_so_far=counts['new_expert']))
                cap=capture(model,frames,ids,metadata['caption'],noop=stage=='screen'and i<2)
                counts['Source_forwards']+=cap['source_forwards'];counts['native_noop_queries']+=int(cap['native_noop'])
                old_receipt=D/'cgstvg_v4'/'screen'/ds/(key+'.receipt.json')
                if stage=='screen'and old_receipt.exists():
                    previous=read(old_receipt);assert sha(D/previous['capture_path'])==previous['capture_sha256']
                    saved=torch.load(D/previous['capture_path'],map_location='cpu',weights_only=False)
                    assert np.array_equal(cap['Source'],saved['Source'])and np.array_equal(cap['q0'],saved['q0'])
                    assert cap['native_map']==saved['native_map']and cap['native_mbr']==saved['native_mbr']
                    assert all(torch.equal(cap['heads'][k],v)for k,v in saved['heads'].items())
                    assert all(torch.equal(cap['grids'][k],v)if torch.is_tensor(v)else cap['grids'][k]==v for k,v in saved['grids'].items())
                    cap['previous_v4_capture_bitwise']=True
                iv=cap['native_map'];pos=observation_positions(ids,[ids.index(iv[0]),ids.index(iv[1]-1)])
                if key in read(D/'cgstvg_v9_failure_acquisition_guard.json')['query_ids']:
                    from stvg.teacher_bridge import evidence_key
                    cache_key,context=evidence_key(frames[pos],metadata['caption'],positions=pos,frame_ids=ids,dataset=ds,fingerprint=p['teacher_runtime_fingerprint'])
                    cached=D/'cgstvg_teacher_cache'/(cache_key+'.json');assert cached.exists(),'Previously paid expertcontext must be recovered, never blindly reacquired'
                    assert read(cached)['context']==context and sha(cached.with_suffix('.pt'))==read(cached)['sha256']
                native=teacher(frames[pos],metadata['caption'],positions=pos,frame_ids=ids,dataset=ds);native['positions']=pos;counts['new_expert']+=native['query_forwards']
                acquired=dest/(key+'.acquired.pt');torch.save(dict(Source_capture=cap,teacher=native),acquired)
                atomic(dest/(key+'.acquired.json'),dict(path=str(acquired.relative_to(D)),sha256=sha(acquired),fingerprint=p['fingerprint'],Source_forwards=cap['source_forwards'],teacher_new_forwards=native['query_forwards'],teacher_context_cache_key=native['cache_key'],GT_read=False))
                arrays,details,cache=feedback(cap,native,ds)
                # A second CPU decode from saved actual features verifies coupled
                # geometry/pool/posterior/boxes before committing each bridge.
                if cap['native_noop']:
                    replay,again,_=feedback(cap,native,ds)
                    assert all(np.array_equal(arrays[k],replay[k])for k in arrays)
                    details['coupled_cached_CPU_replay_bitwise']=True
                if stage=='screen'and old_receipt.exists():
                    assert sha(D/previous['path'])==previous['sha256']
                    with np.load(D/previous['path'])as old:
                        assert all(np.array_equal(arrays[k],old[k])for k in arrays)
                    assert native['cache_hit']and native['query_forwards']==0
                    details['previous_v4_all_predictions_bitwise']=True
                    details['old_acquisition_cost_preserved_in_historical_origin']=True
                np.savez_compressed(out,**arrays)
                cap.update(method_cache=cache,teacher_evidence_key=native['cache_key'],positions8=pos)
                torch.save(cap,cp)
                rec=dict(path=str(out.relative_to(D)),sha256=sha(out),capture_path=str(cp.relative_to(D)),capture_sha256=sha(cp),fingerprint=p['fingerprint'],worker=identity(),metadata_frame_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),frames=len(ids),native_noop=cap['native_noop'],source_forwards=cap['source_forwards'],source_seconds=cap['source_seconds'],source_load=load,expert_new_forwards=native['query_forwards'],expert_cache_hit=native['cache_hit'],expert_cost=native.get('evidence_original_cost',native['cost']),expert_model_load_seconds=native['model_load_seconds'],pipeline_seconds=time.perf_counter()-tick,details=details,GT_read=False)
                atomic(rp,rec);files[ds][key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['queries']+=1
                atomic(D/'cgstvg_v9_progress.json',dict(stage=stage,dataset=ds,counts=counts,seconds=time.perf_counter()-started,**identity(),GT_read=False));print(json.dumps(dict(stage=stage,dataset=ds,queries=counts['queries'],native_noop=cap['native_noop'])),flush=True)
                del frames,cap,arrays,cache,native;gc.collect();torch.cuda.empty_cache()
            del model;gc.collect();torch.cuda.empty_cache()
        assert counts['queries']==(128 if stage=='screen'else 256)
        assert counts['native_noop_queries']==(4 if stage=='screen'else 0)
        atomic(D/('cgstvg_v9_'+stage+'_sealed.json'),dict(files=files,counts=counts,fingerprint=p['fingerprint'],GT_read=False,TA_dependency_receipts=dependencies,seconds=time.perf_counter()-started))
    finally:teacher.close();gpu.close()


def score(stage):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    from full_score import metric,cluster
    from cgstvg_score import native_official_metric
    torch.set_num_threads(1);p=frozen();seal=read(D/('cgstvg_v9_'+stage+'_sealed.json'));assert not seal['GT_read']
    assert seal['counts']['queries']==(128 if stage=='screen'else 256)
    rows_by_domain={};checks=0;maxerr=0.;selection=read(D/'selection.json');job=read(RUN/'plan.json')['jobs']
    for ds,groups in selection.items():
        label=PROVIDER/job[ds]['labels'];assert sha(label)==job[ds]['labels_sha256'];gt={x['id']:x for x in map(json.loads,label.read_text().splitlines())if x['dataset']==ds};rows=[];signed=[];severe=[];cost=[]
        for i,row in enumerate(groups[stage],1):
            key=row['id'];e=seal['files'][ds][key];assert sha(D/e['path'])==e['sha256']and sha(D/e['capture_path'])==e['capture_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];g=gt[key];truth={int(k):v for k,v in g['boxes'].items()};vals={}
            with np.load(D/e['path'])as z:
                r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
                for arm in ARMS:
                    iv=z[arm+'_interval'].tolist();vals[arm],err=metric(z[arm],r,truth,g['interval'],iv,z['positions8']);original=native_official_metric(z[arm],z['frame_ids'].tolist(),r['metadata'],truth,g['interval'],iv,ds);err=max(err,max(abs(vals[arm][f]-original[f])for f in ['v','t','s']));assert err<2e-10;maxerr=max(maxerr,err);checks+=1
            group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s'
            rows.append(dict(parent=str(g['parent']),values=vals,duration_group=group));cost.append(e)
            signed.append([f'Q{i:03d}',group]+[vals[a][f]for a in ARMS for f in ['t','v']])
            for arm in ARMS[2:]:
                dt=100*(vals[arm]['t']-vals['Source']['t']);dv=100*(vals[arm]['v']-vals['Source']['v']);severe.append([f'Q{i:03d}',arm,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5])
        pairs=[('Source','SourceNative'),('SharedCorrection','Source'),('Space','SharedCorrection'),('Full','Space'),('Full','Source')]
        groups_report={gg:dict(queries=sum(x['duration_group']==gg for x in rows),paired={a+'-'+b:cluster([x for x in rows if x['duration_group']==gg],a,b)for a,b in pairs})for gg in ['short_le5s','long_gt5s']if any(x['duration_group']==gg for x in rows)}
        report=dict(dataset=ds,stage=stage,backbone='CG-STVG',queries=len(rows),parents=len(rows),means={a:{f:float(np.mean([x['values'][a][f]for x in rows]))for f in ['v','t','s']}for a in ARMS},paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs},time_groups=groups_report,severe_union={a:sum((x['values'][a]['t']-x['values']['Source']['t']< -.05)or(x['values'][a]['v']-x['values']['Source']['v']< -.05)for x in rows)for a in ARMS[2:]},cost=dict(source_forwards=sum(x['source_forwards']for x in cost),new_expert_forwards=sum(x['expert_new_forwards']for x in cost),mean_Source_seconds=float(np.mean([x['source_seconds']for x in cost])),mean_pipeline_seconds=float(np.mean([x['pipeline_seconds']for x in cost])),native_noop_queries=sum(x['native_noop']for x in cost)),fingerprint=p['fingerprint'],exposure='Fixed new contrast; full target sets historically exposed. No target parameter selection.',source_baseline='SourceNative=official MAP; Source=native last-layer MBR with identical native boxes',calibration=p['calibration'])
        pub=P/'cgstvg_v9'/ds/stage;atomic(pub/'report.json',report);atomic(D/'cgstvg_v9'/stage/ds/'metrics.json',dict(report,private_rows=rows))
        for filename,header,values in [('signed.tsv',['query','duration_group']+[a+'_'+f for a in ARMS for f in ['t','v']],signed),('severe_signed.tsv',['query','arm','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once'],severe)]:
            with(pub/filename).open('w')as out:w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(header);w.writerows(values)
        rows_by_domain[ds]=len(rows)
    atomic(D/('cgstvg_v9_'+stage+'_score_receipt.json'),dict(status='scored',stage=stage,domain_queries=rows_by_domain,checks=checks,max_error=maxerr,metrics='pinned official CG evaluator + common official evaluator + independent dense',seal_sha256=sha(D/('cgstvg_v9_'+stage+'_sealed.json')),GT_only_after_whole_stage_seal=True))


def controller():
    p=frozen();ta_complete_gate()
    atomic(D/'cgstvg_v9_controller.json',identity());atomic(D/'cgstvg_v9_state.json',dict(status='running_after_all_TA',fingerprint=p['fingerprint'],controller=identity()))
    for stage in ['screen','confirm']:
        if stage=='confirm':
            s=read(D/'cgstvg_v9_screen_sealed.json');receipt=read(D/'cgstvg_v9_screen_score_receipt.json');assert s['counts']['native_noop_queries']==4 and receipt['domain_queries']==dict(hc2=64,vidstg=64)
            atomic(D/'cgstvg_v9_qualification.json',dict(eligible=True,basis='Complete two-domain screen and all4 model/coupled no-op bridges. No metric-based selection.',accuracy_selection=False))
        for command in ['predict','score']:
            assert not(D/('cgstvg_v9_'+stage+('_sealed.json'if command=='predict'else'_score_receipt.json'))).exists()
            atomic(D/'cgstvg_v9_state.json',dict(status=stage+'_'+command,controller=identity(),fingerprint=p['fingerprint']))
            child=subprocess.Popen([sys.executable,'-B',__file__,command,'--stage',stage]);atomic(D/('cgstvg_v9_'+stage+'_'+command+'_child.json'),identity(child.pid));code=child.wait();atomic(D/('cgstvg_v9_'+stage+'_'+command+'_exit.json'),dict(code=code))
            if code:
                atomic(D/'cgstvg_v9_state.json',dict(status='failed_preserved_no_automatic_retry',stage=stage,command=command,exit=code,fingerprint=p['fingerprint']));raise RuntimeError('CG failed; preserve frozen source/logs/partial outputs and inspect before any new version')
    atomic(D/'cgstvg_v9_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','predict','score']);ap.add_argument('--stage',choices=['screen','confirm']);args=ap.parse_args()
    {'prepare':prepare,'run':controller,'predict':lambda:predict(args.stage),'score':lambda:score(args.stage)}[args.command]()
