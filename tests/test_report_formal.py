"""Synthetic report tests only; no labels, predictions, RGB or models are read."""
import importlib.util
import json
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('formal_report',Path(__file__).resolve().parents[1]/'tools/report_formal.py')
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_pending_jobs_stay_empty(tmp_path):
    public,markdown=report.build_report(tmp_path)
    assert public['status']=='partial_or_pending'
    assert len(public['jobs'])==4
    assert all(j['status']=='pending'and'query_main_table'not in j for j in public['jobs'])
    assert '| tastvg | hc2 | Source | — | — | — | pending |'in markdown
    assert 'Standalone Source total s'in markdown
    assert not (tmp_path/'results').exists()  # Build is pure; only explicit CLI writes reports.


def test_completed_reports_are_hash_bound_anonymous_and_signed(tmp_path):
    output=tmp_path/'outputs/formal/tastvg/hc2';files={};rows=[]
    for index,value in enumerate([.2,.8]):
        key='private-query-'+str(index);values={}
        for arm,delta in [('Source',0.),('Spatial',.02),('Temporal',.03),('Full',.1 if index==0 else -.2),('WithoutOPD',.01)]:
            values[arm]=dict(v=value+delta,t=.5,s=.6,unobserved_actual_sIoU=.4)
        rows.append(dict(id=key,parent='PRIVATE-PARENT-'+str(index),dataset='hc2',caption='PRIVATE CAPTION',values=values))
        prediction=output/'predictions'/(key+'.npz');prediction.parent.mkdir(parents=True,exist_ok=True);prediction.write_bytes(('synthetic prediction '+str(index)).encode());prediction_hash=report.digest(prediction)
        files[key]=dict(path='predictions/'+key+'.npz',sha256=prediction_hash)
        put(output/'predictions'/(key+'.json'),dict(resume_hash='run-binding',prediction_sha256=prediction_hash,labels_read=False,
            total_seconds=12.+2*index,adapt=dict(fit_seconds=2.+2*index),
            engineering=dict(worker_peak_allocated_GPU_bytes=(14.+2*index)*2**30),
            capture=dict(peak_GPU_bytes=(16.-index)*2**30,Source224_seconds=1.,ReferDINO_query_forwards=1,caption='PRIVATE CAPTION',video_path='/private/video.mp4')))
    table={a:{k:.5 for k in report.ABLATION+report.SUPPLEMENT}for a in report.ARMS}
    metrics=dict(status='scored_sealed_predictions',provenance=dict(manifest_sha256='cohort-binding'),rows=rows,
        datasets=dict(hc2=dict(queries=2,query_main_table=table,paired={'Full-Source':dict(mean_pp=-5.)})))
    seal=dict(status='sealed_before_offline_scoring',labels_read=False,manifest_sha256='cohort-binding',queries=2,files=files,resume_hash='run-binding')
    put(output/'metrics.json',metrics);put(output/'sealed.json',seal)
    state=dict(jobs={'tastvg-hc2':dict(status='scored',metrics_sha256=report.digest(output/'metrics.json'),seal_sha256=report.digest(output/'sealed.json'))})
    put(tmp_path/'outputs/formal/state.json',state)
    public,markdown=report.build_report(tmp_path);job=public['jobs'][0]
    assert job['status']=='completed'and len(job['signed_queries'])==2
    assert job['signed_queries'][0]['change_vIoU_pp']['Full-Source']>0
    assert job['signed_queries'][1]['change_vIoU_pp']['Full-Source']<0
    assert job['signed_queries'][0]['parent']=='Parent00001'
    cost=job['efficiency'];assert cost['actual_full_query_wall_seconds_mean']==13.
    assert cost['actual_adapt_seconds_mean']==3.
    assert cost['peak_allocated_cuda_bytes_max']==16*2**30
    assert cost['Source_standalone_end_to_end_seconds_mean']is None
    serialized=json.dumps(public)+markdown
    for private in ['PRIVATE-PARENT','private-query','PRIVATE CAPTION','/private/video.mp4']:
        assert private not in serialized
    # A changed completion file cannot contribute a single result number.
    (output/'metrics.json').write_text('{}')
    public,_=report.build_report(tmp_path)
    assert public['jobs'][0]['status']=='blocked'
    assert 'signed_queries'not in public['jobs'][0]


def test_benchmark_actual_schema_is_verified_and_private_rows_are_not_published(tmp_path):
    output=tmp_path/'job';seal=dict(manifest_sha256='cohort-binding',runtime_fingerprint='frozen-runtime',files={'private-q1':{},'private-q2':{}})
    measured=[]
    for index in range(2):
        measured.append(dict(id='private-q'+str(index+1),parent='PRIVATE-PARENT-'+str(index),
            bitwise_sealed_regression=True,GT_read=False,
            Source=dict(wall_seconds=2.+2*index,peak_allocated_cuda_bytes=3+2*index),
            Full=dict(wall_seconds=6.+2*index,peak_allocated_cuda_bytes=10+2*index)))
    summary={}
    for arm in ['Source','Full']:
        times=[r[arm]['wall_seconds']for r in measured];peaks=[r[arm]['peak_allocated_cuda_bytes']for r in measured]
        summary[arm]=dict(queries=2,wall_seconds_mean=sum(times)/2,wall_seconds_min=min(times),wall_seconds_max=max(times),peak_allocated_cuda_bytes_max=max(peaks),peak_allocated_cuda_bytes_mean=sum(peaks)/2)
    summary['Full_to_Source_wall_ratio']=7/3
    benchmark=dict(status='passed',backbone='tastvg',dataset='hc2',manifest_sha256='cohort-binding',runtime_fingerprint='frozen-runtime',GT_read=False,
        sample_parents=2,warmup_excluded_from_means=True,warmup=measured[0],rows=measured,summary=summary,
        caches=dict(RGB=False,visual=False,teacher_results=False,model_weights_resident=True),model_load_seconds=dict(source=4.,teacher=5.))
    assert report.read_benchmark(output,'tastvg','hc2',seal)=={'status':'pending'}
    put(output/'benchmark.json',benchmark)
    public=report.read_benchmark(output,'tastvg','hc2',seal)
    assert public['status']=='passed'and public['predictions_verified']is True
    assert public['summary']['SourceStandalone']['wall_seconds_mean']==3.
    assert public['summary']['Full']['wall_seconds_mean']==7.
    assert public['Full_to_SourceStandalone_wall_ratio']==7/3
    assert 'private-q'not in json.dumps(public)and'PRIVATE-PARENT'not in json.dumps(public)
    benchmark['runtime_fingerprint']='wrong-runtime';put(output/'benchmark.json',benchmark)
    assert report.read_benchmark(output,'tastvg','hc2',seal)['status']=='blocked'
    benchmark['runtime_fingerprint']='frozen-runtime';benchmark['rows'][1]['bitwise_sealed_regression']=False;put(output/'benchmark.json',benchmark)
    assert report.read_benchmark(output,'tastvg','hc2',seal)['status']=='blocked'
    benchmark['rows'][1]['bitwise_sealed_regression']=True;benchmark['summary']['Source']['wall_seconds_mean']=.001;put(output/'benchmark.json',benchmark)
    assert report.read_benchmark(output,'tastvg','hc2',seal)['status']=='blocked'
    assert report.read_benchmark(output,'cgstvg','hc2',seal)=={'status':'pending'}
