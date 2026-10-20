"""CPU-only synthetic checks; never start models or read real labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import benchmark_formal as benchmark
from tools.run_formal import canonical


def row(index,parent=None,length=None):
    return dict(id=f'q{index:03d}',dataset='hc2',key=f'seed-alias-{index}',metadata=dict(
        source=parent or f'parent-{index}',video_sha256=f'{index:064x}',frame_ids=list(range(length or index+8))))


def test_parent_key_uses_official_source_before_original_alias():
    item=row(0,parent='official-parent')
    item['metadata']['original_video_id']='query-specific-original-alias'
    assert benchmark.parent_key(item)=='official-parent'
    del item['metadata']['source']
    assert benchmark.parent_key(item)=='query-specific-original-alias'
    del item['metadata']['original_video_id']
    assert benchmark.parent_key(item)==item['metadata']['video_sha256']


def test_selection_uniform_input_ranks_and_distinct_parents():
    records=[row(i)for i in range(40)]
    selected=benchmark.select_records(list(reversed(records)))
    assert len(selected)==16 and len({benchmark.parent_key(r)for r in selected})==16
    expected=[round(i*39/15)for i in range(16)]
    assert [r['id']for r in selected]==[f'q{i:03d}'for i in expected]
    duplicates=records+[row(99,parent='parent-0',length=8)]
    selected=benchmark.select_records(duplicates)
    assert len(selected)==16 and len({benchmark.parent_key(r)for r in selected})==16
    assert benchmark.select_records(records[:3])==records[:3]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path):
    folder=tmp_path/'outputs/formal';output=folder/'tastvg/hc2'
    manifest=tmp_path/'data/evaluation/formal/hc2.jsonl';manifest.parent.mkdir(parents=True)
    record=row(0);manifest.write_text(json.dumps(record)+'\n')
    proof={'code':{}};fingerprint=hashlib.sha256(canonical(proof)).hexdigest()
    binding=dict(manifest_sha256=benchmark.digest(manifest),runtime_fingerprint=fingerprint,backbone='tastvg')
    resume=hashlib.sha256(canonical(binding)).hexdigest()
    run=dict(**binding,resume_hash=resume,queries=1,labels_read=False)
    write(output/'run.json',run);write(output/'runtime_fingerprint.json',proof)
    prediction=output/'predictions/q000.npz';prediction.parent.mkdir();prediction.write_bytes(b'synthetic prediction')
    entry=dict(path='predictions/q000.npz',sha256=benchmark.digest(prediction))
    write(prediction.with_suffix('.json'),dict(resume_hash=resume,request_sha256=hashlib.sha256(canonical(record)).hexdigest(),prediction_sha256=entry['sha256'],labels_read=False))
    seal=dict(**run,status='sealed_before_offline_scoring',files={'q000':entry});write(output/'sealed.json',seal)
    metrics=output/'metrics.json';metrics.write_text('Scored bytes are hashed, never parsed or used to select samples.')
    job=dict(id='tastvg-hc2',backbone='tastvg',dataset='hc2',manifest=str(manifest.relative_to(tmp_path)),manifest_sha256=binding['manifest_sha256'],queries=1,output=str(output.relative_to(tmp_path)))
    plan=dict(jobs=[job]);write(folder/'plan.json',plan)
    state=dict(plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),jobs={'tastvg-hc2':dict(status='scored',metrics_sha256=benchmark.digest(metrics),seal_sha256=benchmark.digest(output/'sealed.json'))})
    write(folder/'state.json',state)
    return folder,output,record,state


def test_gate_accepts_only_complete_hash_bound_job_without_labels(tmp_path):
    folder,output,record,state=fixture(tmp_path)
    records,actual,seal,run=benchmark.completed_job(tmp_path,'tastvg','hc2')
    assert records==[record] and actual==output and seal['queries']==1
    assert not list(tmp_path.rglob('*labels*'))
    (output/'predictions/q000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='prediction changed'):benchmark.completed_job(tmp_path,'tastvg','hc2')


def test_pending_gate_refuses_before_any_inference(tmp_path):
    folder,output,record,state=fixture(tmp_path)
    state['jobs']['tastvg-hc2']['status']='running';write(folder/'state.json',state)
    with pytest.raises(ValueError,match='whole job sealed and scored'):benchmark.benchmark(tmp_path,'tastvg','hc2')
    with pytest.raises(NotImplementedError,match='deferred'):benchmark.benchmark(tmp_path,'cgstvg','hc2')


def test_source_and_full_must_both_reproduce_saved_arrays(tmp_path):
    boxes=np.ones((8,4));ids=list(range(8));source={'Source':boxes,'Native_interval':[0,8]}
    full=dict(Source=boxes,Live=boxes,intervals=dict(Native=[0,8],Strong=[1,7]))
    path=tmp_path/'prediction.npz';np.savez(path,Source=boxes,Native_interval=[0,8],Live=boxes,Strong_interval=[1,7],frame_ids=ids)
    assert benchmark.verify_outputs(source,full,ids,path)
    full['Live']=boxes+.001
    with pytest.raises(AssertionError,match='Live'):benchmark.verify_outputs(source,full,ids,path)


def test_summary_uses_actual_standalone_wall_not_source_component():
    rows=[dict(Source=dict(wall_seconds=2.,peak_allocated_cuda_bytes=3),Full=dict(wall_seconds=6.,peak_allocated_cuda_bytes=10)),
          dict(Source=dict(wall_seconds=4.,peak_allocated_cuda_bytes=5),Full=dict(wall_seconds=8.,peak_allocated_cuda_bytes=12))]
    summary=benchmark.aggregate(rows)
    assert summary['Source']['wall_seconds_mean']==3.
    assert summary['Full']['wall_seconds_mean']==7.
    assert summary['Full_to_Source_wall_ratio']==7/3
