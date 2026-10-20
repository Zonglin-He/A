"""Synthetic subprocess mocks only: never import or execute a model worker."""
import importlib.util
import json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('formal_controller',Path(__file__).resolve().parents[1]/'tools/run_formal.py')
formal=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(formal)


def fixture(tmp_path):
    (tmp_path/'stvg').mkdir();(tmp_path/'stvg/worker.py').write_text('# synthetic code\n')
    (tmp_path/'configs').mkdir();(tmp_path/'configs/default.json').write_text('{"frozen":true}')
    directory=tmp_path/'data/evaluation/formal';directory.mkdir(parents=True)
    for ds in ['hc2','vidstg']:
        (directory/(ds+'.jsonl')).write_text(json.dumps(dict(id='synthetic',dataset=ds,key='seed41',metadata=dict(source='private-synthetic-parent')))+'\n')
        # No labels are read by the controller, even during mocked offline scoring.
        (directory.parent/(ds+'_labels.jsonl')).write_text('synthetic label boundary, not parsed')
    return tmp_path


class FakeProcess:
    def __init__(self,code):self.pid=424242;self.code=code
    def wait(self):return self.code
    def poll(self):return self.code
    def terminate(self):raise AssertionError('No synthetic child remains running')


class FakeProcesses:
    def __init__(self,fail_worker=False):self.commands=[];self.fail_worker=fail_worker
    def __call__(self,command,**kwargs):
        self.commands.append(command);get=lambda name:command[command.index(name)+1]
        output=Path(get('--output'))
        if command[3]=='stvg.worker':
            assert '--labels'not in command
            if self.fail_worker:return FakeProcess(17)
            backbone=get('--backbone');manifest=Path(get('--manifest'));rows=[json.loads(x)for x in manifest.read_text().splitlines()]
            proof=dict(code={},assets={},backbone=backbone)
            fingerprint=__import__('hashlib').sha256(formal.canonical(proof)).hexdigest()
            binding=dict(manifest_sha256=formal.digest(manifest),runtime_fingerprint=fingerprint,backbone=backbone)
            resume=__import__('hashlib').sha256(formal.canonical(binding)).hexdigest()
            run=dict(**binding,resume_hash=resume,queries=len(rows),labels_read=False)
            formal.write(output/'run.json',run);formal.write(output/'runtime_fingerprint.json',proof);files={}
            for row in rows:
                p=output/'predictions'/(row['id']+'.npz');p.parent.mkdir(exist_ok=True);p.write_bytes(b'synthetic-prediction')
                formal.write(p.with_suffix('.json'),dict(resume_hash=resume,request_sha256=__import__('hashlib').sha256(formal.canonical(row)).hexdigest(),prediction_sha256=formal.digest(p),labels_read=False))
                files[row['id']]=dict(path=str(p.relative_to(output)),sha256=formal.digest(p))
            formal.write(output/'sealed.json',dict(**run,status='sealed_before_offline_scoring',files=files))
        else:
            assert command[3:5]==['stvg','score']
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
            predictions=Path(get('--predictions'));assert (predictions/'sealed.json').exists()
            formal.write(output,dict(status='scored_sealed_predictions',synthetic=True))
        return FakeProcess(0)


def test_plan_only_never_launches_subprocess(tmp_path):
    root=fixture(tmp_path);fake=FakeProcesses();state=formal.controller(root,popen=fake)
    assert state['status']=='planned'and not fake.commands
    plan=json.loads((root/'outputs/formal/plan.json').read_text())
    assert [j['id']for j in plan['jobs']]==['tastvg-hc2','tastvg-vidstg']
    assert all(j['manifest_sha256']for j in plan['jobs'])


def test_ta_stage_runs_and_scores_sequentially_then_bound_resume(tmp_path):
    root=fixture(tmp_path);fake=FakeProcesses();state=formal.controller(root,execute=True,popen=fake)
    assert state['status']=='completed'and len(fake.commands)==4
    for job in state['jobs'].values():
        assert job['status']=='scored'and job['metrics_sha256']
        assert [e['phase']for e in job['processes']]==['worker','score']
        assert all(e['pid']==424242 and e['exit_code']==0 for e in job['processes'])
    again=FakeProcesses();state=formal.controller(root,execute=True,resume=True,popen=again)
    assert state['status']=='completed'and not again.commands
    with pytest.raises(ValueError,match='explicit --resume'):
        formal.controller(root,execute=True,popen=FakeProcesses())


def test_failure_stops_without_retry_or_scoring(tmp_path):
    root=fixture(tmp_path);fake=FakeProcesses(fail_worker=True)
    with pytest.raises(RuntimeError,match='17'):formal.controller(root,execute=True,popen=fake)
    state=json.loads((root/'outputs/formal/state.json').read_text())
    assert state['status']=='failed'and len(fake.commands)==1
    assert state['jobs']['tastvg-hc2']['status']=='failed'
    assert state['jobs']['tastvg-vidstg']['status']=='pending'


def test_changed_code_rejects_resume_without_launch(tmp_path):
    root=fixture(tmp_path);formal.controller(root,popen=FakeProcesses())
    (root/'stvg/worker.py').write_text('# changed frozen code\n');fake=FakeProcesses()
    with pytest.raises(ValueError,match='Frozen plan changed'):
        formal.controller(root,execute=True,resume=True,popen=fake)
    assert not fake.commands


def test_corrupt_sealed_prediction_blocks_skipping_scored_job(tmp_path):
    root=fixture(tmp_path);formal.controller(root,execute=True,popen=FakeProcesses())
    (root/'outputs/formal/tastvg/hc2/predictions/synthetic.npz').write_bytes(b'changed')
    fake=FakeProcesses()
    with pytest.raises(ValueError,match='Sealed prediction changed'):
        formal.controller(root,execute=True,resume=True,popen=fake)
    assert not fake.commands


def test_live_previous_child_prevents_duplicate_execution(tmp_path):
    import os
    root=fixture(tmp_path);formal.controller(root,popen=FakeProcesses())
    path=root/'outputs/formal/state.json';state=json.loads(path.read_text())
    state['jobs']['tastvg-hc2'].update(status='running',pid=os.getpid(),process_start_ticks=formal.process_identity(os.getpid()))
    formal.write(path,state);fake=FakeProcesses()
    with pytest.raises(RuntimeError,match='still running'):
        formal.controller(root,execute=True,resume=True,popen=fake)
    assert not fake.commands


def test_postprocessing_finishes_ta_without_queuing_cg(tmp_path,monkeypatch):
    root=fixture(tmp_path);formal.controller(root,execute=True,popen=FakeProcesses())
    (root/'docs').mkdir();(root/'docs/history.md').write_text('当前任务：synthetic\n')
    calls=[]
    def fake_launch(command,root,log,state,path,task,phase,popen):
        calls.append(command)
        if 'tools/benchmark_formal.py'in command:
            ds=command[command.index('--dataset')+1];formal.write(root/f'outputs/formal/tastvg/{ds}/benchmark.json',{'synthetic':True})
        elif 'tools/qualitative.py'in command:
            for ds in ['hc2','vidstg']:formal.write(root/f'outputs/formal/qualitative/tastvg/{ds}/selection.json',{'synthetic':True})
        elif 'tools/report_formal.py'in command:
            formal.write(root/'results/formal/results.json',{'synthetic':True});(root/'results/formal/report.md').write_text('synthetic')
        elif command[-1]=='snapshot':formal.write(root/'docs/sources.json',{'synthetic':True})
    monkeypatch.setattr(formal,'launch',fake_launch)
    formal.finish_ta_stage(root)
    state=json.loads((root/'outputs/formal/state.json').read_text())
    assert state['status']=='awaiting_github_sync'and len(calls)==7
    assert all('cgstvg'not in cmd for cmd in calls)
    assert all(t['status']=='completed'for t in state['postprocess'].values())
    formal.finish_ta_stage(root)
    assert len(calls)==7
