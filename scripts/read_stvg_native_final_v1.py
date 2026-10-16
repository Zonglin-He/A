"""Verify saved seals and aggregate Native-clock control; no model or GT access."""
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_component_ablation128_v1'
NATIVE = BASE / 'native_clock/revision002/trials/N000'


def main():
    hashes = {}

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    def verify(path):
        cpu = read(path / 'CPU_COMPLETION.json')
        score = read(path / 'POSTSEAL_SCORE_BARRIER.json')
        student = read(path / 'STUDENT_GLOBAL_BARRIER.json')
        root = read(path / 'ROOT_READBACK.json')
        global_ = read(path / 'GLOBAL_PREDICTION_BARRIER.json')
        assert cpu['status'] == score['status'] == student['status'] == global_['status'] == 'complete'
        assert root['status'] == 'pass'
        assert cpu['count'] == root['count'] == student['count'] == global_['count'] == 128
        assert cpu['score_barrier_sha256'] == sha(path / 'POSTSEAL_SCORE_BARRIER.json')
        assert root['barrier_sha256'] == sha(path / 'STUDENT_GLOBAL_BARRIER.json')
        assert global_['student_barrier_sha256'] == sha(path / 'STUDENT_GLOBAL_BARRIER.json')
        assert global_['root_readback_sha256'] == sha(path / 'ROOT_READBACK.json')
        for key, filename in [('rows', 'POSTSEAL_ROWS.json'), ('summary', 'POSTSEAL_SUMMARY.json'), ('paired', 'POSTSEAL_PAIRED.json')]:
            assert score[key + '_sha256'] == sha(path / filename)
        assert score['gt_exposure_sha256'] == sha(path / 'POSTSEAL_GT_EXPOSURE.json')
        rows = read(path / 'POSTSEAL_ROWS.json')['rows']
        summary = read(path / 'POSTSEAL_SUMMARY.json')['summary']
        errors = []
        for row in summary:
            if row['metric'] not in rows[0]['metrics']:
                continue
            values = [r['metrics'][row['metric']] for r in rows if r['dataset'] == row['dataset'] and r['arm'] == row['arm'] and r['metrics'][row['metric']] is not None]
            assert len(values) == row['n_valid']
            errors.append(abs(math.fsum(values) / len(values) - row['mean']) if values else 0.)
        assert max(errors) < 1e-12
        return rows, max(errors), student

    native, native_error, student = verify(NATIVE)
    old, old_error, _ = verify(BASE / 'trials/A000')
    table, pairs = [], []
    for ds in ['hc2', 'vidstg']:
        arms = {}
        for label, rows, arm in [('Frozen', old, 'F'), ('Temporal', old, 'T'), ('Spatial_QA_clock', old, 'S_M2'), ('Full', old, 'TS_M2'), ('Spatial_Native_clock', native, 'S_M2')]:
            rr = [r for r in rows if r['dataset'] == ds and r['arm'] == arm]
            assert len(rr) == 64 and len({r['case_id'] for r in rr}) == 64
            arms[label] = {r['case_id']: r for r in rr}
            table.append(dict(dataset=ds, arm=label, count=64, metrics_percent={m: math.fsum(r['metrics'][m] for r in rr) * 100 / 64 for m in ['m_vIoU', 'm_tIoU', 'm_sIoU_dense_GT']}))
        for a, b in [('Spatial_Native_clock', 'Frozen'), ('Full', 'Spatial_Native_clock'), ('Spatial_QA_clock', 'Spatial_Native_clock'), ('Full', 'Spatial_QA_clock'), ('Full', 'Temporal')]:
            assert arms[a].keys() == arms[b].keys()
            delta = [{ 'case_id': c, 'delta_pp': (arms[a][c]['metrics']['m_vIoU'] - arms[b][c]['metrics']['m_vIoU']) * 100} for c in arms[a]]
            pairs.append(dict(dataset=ds, contrast=a+' minus '+b, mean_delta_pp=math.fsum(r['delta_pp'] for r in delta)/64, positive=sum(r['delta_pp']>1e-10 for r in delta), negative=sum(r['delta_pp']<-1e-10 for r in delta), neutral=sum(abs(r['delta_pp'])<=1e-10 for r in delta), rows=delta))
        for c in arms['Frozen']:
            assert arms['Frozen'][c]['metrics']['m_vIoU'] == next(r for r in native if r['dataset']==ds and r['case_id']==c and r['arm']=='F')['metrics']['m_vIoU']
    epoch = int(time.time())
    output = dict(status='pass', time=epoch, scope='128 exposed development sources; saved scores only', table=table, paired=pairs, summary_max_error=max(native_error,old_error), source_hashes=hashes, native_actual_fit_invocations=student['actual_fit_invocations'], native_full_readout_calls=student['full_readout_calls'], native_mode_optimizer_steps=student['mode_optimizer_steps'], new_model_inference=False, new_raw_GT_read=False, new_scoring=False, bootstrap_executed=False, method_promoted=False, study_FINAL_complete=False, limitation='Native vs QA reference clocks change training support; support-specific sIoU is not a matched-support propagation comparison; saved witness audit does not repeat full backward')
    output['reader_sha256'] = sha(Path(__file__))
    target = BASE / 'analysis' / f'NATIVE_FINAL_READBACK_{epoch}.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(target.relative_to(ROOT))
    print(json.dumps(table,ensure_ascii=False,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in pairs],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
