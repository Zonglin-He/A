"""Close only the measured eight-source native-instance development scope."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_native_instance_spatial_opd_v1'
OUT = ROOT / 'results/stvg_native_instance/2026-10-14/closing'

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(d, f, indent=2, allow_nan=False)
        f.write('\n')

def main():
    required = ['ROOT_STUDENT_MATH_READBACK.json', 'ROOT_COST_READBACK.json',
                'ROOT_POSTSEAL_READBACK.json', 'ROOT_INDEPENDENT_GEOMETRY_READBACK.json',
                'ROOT_POOL_COVERAGE_READBACK.json', 'ROOT_VISUAL_REVIEW.json',
                'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json', 'ROOT_FINAL_EXECUTION_READBACK.json',
                'PUBLIC_SCALAR_AUDIT.json', 'ROOT_ANONYMOUS_SCALAR_READBACK.json',
                'FINAL_GITHUB_RECEIPT.json', 'ARCHIVE_VERIFICATION_RECEIPT.json']
    for n in required:
        assert read(BASE/n)['status'] in ('pass', 'complete'), n
    pub = read(BASE/'FINAL_GITHUB_RECEIPT.json')
    assert pub['all_remote_bytes_SHA256_Gitblob_mode_tree_exact'] is True
    refs = read(BASE/'PUBLIC_REF_READBACK.json')
    assert refs['status'] == 'pass' and len(refs['refs']) == 2
    assert set(refs['refs'].values()) == {pub['remote_commit']}
    assert sha(ROOT/'methods/CURRENT_METHOD.json') == '146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4'
    decision = read(BASE/'TECHNICAL_DECISION.json')
    assert decision['status'] == 'finite_development_complete_not_ready_for_independent_expansion'
    stamp = datetime.now(timezone.utc).isoformat()
    closing = dict(status='complete', scope='eight exposed sources; native instance evidence and matched sparse spatial update',
        time=stamp, case_count=8, mask_frame_files=5694, fit_paths=15, actual_optimizer_steps=65,
        final_student_redecodes=15, score_rows=72, summary_cells=90, paired_contrasts=80,
        private_contacts_viewed=8, report_pdf_pages_viewed=1,
        complete_current_finite_development=True, independent_confirmation_launched=False,
        full_data_expansion_launched=False, production_promoted=False, new_GT_oracle_fits=0,
        root_receipt_sha256={n:sha(BASE/n) for n in required},
        prediction_seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        public_measurement_commit=pub['remote_commit'], public_measurement_tree=pub['remote_tree'],
        positive_negative_neutral_unknown_preserved=True,
        scientific_conclusion='mean T+IoU gains do not establish stable cross-source spatial benefit; weighting adds no demonstrated advantage',
        independent_autograd_or_model_replay=False)
    write(BASE/'ROOT_CLOSING.json', closing)
    write(BASE/'FINAL_COMPLETION.json', dict(status='complete', scope=closing['scope'], time=stamp,
        complete_current_finite_development=True, root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),
        method_validated_for_expansion=False, future_research_complete=False, CURRENT_changed=False))
    receipts = ['ROOT_CLOSING.json','FINAL_COMPLETION.json','PUBLIC_SCALAR_AUDIT.json',
                'ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',
                'ROOT_FINAL_EXECUTION_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_GITHUB_RECEIPT.json',
                'PUBLIC_REF_READBACK.json','TECHNICAL_DECISION.json','ARCHIVE_VERIFICATION_RECEIPT.json']
    stage = BASE/'closing_public_stage'
    rows = []
    sources = [(BASE/n, OUT/n) for n in receipts]
    sources += [(ROOT/n,ROOT/n) for n in ['docs/STVG_NATIVE_INSTANCE_DECISION.md',
                  'scripts/review_stvg_native_instance_public_v1.py','scripts/close_stvg_native_instance_v1.py']]
    for src, dst in sources:
        d = src.read_bytes()
        if src.suffix == '.json':
            assert b'/home/wwww/' not in d and b'query_text' not in d and b'raw_output' not in d
        if dst != src:
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open('xb') as f:f.write(d)
        rel = dst.relative_to(ROOT).as_posix()
        target=stage/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(d)
        rows.append(dict(path=rel, bytes=len(d), sha256=hashlib.sha256(d).hexdigest(),encoding='utf-8'))
    write(BASE/'CLOSING_PUBLIC_MANIFEST.json', rows)
    print(json.dumps(dict(status='closing_staged_not_pushed',files=len(rows),bytes=sum(r['bytes'] for r in rows))))

if __name__ == '__main__':
    main()
