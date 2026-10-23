"""Independent CPU-only repair of one recovered-screen report path.

The frozen predictor/scorer and prediction fingerprint remain unchanged.
Reverse-edit equality is required before executing the identical score math.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(Path(__file__).parent)]
from full import read,sha,atomic
D=R/'outputs/paper_review_suite_v1'


def main(stage):
    plan=read(D/'budget_score_adapter_v3_plan.json')
    assert plan['code_sha256']==sha(Path(__file__))
    source=Path(__file__).with_name('budget_v2.py')
    assert sha(source)==plan['frozen_score_source_sha256']
    text=source.read_text()
    before="read(P/'budget'/ds/'screen/report.json')"
    after="read(P/'budget_v2'/ds/'screen/report.json')"
    assert text.count(before)==1 and after not in text
    repaired=text.replace(before,after)
    assert repaired.replace(after,before)==text
    seal=D/('budget_v2_'+stage+'_sealed.json')
    assert not read(seal)['GT_read']
    assert not (D/('budget_v2_'+stage+'_score_receipt.json')).exists()
    namespace={'__file__':str(source),'__name__':'budget_recovered_CPU_score_only'}
    exec(compile(repaired,str(source),'exec'),namespace)
    namespace['score'](stage)
    receipt=D/('budget_v2_'+stage+'_score_receipt.json')
    proof=dict(status='complete_CPU_score',stage=stage,adapter_fingerprint=plan['fingerprint'],original_prediction_fingerprint=read(D/'budget_v2_plan.json')['fingerprint'],reverse_edit_byte_exact=True,only_change='screen qualification reads recovered budget_v2 reports',seal_sha256=sha(seal),score_receipt_sha256=sha(receipt),GPU_or_expert_forwards=0)
    atomic(D/('budget_v2_'+stage+'_score_adapter_receipt.json'),proof)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['screen','confirm'],required=True)
    main(parser.parse_args().stage)
