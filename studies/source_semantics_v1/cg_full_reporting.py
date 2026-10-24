"""Explicit original-Source versus Source+MBR comparisons from sealed CG metrics.

This is a report-only adapter. Frozen inference/scoring files and internal aliases
are untouched; no model, video, annotation or GT rescoring is performed.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,argparse,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/paper_followup_v1')]
from common import report,read,sha,atomic
D=R/'outputs/cgstvg_full_v1';P=R/'results/cgstvg_full_v1'

def run(ds):
    expected=1902 if ds=='hc2' else 10303
    sp=D/f'{ds}_sealed.json';scorepath=D/f'{ds}_score_receipt.json'
    assert sp.exists()and scorepath.exists(),'Whole-domain seal and successful score required'
    seal=read(sp);score=read(scorepath)
    assert seal['queries']==expected==score['queries'] and score['whole_domain_sealed_before_GT']
    assert score['seal_sha256']==sha(sp) and score['checks']==5*expected
    original=read(D/ds/'metrics.json')['private_rows'];assert len(original)==expected
    rows=[{**r,'values':{'Source':r['values']['SourceNative'],
                        'SourcePlusMBR':r['values']['Source'],'Full':r['values']['Full']}}
          for r in original]
    folder=P/ds/'named_source_comparison'
    result=report(rows,['Source','SourcePlusMBR','Full'],
                  [('SourcePlusMBR','Source'),('Full','SourcePlusMBR'),('Full','Source')],folder)
    info=dict(status='complete_report_from_existing_scored_metrics',dataset=ds,
        display_Source='CG-STVG source model with official native MAP readout, under this local native420FP32/cross-domain protocol',
        display_SourcePlusMBR='Same CG source boxes/logits plus OUR MBR decoder',
        immutable_file_alias_mapping={'SourceNative':'Source','Source':'SourcePlusMBR','Full':'Full'},
        attribution='Full-Source contains readout and feedback effects. Full-SourcePlusMBR measures incremental feedback under matched MBR. Neither is hidden or substituted for the other.',
        seal_sha256=sha(sp),score_sha256=sha(scorepath),metrics_sha256=sha(D/ds/'metrics.json'),
        reporting_source_sha256=sha(Path(__file__)),new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0)
    atomic(folder/'semantics_receipt.json',info)
    print({'dataset':ds,'queries':expected,'means':result['means'],'paired':result['paired']['Full-Source']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['hc2','vidstg'],required=True)
    run(p.parse_args().dataset)
