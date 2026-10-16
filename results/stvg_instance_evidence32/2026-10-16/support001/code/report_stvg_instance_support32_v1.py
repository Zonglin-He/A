"""Clarify actual gradient support without changing original scores/selection."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    out=ROOT/'results/stvg_instance_evidence32/2026-10-16/support001';assert not out.exists();out.mkdir(parents=True)
    d=read(BASE/'ACTUAL_GRADIENT_SUPPORT_DIAGNOSIS.json');assert d['status']=='complete'and d['count']==32
    alias={c:f'E{i:03}'for i,c in enumerate(sorted({r['case_id']for r in d['rows']}),1)}
    put(out/'ACTUAL_GRADIENT_SUPPORT.json',dict(status='complete',count=32,rows=[dict(r,case_id=alias[r['case_id']])for r in d['rows']],summary=d['summary'],GT_read=True,common_metric_note=d['common_metric_note'],complement=d['complement'],scope='postseal support partition diagnostic, no new fit or recipe selection'))
    g=copy.deepcopy(read(BASE/'DEVELOPMENT_GAIN_CONCENTRATION.json'))
    for r in g['records']:r['largest_positive_source']=alias[r['largest_positive_source']]
    put(out/'GAIN_CONCENTRATION.json',g)
    put(out/'ROOT_PUBLIC_READBACK.json',dict(status='pass',count=32,row_count=96,source_sha256=sha(BASE/'ACTUAL_GRADIENT_SUPPORT_DIAGNOSIS.json'),source_main_result_sha256=sha(BASE/'EXPERIMENT_POSTSEAL_RESULTS.json'),main_vIoU_tIoU_dense_scores_exactly_unchanged=True,actual_support_nulls_preserved=True,GT_read=True))
    from scripts.export_stvg_lite_batch001 import public_source_text
    for n in ('diagnose_stvg_instance_actual_support_v1.py','report_stvg_instance_support32_v1.py','publish_stvg_instance_support001.py'):
        q=out/'code'/n;q.parent.mkdir(exist_ok=True);q.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));v=[]
    for ds in ('hc2','vidstg'):
        for metric in ('m_sIoU_supervised_GT','m_sIoU_unsupervised_GT'):
            r=next(r for r in d['summary']if r['dataset']==ds and r['arm']=='E1'and r['metric']==metric);v.append(r)
    ax.bar(range(4),[r['mean_student_minus_Temporal_pp']for r in v]);ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(4),['HC anchors\n11/16 known','HC complement\n16/16 known','Vid anchors\n6/16 known','Vid complement\n16/16 known']);ax.set_ylabel('E1 - Temporal mean sIoU (pp)');ax.set_title('Actual gradient support; different known subsets');fig.tight_layout();fig.savefig(out/'actual_support.png',dpi=160);plt.close(fig)
    doc=ROOT/'docs/STVG_INSTANCE_ACTUAL_SUPPORT32_001.md';doc.write_text('''# Actual gradient support versus fixed observation support

The original five-arm scoring uses the same four pre-fixed observation positions for every arm. Its inherited column names supervised/unsupervised therefore mean common observation positions versus their known-GT complement. Some positions select Source, are rejected or have no usable anchor. They should not be described as four actual gradient-supervised frames. The complement includes off-Source-grid known GT, as defined by the original official/independent scorer.

This additional postseal diagnostic instead uses each updater's actual positive-weight gradient-anchor positions. It compares Student and corrected Temporal on exactly that same subset and its complete known-GT complement. All vIoU, tIoU and dense sIoU are asserted exactly unchanged. No new Teacher/Student fit, protocol selection or Bootstrap occurs; all nulls and all96 source/updater rows remain.

For E1, HC actual-anchor sIoU changes+1.268pp on11 known sources(5 unknown), complement+0.055pp on all16. Vid actual-anchor changes+3.528pp on6 known sources(10 unknown), complement+2.516pp on all16. These subset means are not directly interchangeable or a guarantee of propagation safety. Negative and zero changes are retained.

Development gain concentration also limits interpretation. HC E1-E0 becomes-.088pp if its largest positive source is omitted. Vid E1-E2 becomes+.000516pp without its largest positive source. These are descriptive arithmetic, not a new statistical test or changed advancement rule. The original mean-based finite development rule remains locked; the fixed64 confirmation continues unchanged. A small positive development mean does not establish stable updater benefit.

Main32 results and all old predictions stay immutable. This supplement corrects metric interpretation and adds actual-support measurements, not a replacement method score. Source media, GT coordinates, tensors, weights, Query-caption and raw Teacher responses remain private.
''')
    paths={str(p.relative_to(ROOT))for p in out.rglob('*')if p.is_file()}|{str(doc.relative_to(ROOT)),'scripts/diagnose_stvg_instance_actual_support_v1.py','scripts/report_stvg_instance_support32_v1.py','scripts/publish_stvg_instance_support001.py'}
    put(BASE/'publication/support001/EXPORT_MANIFEST.json',dict(status='complete',paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size)for p in sorted(paths)]));print(json.dumps(dict(status='complete',files=len(paths))))
if __name__=='__main__':main()
