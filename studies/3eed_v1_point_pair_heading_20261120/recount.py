"""Independent standard-library recount of all public IoUs and every threshold transition."""
import csv
import gzip
import json
from pathlib import Path
from statistics import fmean


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    root=Path(__file__).resolve().parent
    out=root/'results'
    methods=json.loads((root/'PROTOCOL.json').read_text())['arms']+['gt_yaw_diagnostic']
    report={}
    for stage,total in [('source',384),('pilot',3072),('ordinary',18620)]:
        groups={}
        ids=set()
        for filename in sorted(out.glob(stage+'_*_rows.csv.gz')):
            direction=filename.name[len(stage)+1:-len('_rows.csv.gz')]
            with gzip.open(filename,'rt') as stream:
                rows=list(csv.DictReader(stream))
            keys={r['row'] for r in rows}
            assert len(keys)==len(rows) and keys.isdisjoint(ids)
            ids.update(keys)
            groups[direction]=rows
        assert len(ids)==total
        metrics={}
        for d,rows in groups.items():
            for arm in methods:
                iou=[float(r[arm+'_iou']) for r in rows]
                assert all(0<=v<=1 for v in iou)
                metrics[d,arm]=[100*fmean(v>t for v in iou) for t in [.25,.50]]+[100*fmean(iou)]
        for arm in methods:
            metrics['macro',arm]=[fmean(metrics[d,arm][i] for d in groups) for i in range(3)]
        errors=[]
        for r in read_csv(out/f'{stage}_metrics.csv'):
            errors.extend(abs(v-float(r[k])) for v,k in zip(metrics[r['direction'],r['arm']],('acc25','acc50','miou')))
        assert max(errors)<1e-10
        groups['macro']=sum(groups.values(),[])
        for r in read_csv(out/f'{stage}_paired.csv'):
            pairs=[(float(v[r['arm']+'_iou']),float(v[r['baseline']+'_iou'])) for v in groups[r['direction']]]
            for t,label in [(.25,'25'),(.5,'50')]:
                assert sum(a>t>=b for a,b in pairs)==int(r['rescue'+label])
                assert sum(b>t>=a for a,b in pairs)==int(r['harm'+label])
        for r in read_csv(out/f'{stage}_coverage.csv'):
            rows=groups[r['direction']]
            assert sum(int(v['eligible']) for v in rows)==int(r['eligible'])
            assert sum(float(v[r['arm']+'_iou'])>.5>=float(v['prior_pool_best']) for v in rows)==int(r['rescued_outside_pool50'])
        report[stage]=dict(n=total,n_directions=len(groups)-1,metric_max_error=max(errors),
                           pairs_recounted=True,pool_rescues_recounted=True)
    (out/'recount.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    main()
