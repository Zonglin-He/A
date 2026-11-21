"""Post-seal edge-error decomposition and agreement with the known true box."""
import csv
from scipy.stats import rankdata
import numpy as np
from summarize import PUBLIC, PLATFORMS, parse

SIDES=('left','top','right','bottom')


def auc(rows):
    if not rows: return None
    bad=np.array([r['iou']<=.5 for r in rows])
    n=int(bad.sum())
    if n==0 or n==len(rows): return None
    ranks=rankdata([r['frozen_energy_px2'] for r in rows])
    return float((ranks[bad].sum()-n*(n+1)/2)/(n*(len(rows)-n)))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    with (PUBLIC/'rows.csv').open() as f:
        rows=[{k:parse(v) for k,v in r.items()} for r in csv.DictReader(f)]
    out=[]; side_rows=[]
    for t in ('all',*PLATFORMS):
        for branch in ('all','selected','fallback'):
            g=[r for r in rows if (t=='all' or r['target']==t) and
               (branch=='all' or r['selected']==(branch=='selected'))]
            for tag in ('truth','native','frozen'):
                available=[r for r in g if r[tag+'_pass'] is not None]
                failures=[r for r in available if not r[tag+'_pass']]
                # The sum of opposing signed overflows is w_Q - w_projection on that axis.
                # If larger than 2*tau, even translating the two-dimensional projection
                # cannot make the two corresponding sides pass with its current size.
                oversize=sum(r[tag+'_oversize_2d'] for r in failures)
                for s in SIDES:
                    sided=[r for r in available if r[tag+'_exceeds_'+s]]
                    side_rows.append(dict(platform=t,branch=branch,geometry=tag,side=s,observable=len(available),
                                          exceeded=len(sided)))
                for outcome in ('all','failed50','passed50'):
                    flagged=[r for r in available if r['frozen_pass'] is False and
                             (outcome=='all' or (r['iou']<=.5)==(outcome=='failed50'))]
                    diff=np.array([r['frozen_energy_px2']-r[tag+'_energy_px2'] for r in flagged])
                    out.append(dict(platform=t,branch=branch,geometry=tag,flagged_outcome=outcome,
                           observable=len(available),violations=len(failures),qwen_oversize_even_after_2d_shift=oversize,
                           two_dimensional_shift_could_cover=len(failures)-oversize,flagged_compared=len(diff),
                           label_reduces_hinge=int((diff>1e-8).sum()),label_equal_hinge=int((abs(diff)<=1e-8).sum()),
                           label_increases_hinge=int((diff< -1e-8).sum()),
                           frozen_hinge_auc_failure50=auc([r for r in g if r['frozen_pass'] is not None]),
                           frozen_hinge_auc_failure50_among_flagged=auc([r for r in g if r['frozen_pass'] is False])))
    write_csv(PUBLIC/'edge_diagnostics.csv',out)
    write_csv(PUBLIC/'sides.csv',side_rows)


if __name__=='__main__':
    main()
