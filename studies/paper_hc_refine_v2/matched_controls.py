"""Descriptive matched contrasts from already sealed/scored HC profiles; no GT access."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,read,sha,atomic
assert (D/'reporting_receipt.json').exists()
score=read(D/'score_receipt.json');assert score['checks']==23154
assert score['seals_sha256']==[sha(D/f'sealed_{i}.json')for i in range(2)]
rows=read(D/'metrics.json')['private_rows'];assert len(rows)==227
names=['u0.5_l0.25_b0.25','u0.0_l0.0_b0.25','u0.5_l0.25_b0.5','u0.0_l0.0_b0.5','u0.0_l0.25_b0.25','u0.5_l0.0_b0.25']
out=[]
for a in names:
    d=dict(profile=a,means={k:float(np.mean([r['values'][a][k]for r in rows if r['values'][a][k]is not None]))for k in ['t','v','unobserved_actual_sIoU']},severe={k:sum(r['values'][a][k]-r['values']['Source'][k]<-.05 for r in rows)for k in ['t','v']})
    d['severe']['union']=sum(any(r['values'][a][k]-r['values']['Source'][k]<-.05 for k in ['t','v'])for r in rows);out.append(d)
atomic(P/'descriptive_matched_controls.json',dict(scope='Descriptive contrasts within the predeclared100 profiles on all227historically exposed parents; not separate independent confirmation, no new scoring or selection.',profiles=out))
print('Six matched descriptive profiles exported; 0 model/GT scoring/selection')
