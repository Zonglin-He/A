"""Direct paired new-order state contrasts; no re-selection of Ours parameters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import gzip
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.stvg_opd_order_supplement_common_v1 import *
from scripts.finalize_stvg_opd_design_ablation_v1 import statistics

def run():
    verify(); c=read(BASE/'CPU_SCORING_COMPLETION.json'); assert c['counts']['complete_fit_math']==2304
    allstats={}; progress={}; table=[]
    for name,s in stages().items():
        path=PUB/name/'ROWS.jsonl.gz'; assert sha(path)==c['outputs'][name]['rows_sha256']
        with gzip.open(path,'rt') as f: rows=[json.loads(x) for x in f]
        cells={(r['arm'],r['order'],r['query_ordinal']):r for r in rows}; assert len(cells)==1152
        paired=[]
        for r in rows:
            if r['arm']!='full': continue
            pair=dict(r)
            for a in ['alpha0','alpha1']:
                control=cells[(a,r['order'],r['query_ordinal'])]
                assert control['source_id']==r['source_id'] and control['arrival']==r['arrival'] and control['Frozen_v']==r['Frozen_v'] and control['After_t']==r['After_t']
                for m in ['v','s','t']:
                    pair['Full_minus_'+a+'_'+m]=r['After_'+m]-control['After_'+m]
            paired.append(pair)
        fields=['After_v','Before_v','Frozen_v','delta_current_v','delta_inherited_v','Full_minus_alpha0_v','Full_minus_alpha1_v']
        allstats[name]=dict(pooled=statistics(paired,fields),per_order={o:statistics([r for r in paired if r['order']==o],fields) for o in s['orders']},
            arm_absolute={a:statistics([r for r in rows if r['arm']==a],['After_v','Before_v','Frozen_v','After_s','After_t']) for a in ARMS},rows_sha256=sha(path))
        progress[name]={}
        for o in s['orders']:
            progress[name][o]={a:[dict(start=lo,end=lo+16,statistics=statistics([r for r in rows if r['arm']==a and r['order']==o and lo<=r['arrival']<lo+16],['Before_v','After_v','Frozen_v','delta_inherited_v','delta_current_v'])) for lo in range(0,128,16)] for a in ARMS}
            for a in ['alpha0','alpha1']:
                m=allstats[name]['per_order'][o]['metrics']['Full_minus_'+a+'_v']
                table.append(dict(stage=name,order=o,contrast='Full - '+a,mean_pp=m['mean']*100,ci95_pp=[v*100 for v in m['ci95']]))
    write(PUB/'ORDER_STATISTICS.json',dict(status='complete_postseal_direct_contrasts',statistics=allstats,per_order_table=table,bootstrap=10000,seed=20261011,unadjusted_CI=True,historically_exposed=True,time=time.time()))
    write(PUB/'STREAM_PROGRESS.json',dict(bin_size=16,curves=progress,ordered_progress_not_independent_replicates=True))
    lines=['# Additional fixed stream orders','','All 2304 new predictions globally sealed before GT. Same historical 128 parents per target; fixed method parameters. Three newly predeclared orders.','','| Target | Order | Direct contrast | Difference (pp) | Paired parent 95% CI |','|---|---|---|---:|---|']
    for r in table: lines.append(f"| {r['stage']} | {r['order']} | {r['contrast']} | {r['mean_pp']:+.3f} | [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}] |")
    lines+=['','Before–Frozen measures inherited-state attribution. It is not substituted for Full–alpha0. Bootstrap resamples parent sources jointly across paired arms and orders, not arrivals or optimization rounds. All negative tails and absolute/control scores are retained in the JSON companion. This historical-cohort order perturbation is not fresh-domain generalization. Actual root figure review and remote publication remain necessary.']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_root_visual_and_publication',new_formal_fits=2304,counts=c['counts'],experiment_complete=False,all_negative_results_retained=True,output_files={str(p.relative_to(ROOT)):sha(p) for p in PUB.rglob('*') if p.is_file()},time=time.time()))

if __name__=='__main__': run()
