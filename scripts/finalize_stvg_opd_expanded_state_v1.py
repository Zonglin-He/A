"""Direct paired state endpoints over the predeclared 905 non-dev parents."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.stvg_opd_expanded_state_common_v1 import *
from scripts.finalize_stvg_opd_design_ablation_v1 import statistics
from scripts.review_stvg_opd_supplement_v1 import independent_stats, compare_stats


def run():
    verify(); c=read(BASE/'CPU_SCORING_COMPLETION.json')
    assert c['counts']['complete_fit_math']==5430
    results={}; table=[]; comparisons=0; bitwise_coordinates=0
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    for name,s in stages().items():
        path=PUB/name/'ROWS.jsonl.gz'
        assert sha(path)==c['outputs'][name]['rows_sha256']
        with gzip.open(path,'rt') as f: rows=[json.loads(x) for x in f]
        assert len(rows)==s['expected_new_fits']
        assert len({r['source_id'] for r in rows})==s['parent_count']
        cells={(r['arm'],r['order'],r['query_ordinal']):r for r in rows}
        assert len(cells)==len(rows)
        paired=[]
        for r in rows:
            if r['arm']!='full':continue
            pair=dict(r)
            for a in ['alpha0','alpha1']:
                control=cells[a,r['order'],r['query_ordinal']]
                assert control['source_id']==r['source_id'] and control['arrival']==r['arrival']
                assert control['Frozen_v']==r['Frozen_v'] and control['After_t']==r['After_t']
                for m in ['v','s','t']:
                    pair['Full_minus_'+a+'_'+m]=r['After_'+m]-control['After_'+m]
                    comparisons+=1
            paired.append(pair)
        fields=['After_v','Before_v','Frozen_v','Full_minus_alpha0_v','Full_minus_alpha1_v']
        pooled=statistics(paired,fields)
        by_order={o:statistics([r for r in paired if r['order']==o],fields) for o in s['orders']}
        # Independent scalar mean from matching per-query endpoints, no attribution substitution.
        for group,stat in [(paired,pooled)]+[([r for r in paired if r['order']==o],by_order[o]) for o in s['orders']]:
            comparisons+=compare_stats(stat,independent_stats(group,fields,balanced=True))
            for field in fields:
                reference=math.fsum(r[field] for r in group)/len(group)
                assert abs(reference-stat['metrics'][field]['mean'])<2e-12
                comparisons+=1
        for o,seq in s['orders'].items():
            for a in ARMS:
                for at in range(2):
                    p=BASE/'stages'/name/'clean'/o/a/f'{at:05}.pt'
                    q=BASE/'qualification'/name/'clean'/o/a/f'{at:05}.pt'
                    z=load(p);qualified=load(q)
                    assert z['query_ordinal']==qualified['query_ordinal']==seq[at]
                    check=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                    equal(scientific(z['fit']),scientific(qualified['fit']),check)
                    bitwise_coordinates+=check['tensor_coordinates']
        absolute={}
        for a in ARMS:
            part=[r for r in rows if r['arm']==a]
            absolute[a]=statistics(part,['After_v','Before_v','Frozen_v'])
            comparisons+=compare_stats(absolute[a],independent_stats(part,['After_v','Before_v','Frozen_v'],balanced=True))
        results[name]=dict(pooled=pooled,per_order=by_order,parent_count=s['parent_count'],
            arm_absolute=absolute,
            rows_sha256=sha(path),all_negative_results_retained=True)
        for o,stat in [('pooled',pooled)]+list(by_order.items()):
            for a in ['alpha0','alpha1']:
                m=stat['metrics']['Full_minus_'+a+'_v']
                table.append(dict(dataset=s['dataset'],order=o,contrast='Full - '+a,
                    mean_pp=m['mean']*100,ci95_pp=[v*100 for v in m['ci95']],
                    parent_count=stat['sources'],negative_parents=m['negative_parent_count']))
    write(PUB/'DIRECT_CONTRASTS.json',dict(status='complete_post_global_seal',statistics=results,table=table,
        bootstrap=10000,seed=20261011,unadjusted_CI=True,nondevelopment_parents=905,
        historically_evaluated_not_fresh=True,GT_used_to_select_cohort=False))
    lines=['# Expanded direct two-timescale contrasts','','All 5430 new fits sealed before GT. Original non-development balanced representatives: 205 HC parents and 700 Vid parents. Two fixed orders; no parameter selection.','','| Target | Order | Paired contrast | Difference (pp) | Parent 95% CI |','|---|---|---|---:|---|']
    for r in table:
        lines.append(f"| {r['dataset']} | {r['order']} | {r['contrast']} | {r['mean_pp']:+.3f} | [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}] |")
    lines+=['','These are direct final-output contrasts, not Before-Frozen attribution. Both orders and all arms stay paired in parent resampling. Intervals are unadjusted, and the cohort has historical evaluation exposure. All negative tails are retained in the companion statistics.']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    for ax,(name,result) in zip(axes,results.items()):
        for j,a in enumerate(['alpha0','alpha1']):
            vals=[];lower=[];upper=[]
            for o in ['pooled','order1','order2']:
                stat=result['pooled'] if o=='pooled' else result['per_order'][o]
                m=stat['metrics']['Full_minus_'+a+'_v'];v=m['mean']*100
                vals.append(v);lower.append(v-m['ci95'][0]*100);upper.append(m['ci95'][1]*100-v)
            x=np.arange(3)+(j-.5)*.14
            ax.errorbar(x,vals,yerr=[lower,upper],fmt='o',capsize=3,label='Full - '+('reset' if a=='alpha0' else 'full carry'))
        ax.axhline(0,color='#6b7280',linewidth=.8)
        ax.set_xticks(range(3),['Pooled','Order 1','Order 2'])
        ax.set_title(('HC-STVG-v2' if 'hc2' in name else 'VidSTG')+f" ({result['parent_count']} parents)")
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Paired vIoU difference (pp)');axes[1].legend(frameon=False,fontsize=9)
    fig.tight_layout();out=ROOT/'output/pdf/stvg_opd_claim_strengthening';out.mkdir(parents=True,exist_ok=True)
    pdf=out/'expanded_state_direct_contrasts.pdf';fig.savefig(pdf);fig.savefig(pdf.with_suffix('.png'),dpi=180);plt.close(fig)
    write(BASE/'ROOT_COMPLETE_READBACK.json',dict(status='pass',scope='all postseal independent CPU math/state/input/dense plus actual direct matched endpoints and qualified-formal pairs',
        counts=c['counts'],new_formal_fits=5430,parents=905,paired_endpoint_scalar_comparisons=comparisons,
        qualified_formal_pairs=24,qualified_formal_tensor_coordinates=bitwise_coordinates,
        GT_read='only_post_global_seal',results_sha256=sha(PUB/'DIRECT_CONTRASTS.json'),time=time.time()))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_root_PDF_pixel_view_and_publication',
        experiment_complete=False,new_formal_fits=5430,root_readback_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'),
        pdf_sha256=sha(pdf),time=time.time()))


if __name__=='__main__':run()
