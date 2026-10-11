"""Four real stream arms across three independent registered parent permutations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_mechanism_LN_order_common_v1 import *
from scripts.finalize_stvg_opd_design_ablation_v1 import statistics

def getrows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]

def run():
    verify();complete=read(BASE/'CPU_SCORING_COMPLETION.json');assert complete['counts']['complete_fit_math']==768
    prior=read(PREVIOUS/'ROOT_ALL_FITS_AND_STATISTICS.json');assert prior['status']=='pass'
    allstats={};table=[];paired_input_checks=0
    oldpub=ROOT/'results/stvg_opd_supplement/2026-10-11/orders'
    for name,s in stages().items():
        oldname=name.replace('M_','O_');oldrows=getrows(oldpub/oldname/'ROWS.jsonl.gz');newrows=getrows(PUB/name/'ROWS.jsonl.gz')
        assert len(oldrows)==1152 and len(newrows)==384
        priorbarrier=read(PREVIOUS/'stages'/oldname/'PREDICTION_BARRIER.json');current=read(BASE/'stages'/name/'PREDICTION_BARRIER.json')
        cells={(r['order'],r['query_ordinal'],r['arm']):r for r in oldrows+newrows};assert len(cells)==1536
        pairs=[]
        for r in oldrows:
            if r['arm']!='full':continue
            pair=dict(r)
            for arm in ['alpha0','alpha1','LN_only']:
                cc=cells[(r['order'],r['query_ordinal'],arm)]
                assert r['arrival']==cc['arrival'] and r['source_id']==cc['source_id'] and r['Frozen_v']==cc['Frozen_v'] and r['After_t']==cc['After_t']
                for metric in ['v','s','t']:pair['Full_minus_'+arm+'_'+metric]=r['After_'+metric]-cc['After_'+metric]
            key=f"clean/{r['order']}/full/{r['arrival']:05}";a=priorbarrier['logical_records'][key]
            b=current['logical_records'][f"clean/{r['order']}/LN_only/{r['arrival']:05}"]
            assert a['input']['sha256']==b['input']['sha256'] and a['query_ordinal']==b['query_ordinal'];paired_input_checks+=1
            pairs.append(pair)
        fields=['After_v','Before_v','Frozen_v','delta_inherited_v','delta_current_v','Full_minus_alpha0_v','Full_minus_alpha1_v','Full_minus_LN_only_v']
        allstats[name]=dict(pooled=statistics(pairs,fields),per_order={o:statistics([r for r in pairs if r['order']==o],fields) for o in s['orders']},
           first_parent_per_order=[{f:r[f] for f in fields} for r in pairs if r['arrival']==0],
           old_rows_sha256=sha(oldpub/oldname/'ROWS.jsonl.gz'),new_rows_sha256=sha(PUB/name/'ROWS.jsonl.gz'),
           arms={a:statistics([r for r in cells.values() if r['arm']==a],['Before_v','After_v','Frozen_v','delta_current_v']) for a in ['full','alpha0','alpha1','LN_only']})
        for o,item in allstats[name]['per_order'].items():
            for a in ['alpha0','alpha1','LN_only']:
                met=item['metrics']['Full_minus_'+a+'_v'];table.append(dict(stage=name,order=o,contrast='Full - '+a,mean_pp=met['mean']*100,ci95_pp=[x*100 for x in met['ci95']]))
    assert paired_input_checks==768
    write(PUB/'FOUR_ARM_ORDER_STATISTICS.json',dict(status='postseal_complete_four_actual_arms_three_real_permutations',statistics=allstats,per_order_table=table,
       bootstrap=10000,seed=20261011,unadjusted_descriptive_CI=True,reused_formal_fits=2304,new_formal_fits=768,matched_parent_order_inputs=paired_input_checks,
       historically_exposed_parents=True,Before_minus_Frozen_not_a_causal_carry_contrast=True,time=time.time()))
    lines=['# Direct state transfer comparisons','','Three actual full stream permutations. Every parent has one query. Original hyperparameters are fixed. Existing Full/reset/full-carry scientific streams are reused exactly; only 768 LN-only fits were added.','','| Target | Order | Contrast | Difference (pp) | Parent 95% CI |','|---|---|---|---:|---|']
    for r in table:lines.append(f"| {r['stage']} | {r['order']} | {r['contrast']} | {r['mean_pp']:+.3f} | [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}] |")
    lines += ['', 'These are actual sequence reruns, not bootstrap-created orders. Bootstrap is the uncertainty calculation after paired parent averaging. All source histories, negative tails and first-parent readouts remain in the companion. Before-Frozen is descriptive and cannot substitute for the four-arm contrasts. Actual root figure view, independent statistics readback, remote publication and archive remain required.']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_root_independent_statistics_PDF_view_publication_archive',new_formal_fits=768,reused_formal_fits=2304,
       counts=complete['counts'],paired_input_checks=paired_input_checks,experiment_complete=False,CPU_only=True,
       outputs={str(p.relative_to(ROOT)):sha(p) for p in PUB.rglob('*') if p.is_file()},time=time.time()))

if __name__=='__main__':run()
