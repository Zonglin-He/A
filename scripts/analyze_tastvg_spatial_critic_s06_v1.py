"""Post-reward-seal critic reliability against existing sealed candidate metrics."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_spatial_critic_s06_v1 import OUT,NATIVE,verify
from vg_tta.tastvg_spatial_critic_s06_v1 import ALL_PAIRS,ANTITHETIC,pair_record
from scripts.analyze_tastvg_corruption_c0c1_v1 import stats

def macro(rows,key):
    by=collections.defaultdict(list)
    for r in rows:
        if r[key] is not None:by[r['parent']].append(r[key])
    return stats([np.mean(by[p]) for p in sorted(by)])

def pair_summary(pairs):
    by=collections.defaultdict(list)
    for p in pairs:by[p['parent'],p['condition']].append(p)
    rows=[]
    for (parent,condition),seq in sorted(by.items()):
        r=dict(parent=parent,condition=condition)
        for k in ['accuracy','decisive_accuracy','sign_agreement']:
            a=[x[k] for x in seq if x[k] is not None];r[k]=float(np.mean(a)) if a else None
        rows.append(r)
    eligible=[p for p in pairs if not p['GT_tie']]
    return dict(pairs=len(pairs),GT_ties=sum(p['GT_tie'] for p in pairs),expert_ties=sum(p['expert_tie'] for p in eligible),decisive_pairs=sum(not p['expert_tie'] for p in eligible),critic_cells=len(by),source_count=len({p['parent'] for p in pairs}),accuracy=macro(rows,'accuracy'),decisive_accuracy=macro(rows,'decisive_accuracy'),literal_sign_agreement=macro(rows,'sign_agreement'),pooled_accuracy=float(np.mean([p['accuracy'] for p in eligible])) if eligible else None)

def run():
    lock=verify();bar=read(OUT/'REWARD_BARRIER.json')
    for n,h in bar['files'].items():assert sha(OUT/n)==h
    rewards=read(OUT/'REWARDS.json');cuts=read(OUT/'MARGIN_CUTS.json');source=read(NATIVE/'ROWS.json');old={(r['parent'],r['condition']):r for r in source}
    write(OUT/'METRIC_EXPOSURE.json',dict(existing_metrics_only=True,raw_GT_read=False,existing_exposed_sources=16,source_rows_sha256=sha(NATIVE/'ROWS.json'),reward_barrier_sha256=sha(OUT/'REWARD_BARRIER.json'),time=time.time()))
    pairs=[];rows=[]
    for r in rewards:
        g=old[r['parent'],r['condition']];gt=np.array([m['sIoU'] for m in g['candidate_metrics']]);sel=r['selected'];rank=r['rewards'];group='clean' if r['condition']=='clean' else 'corruption'
        rows.append(dict(parent=r['parent'],condition=r['condition'],rewards=rank,gt_sIoU=gt.tolist(),selected=sel,valid_expert_frames=r['valid_expert_frames'],critic_available=rank is not None,native_s=float(gt[0]),selected_s=float(gt[sel]),selected_gain=float(gt[sel]-gt[0]),uniform_expected_gain=float(gt.mean()-gt[0]),oracle_gain=float(gt.max()-gt[0]),oracle_regret=float(gt.max()-gt[sel]),top1_GT_best=float(gt[sel]>=gt.max()-1e-12) if rank is not None else None))
        if rank is None:continue
        for i,j in ALL_PAIRS:
            rec=dict(parent=r['parent'],condition=r['condition'],i=i,j=j,antithetic=(i,j) in ANTITHETIC,**pair_record(rank[i]-rank[j],gt[i]-gt[j]));rec['all_bin']=['low','mid','high'][int(np.searchsorted(cuts[group]['all'],rec['margin'],side='right'))];rec['antithetic_bin']=['low','mid','high'][int(np.searchsorted(cuts[group]['antithetic'],rec['margin'],side='right'))] if rec['antithetic'] else None;pairs.append(rec)
    summary={}
    for group in ['corruption']+lock['conditions']:
        take=lambda r:r['condition']!='clean' if group=='corruption' else r['condition']==group
        rr=[r for r in rows if take(r)];pp=[p for p in pairs if take(p)];anti=[p for p in pp if p['antithetic']]
        summary[group]=dict(cells=len(rr),sources=16,critic_valid_cells=sum(r['critic_available'] for r in rr),critic_valid_sources=len({r['parent'] for r in rr if r['critic_available']}),all_pairs=pair_summary(pp),antithetic=pair_summary(anti),all_margin={b:pair_summary([p for p in pp if p['all_bin']==b]) for b in ['low','mid','high']},antithetic_margin={b:pair_summary([p for p in anti if p['antithetic_bin']==b]) for b in ['low','mid','high']},directions={str(k+1):pair_summary([p for p in anti if (p['i'],p['j'])==pair]) for k,pair in enumerate(ANTITHETIC)},metrics={k:macro(rr,k) for k in ['native_s','selected_s','selected_gain','uniform_expected_gain','oracle_gain','oracle_regret','top1_GT_best']},source_rows=[])
        for p in sorted({r['parent'] for r in rr}):
            seq=[r for r in rr if r['parent']==p];summary[group]['source_rows'].append(dict(parent=p,selected_gain=float(np.mean([r['selected_gain'] for r in seq])),oracle_gain=float(np.mean([r['oracle_gain'] for r in seq])),critic_valid_cells=sum(r['critic_available'] for r in seq),all_accuracy=pair_summary([z for z in pp if z['parent']==p])['accuracy']['mean'],antithetic_accuracy=pair_summary([z for z in anti if z['parent']==p])['accuracy']['mean']))
    write(OUT/'ROWS.json',rows);write(OUT/'PAIRS.json',pairs);write(OUT/'SUMMARY.json',summary)
    for g in ['corruption','clean']:print(g,{k:summary[g][k]['accuracy'] for k in ['all_pairs','antithetic']},'selected_gain',summary[g]['metrics']['selected_gain'])

if __name__=='__main__':run()
