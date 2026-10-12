"""Independent fixed historical scalar analyses; no model or fresh GT access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import gzip
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,status,sha
OLD=ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08'
BASE=ROOT/'artifacts/stvg_opd_supplement_v1/saved_evidence'
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11/saved_evidence'
FIG=ROOT/'output/pdf/stvg_opd_supplement'

def readrows(p):
    with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]

def statistic(rows,field):
    group=collections.defaultdict(list)
    for r in rows:
        if r[field] is not None:group[r['source_id']].append(r[field])
    ids=sorted(group);vector=np.array([np.mean(group[i]) for i in ids]);rng=np.random.default_rng(20261011)
    boots=np.concatenate([vector[rng.integers(0,len(vector),(100,len(vector)))].mean(1) for _ in range(100)])
    return dict(parents=len(ids),eligible_cells=sum(len(v) for v in group.values()),missing_cells=sum(r[field] is None for r in rows),mean=float(vector.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
        positive_parents=int((vector>0).sum()),negative_parents=int((vector<0).sum()),parent_values=[dict(source_id=i,value=float(v)) for i,v in zip(ids,vector)])

def run():
    assert read(ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2/FINAL_COMPLETION.json')['paper_suite_complete']
    assert read(ROOT/'artifacts/stvg_opd_design_ablation_v1/FINAL_COMPLETION.json')['status']=='complete'
    BASE.mkdir(parents=True,exist_ok=True);PUB.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    specs=[('Cross VidSTG','P3_vidstg_cross_clean','clean'),('Same-domain clean VidSTG','P4_vidstg','clean'),('Cross HC-STVG-v2','P3_hc2_cross_clean','clean')]
    records={};bindings={}
    for label,stage,condition in specs:
        p=OLD/stage/'ROWS.jsonl.gz';part=[r for r in readrows(p) if r['arm']=='on_policy' and r['condition']==condition]
        expected=732 if stage=='P4_vidstg' else 128
        assert len({r['source_id'] for r in part})==expected
        records[label]=dict(unobserved=statistic(part,'unobserved_iou_delta'),observed=statistic(part,'observed_iou_delta'),rows=len(part),source_sha256=sha(p),
            Before_to_After_only=True,conditional_on_fixed_stream_histories=True)
        bindings[str(p.relative_to(ROOT))]=sha(p)
    from scripts.finalize_stvg_opd_design_ablation_v1 import statistics
    p=OLD/'P4_hc2/ROWS.jsonl.gz';allrows=readrows(p);conditions=sorted({r['condition'] for r in allrows if r['condition']!='clean'});twofive=[c for c in conditions if c.endswith('_2.5') or c.endswith('_2p5') or c.endswith('_2_5')]
    if not twofive:twofive=[c for c in conditions if '2.5' in c or '2p5' in c]
    assert len(twofive)==5,conditions
    selected=[r for r in allrows if r['arm']=='on_policy' and r['condition'] in twofive]
    table=statistics(selected,['After_v']);value=table['metrics']['After_v']['mean']*100
    assert f'{value:.3f}'=='34.229'
    write(PUB/'TABLE_ROUNDING_CORRECTION.json',dict(status='verified_from_original_rows',table='Table2',dataset='HC-STVG-v2',conditions=twofive,parents=237,logical_rows=len(selected),original_percent_display='34.230',correct_percent_display=f'{value:.3f}',exact_parent_macro_percent=value,old_rows_sha256=sha(p),old_data_unchanged=True))
    write(PUB/'UNOBSERVED_FRAME_EFFECTS.json',dict(status='independently_recomputed_saved_rows',records=records,bootstrap=10000,seed=20261011,
        comparison='within-query After-Before GT frame IoU; GT frames outside admitted feedback support',missing_is_not_zero=True,historical_cohorts=True,unadjusted_CI=True,no_new_model_calls=True,no_fresh_GT_scoring=True,time=time.time()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.1,3.3),layout='constrained');colors=['#245f86','#6a777f','#b07946']
    for y,(label,z),color in zip([2,1,0],records.items(),colors):
        s=z['unobserved'];m=s['mean']*100;ci=np.array(s['ci95'])*100
        ax.errorbar(m,y,xerr=[[m-ci[0]],[ci[1]-m]],fmt='o',ms=7,color=color,capsize=4,lw=1.6)
        ax.text(max(ci[1],0)+.2,y,f'{m:+.2f}',va='center',color=color,fontsize=11)
    ax.axvline(0,ls='--',color='#888888',lw=1);ax.set_yticks([2,1,0],[f'{k}\n{v["unobserved"]["parents"]} parent videos' for k,v in records.items()])
    ax.set_xlabel('After - Before IoU on unobserved GT frames (pp)');ax.set_xlim(-5.5,8.5);ax.set_ylim(-.6,2.6);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.savefig(FIG/'unobserved_frame_effects.pdf');fig.savefig(PUB/'unobserved_frame_effects.png',dpi=220);plt.close(fig)
    import shutil
    shutil.copy2(FIG/'unobserved_frame_effects.pdf',PUB/'unobserved_frame_effects.pdf')
    # Formula/interface figure: frozen encoder and heads, train only query+selected LN.
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(12,4.5));ax.set_xlim(0,12);ax.set_ylim(0,4.5);ax.axis('off')
    boxes=[(.15,2.55,2.1,1.2,'Native student\nFrozen encoder and heads\nNative WHEN'),(2.85,2.55,2.1,1.2,'Event-scoped Uniform4\nFrozen DINO Top1\nReliable feedback'),(5.55,2.55,2.9,1.2,'On-policy adaptation\nQuery residual: 256\nSelected LN: 1536'),(9.05,2.55,2.75,1.2,'Final trajectory boxes\nOriginal temporal interval\nLast configured round')]
    for x,y,w,h,label in boxes:
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.08',facecolor='#f5f7f8',edgecolor='#667984',lw=1.1);ax.add_patch(patch);ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    for x0,x1 in [(2.3,2.8),(5,5.5),(8.5,9)]:ax.add_patch(FancyArrowPatch((x0,3.15),(x1,3.15),arrowstyle='-|>',mutation_scale=13,color='#526a78',lw=1.4))
    ax.text(1.4,1.2,'Fast state\nQuery residual = 0\nAdam reset per query',ha='center',va='center',fontsize=12,color='#245f86')
    ax.text(6.75,1.3,'Slow state: retain prior LN and attenuate the fitted displacement',ha='center',fontsize=12,color='#245f86')
    ax.text(6.75,.7,r'$\lambda_{n+1}^{G}=\lambda_n^{G}+\alpha(\lambda_n^{\star}-\lambda_n^{G})$',ha='center',fontsize=17,color='#245f86')
    ax.add_patch(FancyArrowPatch((7,2.5),(7,1.7),arrowstyle='-|>',mutation_scale=13,color='#526a78',lw=1.4))
    ax.text(10.6,1.05,'Next query',ha='center',fontsize=12);ax.add_patch(FancyArrowPatch((9.4,1.05),(10.05,1.05),arrowstyle='-|>',mutation_scale=13,color='#526a78',lw=1.4))
    fig.tight_layout();fig.savefig(FIG/'method_state_interfaces.pdf');fig.savefig(PUB/'method_state_interfaces.png',dpi=220);plt.close(fig);shutil.copy2(FIG/'method_state_interfaces.pdf',PUB/'method_state_interfaces.pdf')
    write(PUB/'METHOD_DIAGRAM_CONTRACT.json',dict(status='matches_locked_implementation',formula='lambda_next_global = lambda_initial_global + alpha*(lambda_final_query-lambda_initial_global)',active_query_dimensions=256,active_LN_dimensions=1536,encoder_and_native_temporal_readout_frozen=True,no_full_spatial_decoder_training=True,query_and_Adam_reset_each_query=True,old_paper_files_not_edited=True,GT_online=False))
    write(BASE/'COMPLETION.json',dict(status='pending_actual_root_view',pins=bindings,outputs={str(p.relative_to(ROOT)):sha(p) for p in PUB.iterdir()},new_model_calls=0,fresh_GT_scoring=False,time=time.time()))
    print('SAVED_UNOBSERVED',[(k,round(v['unobserved']['mean']*100,6),[round(x*100,6) for x in v['unobserved']['ci95']]) for k,v in records.items()],flush=True)

if __name__=='__main__':run()
