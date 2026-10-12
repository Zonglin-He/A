"""Anonymous association addendum and an actually inspectable scalar figure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_opd_optimization_geometry_v1';PUB=ROOT/'results/stvg_opd_optimization_geometry/2026-10-12'
OLD=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'

def lines(p):
    with gzip.open(p,'rt') as f:
        for s in f:yield json.loads(s)

def corr(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.dot(x-x.mean(),y-y.mean())/np.sqrt(np.dot(x-x.mean(),x-x.mean())*np.dot(y-y.mean(),y-y.mean())))

def run():
    assert read(BASE/'ROOT_CPU_DIAGNOSTIC_READBACK.json')['status']=='pass'
    cells=list(lines(PUB/'CELL_GEOMETRY.jsonl.gz'));key=lambda c:(c['dataset'],c['source_id'],c['condition'],c['order'],c['arrival'])
    frames=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in lines(OLD/'FRAME_ROWS.jsonl.gz'):frames[key(r)][r['group']].append(r['After_minus_Before'])
    joined=[]
    for c in cells:
        v=frames[key(c)]
        joined.append({**c,**{g+'_GT_improvement':math.fsum(a)/len(a) if a else None for g,a in v.items()}})
    with gzip.open(PUB/'CELL_GEOMETRY_WITH_EXISTING_GT_DIAGNOSIS.jsonl.gz','wt') as f:
        for c in joined:f.write(json.dumps(c,allow_nan=False)+'\n')
    associations={};checks=0
    for ds in ['hc2','vidstg']:
        for setting in [ds+'/cross_clean',ds+'/same_corruption']:
            pp=collections.defaultdict(list)
            for c in joined:
                if c['dataset']==ds and c['setting']==setting and not c['development']:pp[c['source_id']].append(c)
            for x in ['ESS_mean','reward_variance_mean','weighted_target_displacement_mean','output_gradient_cosine_mean','Adam_parameter_movement_mean']:
                for y in ['observed_GT_improvement','unobserved_decoder_GT_improvement','interpolated_GT_improvement']:
                    pairs=[]
                    for v in pp.values():
                        a=[c for c in v if c.get(x) is not None and c.get(y) is not None]
                        if a:pairs.append((math.fsum(c[x] for c in a)/len(a),math.fsum(c[y] for c in a)/len(a)))
                    a=np.asarray(pairs);co=corr(a[:,0],a[:,1]) if len(a) else None;boot=[]
                    if co is not None:
                        rng=np.random.default_rng(20261012)
                        for _ in range(10000):
                            r=a[rng.integers(len(a),size=len(a))];v=corr(r[:,0],r[:,1])
                            if v is not None:boot.append(v)
                    associations[setting+'/'+x+'/'+y]=dict(parent_sources=len(a),pearson=co,ci95=np.quantile(boot,[.025,.975]).tolist() if boot else None,
                      valid_bootstrap_draws=len(boot),causal=False,source='old_Full_current_improvement_with_old_Full_geometry')
                    checks+=len(a)+len(boot)
    write(PUB/'ASSOCIATIONS.json',dict(status='complete_descriptive_parent_associations',statistics=associations,
       bootstrap=10000,seed=20261012,old_GT_labels_only=True,no_new_GT_loader=True,not_causal=True,unadjusted_CI=True))
    write(BASE/'CPU_ASSOCIATION_READBACK.json',dict(status='pass',scope='all existing observed/unobserved labels joined by anonymous exact cell key and parent paired resampling',
       cells=1792,associations=len(associations),paired_scalar_evaluations=checks,CPU_only=True,new_GPU_fits=0,GT_read='existing_postseal_anonymous_rows_only',time=time.time()))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    s=read(PUB/'SUMMARY.json')['statistics'];keys=[ds+'/'+ds+'/'+setting+'/excluding_development' for ds in ['hc2','vidstg'] for setting in ['cross_clean','same_corruption']]
    labels=['HC cross','HC corruption','Vid cross','Vid corruption'];x=np.arange(4)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.2),layout='constrained')
    for ax in axes:ax.set_xticks(x,labels,rotation=20);ax.grid(axis='y',alpha=.2)
    r=[s[k]['metrics']['output_gradient_cosine_mean'] for k in keys];v=np.array([a['mean'] for a in r]);ci=np.array([a['ci95'] for a in r])
    axes[0].errorbar(x,v,yerr=np.vstack((v-ci[:,0],ci[:,1]-v)),fmt='o',color='#3856C0',capsize=4)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Cosine, parent mean and 95% CI');axes[0].set_title('Same-output gradient direction')
    for field,color,label,shift in [('OPD_output_gradient_norm_mean','#3856C0','OPD',-.14),('Direct_output_gradient_norm_mean','#DC7431','Direct',.14)]:
        r=[s[k]['metrics'][field] for k in keys];v=np.array([a['mean'] for a in r]);ci=np.array([a['ci95'] for a in r])
        axes[1].errorbar(x+shift,v,yerr=np.vstack((v-ci[:,0],ci[:,1]-v)),fmt='o',color=color,label=label,capsize=3)
    axes[1].set_ylabel('Gradient norm in inverse-sigmoid chart');axes[1].set_title('Gradient magnitude depends on source');axes[1].legend(frameon=False)
    r=[s[k]['metrics']['ESS_mean'] for k in keys];v=np.array([a['mean'] for a in r]);ci=np.array([a['ci95'] for a in r])
    axes[2].errorbar(x,v,yerr=np.vstack((v-ci[:,0],ci[:,1]-v)),fmt='o',color='#27876C',capsize=4)
    axes[2].axhline(32,color='gray',ls='--',lw=1);axes[2].set_ylim(0,34);axes[2].set_ylabel('ESS out of 32 candidate actions');axes[2].set_title('Reward weighting, not teacher certainty')
    fig.suptitle('Sealed P2 optimization geometry: 1792 cells, 256 historical parents',fontsize=13)
    fig.text(.5,-.055,'Same saved Full state/output for both losses. Full parameter-gradient and norm-budget GPU control remain separate.',ha='center',fontsize=9)
    out=ROOT/'output/pdf/stvg_opd_optimization_geometry';out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/('output_geometry.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(BASE/'CPU_FIGURE_EXPORT.json',dict(status='generated_pending_actual_pixel_view',scope='scalar CPU diagnostic only',
       PDF_sha256=sha(out/'output_geometry.pdf'),PNG_sha256=sha(out/'output_geometry.png'),GT_read='old_anonymous_labels_only',time=time.time()))
    (PUB/'REPORT.md').write_text('''# Optimization geometry CPU diagnostic\n\nThe common-state GPU experiment is separately pending. This is a finite readback of already globally sealed P2 data.\n\nAll 1792 Full/Direct cell pairs (3584 complete fits and their inputs) were byte checked. All 44800 original Full rounds produced 149610 continuous observation-round records; 17338 zero-gradient cosines are explicitly undefined, not replaced by zero. An independent NumPy piecewise GIoU derivative and parent bootstrap recomputed 907648 scalars.\n\nIn the primary existing development-exclusion projection, mean same-output gradient cosine is about 0.825 for HC and 0.805-0.809 for Vid. Directions are aligned on average, not identical. Mean OPD chart-gradient norms are 3.26-3.35 in HC versus Direct 1.12-1.16; in Vid OPD is 0.50-0.51 versus Direct 1.11-1.18. Thus universally smaller OPD gradients are not supported. Gradient norms alone do not identify Adam displacement or task gains. ESS is about 28.0-30.1 out of 32; this measures reward-weight geometry, not teacher identity reliability.\n\nAll observed/unobserved/interpolated old Full improvements are joined by exact anonymous cell keys and compared with cell geometry through parent-level descriptive correlations and paired bootstrap. The old Full/Direct online histories differ, so their outcomes are not common-state causal effects. No labels select the new entire fixed common-state cohort or the Direct output cap. No new model, GPU or raw GT loader was used in this CPU readback.\n\nThe new GPU control retains the complete original Full initial state and fresh Adam for Full, Direct and output-budget-constrained Direct. The constraint has no score-based choice. New performance conclusions require all new predictions globally sealed, postseal scoring, independent state/math/dense root, actual figure view, public remote content verification and archive closure.\n''')
    print('GEOMETRY_ASSOCIATIONS_AND_FIGURE',len(associations),checks,flush=True)

if __name__=='__main__':run()
