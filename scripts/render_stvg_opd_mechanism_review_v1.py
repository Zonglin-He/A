"""Scalar-only diagnostic figures; no private RGB, GT geometry or query text."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.decota_matrix_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1';PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'

def draw(ax,labels,metrics):
    x=np.arange(len(labels));valid=[(i,m) for i,m in enumerate(metrics) if m['mean'] is not None]
    if valid:
        at=np.array([i for i,m in valid]);means=np.array([m['mean']*100 for i,m in valid]);ci=np.array([np.array(m['ci95'])*100 for i,m in valid])
        ax.bar(at,means,color=['#31689b' if v>=0 else '#ad5353' for v in means],width=.62)
        ax.errorbar(at,means,yerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='none',color='#262626',capsize=3)
    for i,m in enumerate(metrics):
        ax.text(i,.97,'n='+str(m['parents']),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        if m['mean'] is None:ax.text(i,0,'NA',ha='center',va='bottom',fontsize=9)
    ax.axhline(0,lw=.7,color='#555555');ax.set_xticks(x,labels,fontsize=9);ax.set_ylabel('Difference (pp)',fontsize=10)
    for sp in ['top','right']:ax.spines[sp].set_visible(False)

def run():
    assert read(BASE/'ROOT_STATISTICS_READBACK.json')['status']=='pass';s=read(PUB/'SUMMARY.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    paths=[];out=ROOT/'output/pdf/stvg_opd_mechanism_review';out.mkdir(parents=True,exist_ok=True)
    for kind in ['quality','propagation','oracles']:
        fig,axes=plt.subplots(2,2,figsize=(11.6,7.2),layout='constrained')
        for ax,setting in zip(axes.flat,['hc2/cross_clean','vidstg/cross_clean','hc2/same_corruption','vidstg/same_corruption']):
            key=setting+'/excluding_development';ax.set_title(setting.replace('/',' - ').replace('hc2','HC-STVG-v2').replace('vidstg','VidSTG'),fontsize=11)
            if kind=='quality':
                labels=['Correct\nzero','Correct\npositive','Incorrect\nzero','Incorrect\npositive']
                metrics=[s[kind][key][k]['statistics']['metrics']['Full_minus_Direct_current'] for k in ['correct/zero','correct/positive','incorrect/zero','incorrect/positive']]
            elif kind=='propagation':
                labels=['Observed','Unobserved\ndecoder','Interpolated','Outside\nNative WHEN']
                metrics=[s[kind][key]['all_eligible'][k]['statistics']['metrics']['After_minus_DINO'] for k in ['observed','unobserved_decoder','interpolated','outside_native']]
            else:
                labels=['Spatial oracle','Temporal oracle'];metrics=[s[kind][key]['pooled']['metrics'][k] for k in ['Spatial_headroom','Temporal_headroom']]
            draw(ax,labels,metrics)
        title={'quality':'OPD - Direct: current effect by expert quality and first-round sampled reward',
           'propagation':'OPD - DINO Refine: spatial IoU by real frame support',
           'oracles':'Offline oracle headroom above Native mvIoU'}[kind]
        fig.suptitle(title,fontsize=13)
        fig.get_layout_engine().set(rect=(0,.065,1,1))
        fig.text(.5,.022,'Development parents excluded. Equal-parent means; paired 95% CIs, 10,000 draws. Historical target exposure.',ha='center',fontsize=9)
        if kind=='propagation':fig.text(.5,.045,'Outside-WHEN values are geometry only; task coverage remains zero. NA is not zero.',ha='center',fontsize=9)
        pdf=out/(kind+'.pdf');png=PUB/(kind+'.png');fig.savefig(pdf);fig.savefig(png,dpi=160);plt.close(fig)
        paths.extend([pdf,png])
    write(BASE/'FIGURE_EXPORT.json',dict(status='exported_pending_actual_pixel_view',files={str(p.relative_to(ROOT)):sha(p) for p in paths},new_model_calls=0))

if __name__=='__main__':run()
