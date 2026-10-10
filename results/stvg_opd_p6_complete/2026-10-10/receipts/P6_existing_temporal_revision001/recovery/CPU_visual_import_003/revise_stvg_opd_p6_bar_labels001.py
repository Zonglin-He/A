"""Saved scalar-only label spacing revision; no inference, GT or fit access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.stvg_opd_p6_existing_common001 import *


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats=read(PUB/'P6/STATISTICS.json');dest=NS/'actual_root_signal_views'
    old=read(dest/'COMPLETION.json');assert old['plot_pairs']==6
    labels=['Native','Actionness','Time head','Spatial OPD','Offline GT head'];colors=['#4b667e','#ed9b40','#657bc8','#24a18c','#969696']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,7));allarms=ARMS+['Offline_GT_Head']
    for i,ds in enumerate(['hc2','vidstg']):
        for j,m in enumerate(['v','t','s']):
            ax=axes[i,j];v=[stats[ds]['metrics'][a+'_'+m] for a in allarms]
            means=np.asarray([x['parent_macro'] for x in v])*100;ci=np.asarray([x['ci95'] for x in v])*100
            ax.bar(range(5),means,color=colors);ax.errorbar(range(5),means,yerr=np.maximum(np.array([means-ci[:,0],ci[:,1]-means]),0),fmt='none',ecolor='black',capsize=3)
            ax.set_xticks(range(5),labels,fontsize=8,rotation=22,ha='right');ax.set_title(ds+' '+{'v':'vIoU','t':'tIoU','s':'fixed-GT spatial IoU'}[m]);ax.set_ylabel('%')
    fig.tight_layout();files=[]
    for ext in ['png','pdf']:
        p=PUB/'P6'/('complete_deployment_and_offline_oracle_labels002.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    plots=[dict(name='complete_deployment_and_offline_oracle_labels002',files=files)]+old['plots'][1:]
    write(dest/'VISUAL_LABEL_REVISION.json',dict(status='complete_display_labels_only',scope='short arm labels and 22-degree spacing on the same six bar charts',
        original_completion_sha256=sha(dest/'COMPLETION.json'),original_scalar_sha256=sha(PUB/'P6/STATISTICS.json'),
        plots=plots,plot_pairs=6,cases=6,original_plot_bytes_preserved=True,bar_values_scales_CI_unchanged=True,
        new_GPU_calls=0,new_GT_read=False,time=time.time()))


if __name__=='__main__':run()
