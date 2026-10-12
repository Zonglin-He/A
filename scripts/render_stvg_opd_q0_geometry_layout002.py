"""CPU presentation correction: move RMS legends outside actual bar data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha

def run():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b=ROOT/'artifacts/stvg_opd_temporal_spatial_complementarity_v1'
    q=read(ROOT/'results/stvg_opd_temporal_spatial_complementarity/2026-10-12/Q0_DIAGNOSTIC_SUMMARY.json')
    groups=['Q_random','Q_observable','LN_observable']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12.3,8.6),constrained_layout=True)
    for j,ds in enumerate(['hc2','vidstg']):
        x=np.arange(3);dat=q[ds]
        for offset,field,label,color in [(-.1,'tanh_retention','Pair tanh retention','#286da8'),(.1,'mean_over_top1_norm','Mean / top1 displacement','#b85b30')]:
            vals=[dat[g]['pair_statistics'][field]['median'] if field=='tanh_retention' else dat[g]['statistics'][field]['median'] for g in groups]
            axes[0,j].bar(x+offset,np.array(vals)*100,width=.2,label=label,color=color)
        axes[0,j].set_xticks(x,['Q-random','Q-observable','LN-observable']);axes[0,j].set_ylabel('Median retention (%)');axes[0,j].set_title('HC' if ds=='hc2' else 'Vid');axes[0,j].legend(fontsize=8,loc='upper left');axes[0,j].grid(axis='y',alpha=.2)
        for offset,field,label,color in [(-.2,'native_Jensen_RMS','Native Jensen gap','#b85b30'),(0,'full_decoder_Jensen_RMS','Full decoder Jensen gap','#765aa4'),(.2,'mean_native_RMS','True parameter-mean native movement','#286da8')]:
            vals=[dat[g]['statistics'][field]['median'] for g in groups]
            axes[1,j].bar(x+offset,vals,width=.2,label=label,color=color)
        axes[1,j].set_yscale('log');axes[1,j].set_xticks(x,['Q-random','Q-observable','LN-observable']);axes[1,j].set_ylabel('Normalized output RMS, median');axes[1,j].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False);axes[1,j].grid(axis='y',alpha=.2)
    fig.suptitle('Correct Native Q0: actual pair cancellation and nonlinear readout',fontsize=15)
    path=ROOT/'output/pdf/stvg_opd_temporal_spatial_complementarity/q0_mean_readout_geometry_layout002.pdf'
    assert not path.exists();fig.savefig(path);plt.close(fig)
    write(b/'FIGURE_LAYOUT_REVISION002.json',dict(status='pending_actual_pixel_view',scope='RMS legend placement only; identical summary/data and original PDF preserved',
        original_export_sha256=sha(b/'FIGURE_EXPORT.json'),new_PDF_sha256=sha(path),new_PDF_bytes=path.stat().st_size,
        source_sha256=sha(Path(__file__)),scientific_values_changed=False,CPU_only=True,GT_read=False,time=time.time()))

if __name__=='__main__':run()
