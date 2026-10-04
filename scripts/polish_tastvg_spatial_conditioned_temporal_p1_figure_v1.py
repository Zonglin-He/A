"""Post-analysis layout-only repair: place paired-CI legend above the axes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
def main():
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pub=ROOT/'results/tastvg_spatial_conditioned_temporal_p1/2026-10-04'
    recovery=ROOT/'artifacts/tastvg_spatial_conditioned_temporal_p1_v1/recovery/figure_layout_001'
    recovery.mkdir(parents=True,exist_ok=False)
    for f in ['teacher_differences.png','teacher_differences.pdf','REPORT_COMPLETION.json']:shutil.copy2(pub/f,recovery/f)
    s=read(pub/'SUMMARY.json');panels=[('vidstg','search'),('vidstg','confirm'),('hc2','search'),('hc2','confirm')]
    fig,ax=plt.subplots(figsize=(9,4.5))
    for offset,b,color in [(-.09,'Full','#267c95'),(.09,'A_ROI','#c27b28')]:
        values=[s[d][sp]['corrupt']['metrics']['Soft_minus_'+b+'_t'] for d,sp in panels]
        x=np.arange(4)+offset
        ax.vlines(x,[100*v['ci95'][0] for v in values],[100*v['ci95'][1] for v in values],color=color,linewidth=1.8)
        ax.scatter(x,[100*v['mean'] for v in values],color=color,label='Soft − '+b,s=36)
    ax.axhline(0,color='#666',linewidth=.8)
    ax.set_xticks(range(4),[d+'\n'+sp for d,sp in panels])
    ax.set_ylabel('Teacher tIoU difference (pp), paired-source 95% CI')
    ax.spines[['top','right']].set_visible(False)
    fig.legend(loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,1.))
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ['png','pdf']:fig.savefig(pub/('teacher_differences.'+ext),dpi=200)
    plt.close(fig)
    original=read(recovery/'REPORT_COMPLETION.json')
    revised=dict(original,hashes={f:sha(pub/f) for f in original['hashes']},
        layout_revision='legend outside CI data area; underlying metrics/CI unchanged',
        previous_report_completion_sha256=sha(recovery/'REPORT_COMPLETION.json'),time=time.time())
    (pub/'REPORT_COMPLETION.json').unlink();write(pub/'REPORT_COMPLETION.json',revised)
    write(pub/'FIGURE_BINDINGS.json',dict(script='scripts/polish_tastvg_spatial_conditioned_temporal_p1_figure_v1.py',
        script_sha256=sha(__file__),summary_sha256=sha(pub/'SUMMARY.json'),changes='legend placement only',
        output_sha256={f:sha(pub/f) for f in ['teacher_differences.png','teacher_differences.pdf']},time=time.time()))
if __name__=='__main__':main()
