"""Display-only repair from unchanged, complete, anonymous fixed-case signals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE,PUB,read,write,sha,verify


def run():
    verify()
    source=PUB/'FIXED_HISTORICAL_CASE_SIGNALS.json'
    data=read(source)
    assert data['status']=='pass' and data['cases']==6 and data['complete_fits']==24
    assert read(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json')['counts']['complete_fits']==5376
    dest=BASE/'fixed_case_signal_visual003';dest.mkdir(parents=True,exist_ok=True)
    old=[PUB/'fixed_historical_case_signals.png',PUB/'fixed_historical_case_signals.pdf']
    write(dest/'CAPTURE_RECEIPT.json',dict(status='preserved',
        scope='actual root view found the original overlong per-axis labels clipped; display-only replacement',
        original_outputs={str(p.relative_to(ROOT)):sha(p) for p in old},
        original_case_completion_sha256=sha(BASE/'actual_fixed_case_views002/COMPLETION.json'),
        scalar_signal_sha256=sha(source),CPU_only=True,model_calls=0,new_GT_read=False,time=time.time()))
    write(dest/'RUNTIME.json',dict(status='locked',source_sha256=sha(Path(__file__)),
        scalar_signal_sha256=sha(source),case_lock_sha256=sha(BASE/'actual_fixed_case_views002/CASE_DIAGNOSTIC_LOCK.json'),
        scope='short two-line y labels; identical saved curves, colors, limits, ordering, legend and case selection',
        CPU_only=True,model_calls=0,new_GT_read=False,time=time.time()))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'event_full':'Event-scoped Full','whole_video':'Whole-video','carry_query':'Query carry','full_LN_carry':'Full LN carry'}
    colors={'event_full':'#206c99','whole_video':'#dd8c35','carry_query':'#a66390','full_LN_carry':'#14897d'}
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    curves=0
    for i,case in enumerate(data['records']):
        ax=axes[i%3,i//3]
        for arm in labels:
            chain=case['signals'][arm]['rounds']
            ax.plot([v['round'] for v in chain],
                [np.nan if v['central_after_GT_IoU'] is None else v['central_after_GT_IoU'] for v in chain],
                label=labels[arm],color=colors[arm]);curves+=1
        ax.set(title=f"{case['dataset']}: historical {case['historical_role'].replace('_',' ')}",
            xlabel='Configured update round',ylabel='GT IoU\n(admitted evidence)',ylim=(0,1))
        ax.grid(alpha=.15)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=4,fontsize=9)
    png=PUB/'fixed_historical_case_signals_visual003.png'
    pdf=PUB/'fixed_historical_case_signals_visual003.pdf'
    assert not png.exists() and not pdf.exists()
    fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    assert sha(source)==read(dest/'RUNTIME.json')['scalar_signal_sha256']
    assert all(sha(ROOT/p)==h for p,h in read(dest/'CAPTURE_RECEIPT.json')['original_outputs'].items())
    write(dest/'COMPLETION.json',dict(status='pending_actual_root_view',scope='display repair only',
        cases=6,unchanged_saved_curves=curves,scalar_signal_sha256=sha(source),
        runtime_sha256=sha(dest/'RUNTIME.json'),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [png,pdf]},
        original_payloads_and_images_preserved=True,CPU_only=True,model_calls=0,new_GT_read=False,time=time.time()))
    print('FIXED_SIGNAL_VISUAL003_RENDERED',curves,flush=True)


if __name__=='__main__':run()
