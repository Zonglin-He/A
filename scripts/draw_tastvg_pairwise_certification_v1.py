"""Publication plots for the single prelocked operating rule; no target tuning."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_pairwise_certification/2026-10-03'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
NAMES={'vidstg':'VidSTG','hc2':'HC-STVG-v2'}
COLORS={'L32':'#536f91','Pair-Raw':'#ce934d','Pair-Norm':'#2c8978'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
def read(p):return json.loads(Path(p).read_text())
data=dict(confirmation={ds:read(OUT/'confirm'/ds/'SUMMARY.json') for ds in NAMES},
          strata={ds:read(OUT/'confirm'/ds/'DIAGNOSTIC_STRATA.json') for ds in NAMES},
          calibration={ds:{arm:read(OUT/ds/(arm+'.json')) for arm in ['Pair-Raw','Pair-Norm']} for ds in NAMES})
receipts=[]
def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();omit=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for t in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo-1e-12<=t.get_loc()<=hi+1e-12:omit.update([id(t.label1),id(t.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in omit:continue
        b=t.get_window_extent(renderer);assert b.x0>=-2 and b.y0>=-2 and b.x1<=w+2 and b.y1<=h+2,(name,t.get_text())
    files=[]
    for ext in ['png','pdf']:
        p=FIG/(name+'.'+ext);fig.savefig(p,dpi=210)
        files.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    receipts.append(dict(stem=name,text_bounds_pass=True,files=files));plt.close(fig)
def grid(ax):
    ax.grid(axis='y',color='#e6e9ec',lw=.65);ax.set_axisbelow(True)
def legend(fig):
    fig.legend(handles=[Line2D([0],[0],marker='o',color=c,label=a,lw=1.2) for a,c in COLORS.items()],
               loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)

fig,axes=plt.subplots(1,2,figsize=(10.5,3.9));fig.subplots_adjust(left=.08,right=.985,bottom=.16,top=.79,wspace=.34)
for ax,(f,label) in zip(axes,[('dv','Complete corruption flow delta vIoU (pp)'),('gross_loss','Complete corruption flow gross loss (pp)')]):
    for i,ds in enumerate(NAMES):
        for slot,(offset,(arm,c)) in enumerate(zip([-.2,0,.2],COLORS.items())):
            z=data['confirmation'][ds]['corruption']['all'][arm]['metrics'][f];y=100*z['mean'];lo,hi=100*np.array(z['ci95'])
            ax.vlines(i+offset,lo,hi,color=c,lw=1.2);ax.scatter(i+offset,y,color=c,s=40)
            ax.annotate(f'{y:+.3f}' if f=='dv' else f'{y:.3f}',(i+offset,y),xytext=(0,5+10*slot),textcoords='offset points',ha='center',fontsize=8,color=c)
    ax.axhline(0,color='#757b82',lw=.8);ax.set_xticks([0,1],NAMES.values());ax.set_xlim(-.5,1.65);ax.set_ylabel(label);grid(ax)
legend(fig);save(fig,'confirmation_readout')

fig,axes=plt.subplots(2,3,figsize=(12.8,6.3));fig.subplots_adjust(left=.065,right=.985,bottom=.11,top=.86,hspace=.5,wspace=.3)
for i,ds in enumerate(NAMES):
    for j in range(3):
        ax=axes[i,j]
        for k,(arm,c) in enumerate(COLORS.items()):
            s=data['confirmation'][ds]['corruption']['expert'][arm]
            if j==0:z=s['metrics']['accepted'];label='Replacement coverage (%)'
            elif j==1:z=s['ratios']['beneficial_precision'];label='Beneficial replacement precision (%)'
            else:z=dict(mean=s['raw_counts']['severe'],ci95=None);label='Severe harm count (delta vIoU < -5 pp)'
            factor=100 if j<2 else 1;y=factor*z['mean'];ax.bar(k,y,color=c,width=.6,alpha=.88)
            if z['ci95'] is not None:
                lo,hi=factor*np.array(z['ci95']);ax.vlines(k,lo,hi,color='#333',lw=1);ax.hlines([lo,hi],k-.09,k+.09,color='#333',lw=.8)
            if j==0:ax.text(k,3,f'{s["raw_counts"]["accepted"]}/{s["cells"]}',ha='center',fontsize=9,color='white')
            if j==2:ax.text(k,y+.3,str(int(y)),ha='center',fontsize=9,color=c)
        ax.set_title(NAMES[ds]);ax.set_xticks(range(3),COLORS.keys());ax.set_ylabel(label);grid(ax)
        if j<2:ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100])
        else:ax.set_ylim(0,15)
legend(fig);save(fig,'confirmation_acceptance_harm')

fig,axes=plt.subplots(2,2,figsize=(10.8,6.3));fig.subplots_adjust(left=.075,right=.985,bottom=.11,top=.86,hspace=.5,wspace=.28)
dec=read(OUT/'DECISIONS.json')
for i,ds in enumerate(NAMES):
    for j,arm in enumerate(['Pair-Raw','Pair-Norm']):
        ax=axes[i,j];m=data['calibration'][ds][arm];x=np.array(m['knots'])
        ax.plot(x,100*np.array(m['mean']),color='#536f91',lw=1.6)
        ax.plot(x,100*np.array(m['lower']),color='#2c8978',lw=1.6);ax.axhline(0,color='#b15a58',ls='--',lw=.8)
        yy=min(min(m['lower'])*100,-1)-1
        for accepted,c in [(False,'#b15a58'),(True,'#2c8978')]:
            vals=[r['evidence'][arm]['margin'] for r in dec if r['dataset']==ds and r['top_index']!=r['anchor_index'] and
                  (r['evidence'][arm]['reason']=='accepted')==accepted]
            if vals:ax.plot(vals,[yy]*len(vals),'|',color=c,alpha=.65,markersize=6)
        ax.set_xscale('log');ax.set_ylabel('Calibrated delta tIoU (pp)');ax.set_xlabel('Raw pair margin' if j==0 else 'Pair margin / (score IQR + epsilon)')
        ax.set_title(f'{NAMES[ds]} | {arm} | {m["source_count"]} sources');grid(ax)
fig.legend(handles=[Line2D([0],[0],color='#536f91',label='Source isotonic mean'),Line2D([0],[0],color='#2c8978',label='Fixed 5th percentile'),
                    Line2D([0],[0],color='#2c8978',marker='|',ls='',label='Accepted target margin'),Line2D([0],[0],color='#b15a58',marker='|',ls='',label='Rejected target margin')],
           loc='upper center',bbox_to_anchor=(.5,.99),ncol=4,frameon=False,fontsize=9)
save(fig,'source_pair_calibration')
(OUT/'FIGURE_DATA.json').write_text(json.dumps(data,indent=2)+'\n')
(OUT/'DRAW_RECEIPT.json').write_text(json.dumps(dict(status='rendered_text_bounds_pass',figures=receipts,visual_review_pending=True),indent=2)+'\n')
print(json.dumps(dict(status='rendered',figures=len(receipts))))
