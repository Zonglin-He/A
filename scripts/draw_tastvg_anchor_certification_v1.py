"""Render sealed CPU calibration results; never fit or choose a target cutoff."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_anchor_certification/2026-10-03'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
BLUE='#315d8b';ORANGE='#c88936';GREEN='#338a73';RED='#ae4f54';GRAY='#70767c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
names={'vidstg':'VidSTG','hc2':'HC-STVG-v2'}
def read(p):return json.loads(Path(p).read_text())
curves={ds:read(OUT/'confirm'/ds/'CURVES.json')['corruption'] for ds in names}
summaries={ds:read(OUT/'confirm'/ds/'SUMMARY.json')['corruption'] for ds in names}
data={'curves':curves,'confirmation':summaries,'scope':'confirmation corruption; source-balanced ratios; pointwise 10000 source bootstrap; no target operating-point selection'}
receipts=[]
def save(fig,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();outside=[]
    # Matplotlib creates extra tick artists outside the visible interval but
    # does not render them. Bounds checking must mirror that rendering rule.
    omitted=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lo-1e-12 or tick.get_loc()>hi+1e-12:
                    omitted.update([id(tick.label1),id(tick.label2)])
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or id(text) in omitted:continue
        b=text.get_window_extent(renderer)
        if b.x0 < -2 or b.y0 < -2 or b.x1>w+2 or b.y1>h+2:outside.append(text.get_text())
    assert not outside,(stem,outside)
    files=[]
    for ext in ['png','pdf']:
        p=FIG/(stem+'.'+ext);fig.savefig(p,dpi=210);files.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    receipts.append({'stem':stem,'text_bounds_pass':True,'files':files});plt.close(fig)
def point(ax,s,field,color,marker='o',size=22,confidence=False,zorder=3):
    if s['cells']==0:return
    ratio=s['ratios'].get(field)
    if not ratio or ratio['mean'] is None:return
    x=100*s['metrics']['accepted']['mean'];y=100*ratio['mean']
    if confidence and ratio['ci95'] is not None:
        lo,hi=np.array(ratio['ci95'])*100;ax.vlines(x,lo,hi,color=color,lw=.8,alpha=.65,zorder=zorder-1)
        lo,hi=np.array(s['metrics']['accepted']['ci95'])*100;ax.hlines(y,lo,hi,color=color,lw=.8,alpha=.65,zorder=zorder-1)
    ax.scatter([x],[y],color=color,marker=marker,s=size,zorder=zorder)
def panel(ax,series,field):
    for prefix,col,style in [('Margin_',ORANGE,'-'),('LCB_',GREEN,'--')]:
        points=[]
        for name,s in series.items():
            ratio=s['ratios'].get(field)
            if name.startswith(prefix) and ratio and ratio['mean'] is not None:
                points.append((100*s['metrics']['accepted']['mean'],100*ratio['mean'],name))
        points.sort()
        if points:ax.plot([p[0] for p in points],[p[1] for p in points],style,color=col,lw=1.25,alpha=.8)
        for _,_,name in points:point(ax,series[name],field,col,size=13,confidence=prefix=='LCB_',zorder=2)
    point(ax,series['L32'],field,BLUE,marker='X',size=55,confidence=True,zorder=4)
    point(ax,series['Selective'],field,GREEN,marker='D',size=55,confidence=True,zorder=5)
    ax.set_xlim(-3,104);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='y',color='#e7e9eb',lw=.65);ax.set_axisbelow(True)
    if field in ['beneficial_precision','accepted_severe_rate']:
        ax.set_ylim(-3,105);ax.set_yticks([0,25,50,75,100])
    else:ax.axhline(0,color=GRAY,lw=.8)
    if series['Selective']['raw_counts']['accepted']==0:
        ax.text(.025,.04,'Locked rule: zero acceptance\nConditional quantity undefined',transform=ax.transAxes,fontsize=8,color=GRAY,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.88))
def legend(fig):
    fig.legend(handles=[Line2D([0],[0],color=ORANGE,marker='o',lw=1.25,label='Source margin cutoffs'),
      Line2D([0],[0],color=GREEN,marker='o',ls='--',lw=1.25,label='Source bootstrap-q family'),
      Line2D([0],[0],color=BLUE,marker='X',ls='',markersize=7,label='Original L32'),
      Line2D([0],[0],color=GREEN,marker='D',ls='',markersize=7,label='Locked 5% rule')],
      loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,fontsize=9)

fig,axes=plt.subplots(2,3,figsize=(13.7,7.1));fig.subplots_adjust(top=.86,bottom=.12,left=.075,right=.985,hspace=.43,wspace=.32)
for i,ds in enumerate(names):
    for j,(field,label) in enumerate([('beneficial_precision','Beneficial replacement precision (%)'),
        ('accepted_mean_delta_t','Accepted mean delta tIoU (pp)'),('accepted_severe_rate','Severe harm among replacements (%)')]):
        ax=axes[i,j];panel(ax,curves[ds]['all'],field);ax.set_ylabel(label);ax.set_title(names[ds])
        if i==1:ax.set_xlabel('Replacement coverage among experts (%)')
legend(fig);save(fig,'precision_coverage')

fig,axes=plt.subplots(2,2,figsize=(10.6,7.0));fig.subplots_adjust(top=.855,bottom=.12,left=.085,right=.985,hspace=.4,wspace=.26)
for i,ds in enumerate(names):
    for j,g in enumerate(['small','large']):
        ax=axes[i,j];s=curves[ds][g]['L32']
        if s['cells']:
            panel(ax,curves[ds][g],'beneficial_precision');ax.set_ylabel('Beneficial replacement precision (%)')
        else:
            ax.set_xlim(-3,104);ax.set_ylim(-3,105);ax.set_xticks([0,25,50,75,100]);ax.set_yticks([0,25,50,75,100]);ax.text(.5,.5,'No proposed large corrections\nin this confirmation panel',ha='center',va='center',transform=ax.transAxes,color=GRAY)
        title='r < 0.5' if g=='small' else 'r >= 0.5'
        ax.set_title(f'{names[ds]} | {title} | {s["cells"]} cells / {s["sources"]} sources')
        if i==1:ax.set_xlabel('Replacement coverage within stratum (%)')
legend(fig);save(fig,'correction_size_curves')

fig,axes=plt.subplots(2,2,figsize=(11.7,7.3));fig.subplots_adjust(top=.85,bottom=.11,left=.075,right=.985,hspace=.42,wspace=.28)
dec=read(OUT/'DECISIONS.json')
for i,ds in enumerate(names):
    model=read(OUT/ds/'CALIBRATION.json');rows=[r for r in read(OUT/ds/'SOURCE_CALIBRATION_ROWS.json') if r['eligible']]
    x=np.array(model['knots']);ax=axes[i,0]
    ax.scatter([r['margin'] for r in rows],[100*r['true_delta_t'] for r in rows],s=25,color=GRAY,alpha=.6,label='Source validation winners')
    ax.plot(x,100*np.array(model['mean']),color=BLUE,lw=1.8,label='Isotonic mean delta')
    ax.plot(x,100*np.array(model['lower']['0.05']),color=GREEN,lw=1.8,label='Pointwise 5th percentile')
    ax.axhline(0,color=RED,lw=.8,ls='--');ax.set_title(f'{names[ds]} | {len(rows)} eligible source winners')
    ax.set_xlabel('Frozen L margin over source native');ax.set_ylabel('True / calibrated delta tIoU (pp)');ax.grid(axis='y',color='#e7e9eb',lw=.65)
    ax=axes[i,1];target=[r['margin'] for r in dec if r['dataset']==ds and r['top_index']!=r['anchor_index']]
    hi=max(target+[float(x[-1])]);bins=np.linspace(0,hi+1e-8,22)
    ax.hist(target,bins=bins,color=BLUE,alpha=.7,label='Target proposed margins')
    ax.axvspan(float(x[0]),float(x[-1]),color=GREEN,alpha=.12,label='Observed source calibration domain')
    ax.axvline(float(x[0]),color=GREEN,ls='--',lw=1);ax.axvline(float(x[-1]),color=GREEN,ls='--',lw=1)
    ax.set_xlabel('Frozen L margin over target A8');ax.set_ylabel('Expert arrivals');ax.set_title(f'{names[ds]} | all 144 expert arrivals')
    ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='y',color='#e7e9eb',lw=.65);ax.set_axisbelow(True)
fig.legend(handles=[Line2D([0],[0],color=GRAY,marker='o',ls='',label='Source validation winners'),
    Line2D([0],[0],color=BLUE,lw=1.8,label='Isotonic mean delta'),
    Line2D([0],[0],color=GREEN,lw=1.8,label='Pointwise 5th percentile')],loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=9)
data['calibration']={ds:read(OUT/ds/'CALIBRATION.json') for ds in names};save(fig,'source_margin_calibration')

fig,axes=plt.subplots(1,2,figsize=(10.3,3.9));fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.9,wspace=.29)
for j,(field,label) in enumerate([('dv','Complete corruption flow delta vIoU (pp)'),('gross_loss','Complete corruption flow gross loss (pp)')]):
    ax=axes[j]
    for i,ds in enumerate(names):
        for off,arm,col in [(-.12,'L32',BLUE),(.12,'Selective',GREEN)]:
            s=summaries[ds]['all'][arm]['metrics'][field];y=s['mean']*100;lo,hi=np.array(s['ci95'])*100
            ax.vlines(i+off,lo,hi,color=col,lw=1.2);ax.scatter([i+off],[y],color=col,s=45,marker='o' if arm=='L32' else 'D')
            ax.annotate(f'{y:+.3f}' if field=='dv' else f'{y:.3f}',(i+off,y),xytext=(6,3),textcoords='offset points',color=col,fontsize=9)
    ax.axhline(0,color=GRAY,lw=.8);ax.set_xticks([0,1],list(names.values()));ax.set_xlim(-.5,1.6);ax.set_ylabel(label);ax.grid(axis='y',color='#e7e9eb',lw=.65);ax.set_axisbelow(True)
axes[0].legend(handles=[Line2D([0],[0],color=BLUE,marker='o',ls='',label='Original L32'),Line2D([0],[0],color=GREEN,marker='D',ls='',label='Locked Selective-L32')],frameon=False,fontsize=9)
save(fig,'confirmation_readout')
(OUT/'FIGURE_DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(OUT/'DRAW_RECEIPT.json').write_text(json.dumps({'status':'rendered_text_bounds_pass','figures':receipts,'negative_results_preserved':True,'human_visual_review_pending':True},indent=2)+'\n')
print(json.dumps({'status':'rendered','figures':len(receipts),'files':sum(len(r['files']) for r in receipts)}))
