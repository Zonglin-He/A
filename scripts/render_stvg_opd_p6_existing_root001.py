"""Postseal P6 complete deployment/oracle contrasts and six actual signal/RGB cases."""
import collections
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import textwrap
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *
OUT=PUB/'P6';DEST=NS/'actual_root_signal_views'


def selected(rows):
    result=[]
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];used=set()
        groups=[('temporal_gain',sorted(rr,key=lambda r:(-r['delta_TemporalHead_minus_Native_v'],r['arrival']))),
            ('temporal_harm',sorted(rr,key=lambda r:(r['delta_TemporalHead_minus_Native_v'],r['arrival']))),
            ('teacher_task_mismatch',sorted(rr,key=lambda r:(r['signal']['gradient_cosine'] if r['signal']['gradient_cosine'] is not None else 2,r['arrival'])))]
        for name,ordered in groups:
            r=next(r for r in ordered if r['query_ordinal'] not in used);used.add(r['query_ordinal'])
            kind=name
            if name=='temporal_gain' and r['delta_TemporalHead_minus_Native_v']<=0:kind='largest_temporal_effect_available'
            if name=='temporal_harm' and r['delta_TemporalHead_minus_Native_v']>=0:kind='least_temporal_effect_available'
            result.append(dict(dataset=ds,kind=kind,arrival=r['arrival'],query_ordinal=r['query_ordinal'],source_id=r['source_id'],
                delta_temporal_v=r['delta_TemporalHead_minus_Native_v'],delta_temporal_t=r['delta_TemporalHead_minus_Native_t'],
                delta_spatial_v=r['delta_SpatialOPD_minus_Native_v'],gradient_cosine=r['signal']['gradient_cosine']))
    return result


def rgb(case,z,orig,inp,row,truth,span,directory):
    from PIL import Image,ImageDraw,ImageFont
    from scripts.run_decota_paper_main_v1 import frames_for
    from scripts.run_tastvg_full_b1_experts_v1 import observation
    from scripts.render_stvg_opd_revised_p1_root_v2 import coordinates
    ds=case['dataset'];frames,ids,reuse=frames_for(ds,row,collections.OrderedDict())
    frames,pixel,spec=observation(row,'clean',frames)
    assert ids==z['frame_ids']==inp['frame_ids'] and pixel==z['pixel_sha256']==inp['pixel_sha256'] and spec is None
    chosen=np.unique(np.linspace(0,len(ids)-1,5).round().astype(int)).tolist()
    width,height=row['input']['width'],row['input']['height'];w=310;h=round(height*w/width)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    big=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    canvas=Image.new('RGB',(110+len(chosen)*320,160+3*(h+36)+190),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f'{ds} {case["kind"]}: existing temporal and original spatial history',font=big,fill='black')
    draw.multiline_text((10,36),'\n'.join(textwrap.wrap(row['input']['caption'],width=140)),font=font,fill='black')
    draw.text((10,112),'Post-hoc illustrative frames; GT green, prediction coral. Original full spatial predecessors are retained.',font=font,fill='black')
    for band,a in enumerate(['Native','TemporalHead','SpatialOPD']):
        y=155+band*(h+36);draw.text((5,y+24),a,font=font,fill='black')
        pr=z['predictions'][a]
        for col,pos in enumerate(chosen):
            x=110+col*320;fid=ids[pos];canvas.paste(Image.fromarray(frames[pos]).resize((w,h)),(x,y+26))
            temporal_active=pr['physical_interval'][0]<=fid<pr['physical_interval'][1]
            draw.text((x,y),f'Frame {fid} | '+('in prediction' if temporal_active else 'outside prediction'),font=font,fill='black')
            boxes=[('Pred',coordinates(pr['boxes'][pos].numpy(),width,height),'#f05245')]
            if fid in truth:boxes.append(('GT',np.asarray(truth[fid]),'#159947'))
            else:draw.text((x+5,y+31),'No spatial GT at this frame',font=font,fill='white',stroke_width=1,stroke_fill='black')
            for tag,box,color in boxes:
                b=np.asarray(box,float)*(w/width);b[[0,2]]+=x;b[[1,3]]+=y+26
                draw.rectangle(b.tolist(),outline='white',width=5);draw.rectangle(b.tolist(),outline=color,width=3)
                draw.text((max(x,min(x+w-45,b[0])),max(y+26,b[1]-18)),tag,font=font,fill=color,stroke_width=1,stroke_fill='white')
    y=155+3*(h+36)+10;left=160;extent=ids[-1]+1-ids[0];scale=(canvas.width-left-20)/extent
    for i,(label,iv,color) in enumerate([('GT',span,'#159947')]+[(a,z['predictions'][a]['physical_interval'],'#f05245') for a in ARMS]):
        yy=y+i*28;draw.text((5,yy),label,font=font,fill='black');draw.line((left,yy+10,canvas.width-20,yy+10),fill='#dddddd',width=2)
        draw.line((left+(iv[0]-ids[0])*scale,yy+10,left+(iv[1]-ids[0])*scale,yy+10),fill=color,width=8)
    p=directory/(ds+'_'+case['kind']+'.png');p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
    return dict(dataset=ds,query_ordinal=case['query_ordinal'],kind=case['kind'],path=str(p.relative_to(ROOT)),sha256=sha(p),
        original_pixel_sha256=pixel,physical_original_pixels_verified=True,official_GT_coordinates_drawn=True,private_not_published=True)


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    torch.set_num_threads(2);verify();complete=read(BASE/'P6_CPU_COMPLETION.json');assert complete['queries']==64
    for f,h in read(DEST/'RUNTIME.json')['pins'].items():assert sha(ROOT/f)==h,f
    for f,h in complete['outputs'].items():assert sha(ROOT/f)==h
    rows=read(OUT/'ROWS.json');stats=read(OUT/'STATISTICS.json');cases=selected(rows);plots=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4b667e','#ed9b40','#657bc8','#24a18c','#969696'];allarms=ARMS+['Offline_GT_Head']
    labels=['Native','Actionness\nprojection','Existing\ntemporal head','Spatial OPD\noriginal history','Offline GT\nhead oracle']
    def savefig(fig,name):
        fig.tight_layout();files=[]
        for ext in ['png','pdf']:
            p=OUT/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight',facecolor='white');files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
        plt.close(fig);plots.append(dict(name=name,files=files))
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for i,ds in enumerate(['hc2','vidstg']):
        for j,m in enumerate(['v','t','s']):
            ax=axes[i,j];values=[stats[ds]['metrics'][a+'_'+m] for a in allarms]
            means=np.asarray([v['parent_macro'] for v in values])*100;ci=np.asarray([v['ci95'] for v in values])*100
            ax.bar(range(5),means,color=colors);ax.errorbar(range(5),means,yerr=np.maximum(np.array([means-ci[:,0],ci[:,1]-means]),0),fmt='none',ecolor='black',capsize=3)
            ax.set_xticks(range(5),labels,fontsize=8);ax.set_title(ds+' '+{'v':'vIoU','t':'tIoU','s':'fixed-GT spatial IoU'}[m]);ax.set_ylabel('%')
    savefig(fig,'complete_deployment_and_offline_oracle')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,ds in enumerate(['hc2','vidstg']):
        rr=[r for r in rows if r['dataset']==ds]
        for j,m in enumerate(['v','t']):
            ax=axes[i,j]
            for a,c in zip(allarms[1:],colors[1:]):
                values=np.sort([r[f'delta_{a}_minus_Native_{m}']*100 for r in rr]);ax.plot(range(1,33),values,label=a,color=c)
            ax.axhline(0,color='black',linewidth=.6);ax.set_title(ds+' '+m+'IoU complete parent effects');ax.set_ylabel('Change vs Native (pp)');ax.set_xlabel('Parent rank independently within each arm');ax.legend(fontsize=8)
    savefig(fig,'complete_negative_tails')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    pairs=list(itertools.combinations(ARMS,2))
    for ax,ds in zip(axes,['hc2','vidstg']):
        values=[stats[ds]['metrics'][f'delta_{b}_minus_{a}_v'] for a,b in pairs]
        mean=np.asarray([v['parent_macro'] for v in values])*100;ci=np.asarray([v['ci95'] for v in values])*100
        ax.errorbar(mean,range(6),xerr=np.maximum([mean-ci[:,0],ci[:,1]-mean],0),fmt='o',capsize=3,color='#657bc8');ax.axvline(0,color='black',linewidth=.6)
        ax.set_yticks(range(6),[b+' minus '+a for a,b in pairs],fontsize=8);ax.set_title(ds+' paired deployment contrasts');ax.set_xlabel('vIoU change (pp), paired 95% CI')
    savefig(fig,'all_paired_deployment_contrasts')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['hc2','vidstg']):
        rr=[r for r in rows if r['dataset']==ds];xx=[r['signal']['gradient_cosine'] for r in rr];yy=[r['delta_TemporalHead_minus_Native_t']*100 for r in rr]
        valid=[i for i,x in enumerate(xx) if x is not None];ax.scatter([xx[i] for i in valid],[yy[i] for i in valid],s=25,color='#657bc8')
        ax.axhline(0,color='black',linewidth=.6);ax.axvline(0,color='black',linewidth=.6);ax.set_xlabel('CPU source-head SSL / GT gradient cosine');ax.set_ylabel('Actual deployed tIoU change (pp)');ax.set_title(ds+' all 32 original parents')
    savefig(fig,'actual_temporal_signal_alignment')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,['hc2','vidstg']):
        rr=[r for r in rows if r['dataset']==ds];values=[np.mean([r['complete_temporal_cost']['complete_capture_fit_audit_reinsertion_seconds'] for r in rr]),np.mean([r['offline_head_cost']['actual_CPU_head_seconds'] for r in rr])]
        ax.bar(['GPU native capture + head\n+ recording/audit + reinsertion','Offline CPU GT-head fit'],values,color=['#657bc8','#969696']);ax.set_ylabel('Actual recorded seconds/query');ax.set_title(ds+' recorded cost, not a hardware-normalized speedup')
    savefig(fig,'actual_recorded_cost')
    fig,axes=plt.subplots(2,3,figsize=(13,7));chains=[];views=[];sample_checks=0
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
    from scripts.render_stvg_opd_revised_p2_root_v2 import signal
    gt={ds:truths(ds,definition(ds)) for ds in ['hc2','vidstg']}
    for ax,c in zip(axes.ravel(),cases):
        ds,at,q=c['dataset'],c['arrival'],c['query_ordinal'];r=next(r for r in rows if r['dataset']==ds and r['arrival']==at)
        z=load(NS/'formal'/ds/f'{at:05}.pt');orig,inp,binding=original(ds,q);row=read_row(ds,q)
        s=r['signal'];ax.plot(range(6),s['path_tIoU'],marker='o',label='Actual GPU head raw state tIoU')
        ax.axhline(s['shrunk_temporal_tIoU'],label='Actual eta .25 deployment',color='#657bc8',linestyle='--')
        ax.axhline(s['oracle_tIoU'],label='Offline GT head',color='#969696',linestyle=':');ax.set_title(ds+' '+c['kind'].replace('_',' '),fontsize=9);ax.set_xlabel('Fixed original AdamW step');ax.set_ylabel('tIoU');ax.legend(fontsize=6)
        if any(row['frame_ids'][pos] in gt[ds][0][q] for pos in orig['fit']['positions']):
            spatial,checks,origins=signal(orig['fit'],row,gt[ds][0][q]);spatial_status='complete_round_signal'
        else:spatial,checks,origins=[],0,{};spatial_status='no_GT_evaluable_admitted_position'
        sample_checks+=checks
        chains.append(dict(case=c,temporal=s,spatial=spatial,spatial_status=spatial_status,sample_GT_checks=checks,action_origins=origins,
            temporal_payload_sha256=sha(NS/'formal'/ds/f'{at:05}.pt'),original_spatial_payload_sha256=binding['sha256']))
        views.append(rgb(c,z,orig,inp,row,gt[ds][0][q],gt[ds][1][q],DEST/'private_RGB'))
    savefig(fig,'six_actual_signal_cases')
    write(OUT/'CASE_SELECTION.json',dict(status='complete',selection='deterministic post-hoc descriptive max/min temporal effect and least head-gradient alignment, distinct within each target',records=cases))
    write(OUT/'CASE_SIGNAL_CHAINS.json',dict(status='complete',records=chains,actual_spatial_sample_GT_checks=sample_checks,GT_after_all_deployments_sealed=True))
    write(DEST/'PRIVATE_RGB_READBACK.json',dict(status='complete',records=views,private_media_not_exported=True))
    write(DEST/'COMPLETION.json',dict(status='pending_actual_root_view_and_publication',plots=plots,plot_pairs=len(plots),cases=len(cases),
        actual_spatial_sample_GT_checks=sample_checks,private_RGB_metadata_sha256=sha(DEST/'PRIVATE_RGB_READBACK.json'),CPU_only=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'CASE_SELECTION.json',OUT/'CASE_SIGNAL_CHAINS.json']},paper_suite_complete=False,time=time.time()))


if __name__=='__main__':run()
