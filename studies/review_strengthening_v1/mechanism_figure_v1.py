"""Actual current predictions, fixed-budget interpolation and private GT figures.

No predictor/model/teacher execution and no new official scoring. Case selection
is explicitly post-hoc illustrative selection from already-audited outcomes.
"""
from review_common import *
import argparse, textwrap

DEST=D/'mechanism_v1'
PUBLIC=P/'mechanism_v1'


def prepare():
    assert not(DEST/'plan.json').exists()
    inputs=[D/'intent.json',R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json']
    for ds in COUNTS:
        inputs += [RUN/ds/'sealed.json',selected.D/f'components_{ds}_sealed.json',
                   selected.D/f'components_{ds}_reporting_receipt.json',
                   selected.D/'components'/ds/'metrics.json',
                   D/'matched_cpu'/ds/'reporting_receipt.json']
        job=read(RUN/'plan.json')['jobs'][ds]
        inputs += [PROVIDER/job['manifest'],PROVIDER/job['labels']]
    plan=dict(schema='current_mechanism_actual_RGB_GT_v1',
              current_method='Shared-corrected-box ROI features, kappa0; first auxiliary DP removed, final interval trajectory DP retained.',
              selection='Three post-hoc illustrative extremes, not representative prevalence: HC largest positive Full-Source unobserved fixed-support sIoU among positive v changes; Vid largest positive Full-Source t change among positive v changes; Vid largest positive Full-FinalIndependent v change. Deterministic ID tie breaks. Require >=3 actually annotated unobserved native sampled frames. DP case must have distinct saved independent/final boxes.',
              frames='Five actual native sampled RGB frames per case: first/last expert observation plus three evenly-spaced GT-supported unobserved nodes; DP case instead uses the three largest saved independent-vs-final box-coordinate differences among GT-supported unobserved nodes. All 8 observation locations shown on timeline. No extra expert observation or interpolated GT.',
              rows='Source with native original-MAP interval; fixed-eight original center/logsize interpolation with common current interval; fixed-candidate independent control and current Ours with identical current interval. Actual GT boxes overlaid only where annotation exists, never fabricated. Timeline separately shows native Source/current/GT boundaries. Source boxes outside its own interval are explicitly marked inactive.',
              isolation='Read-only saved arrays and actual per-case cold selected-core replay; no new model/Source/expert/optimizer or official GT scoring. Offline GT is used only for disclosed illustrative selection, frame annotation support and rendering. Competitor interpolation receives no our DP. SAM2 is pending and is not fabricated or shown as completed.',
              privacy='All query IDs, RGB, captions, frame IDs, predicted/GT coordinates and figure media remain private; public only protocol, scalar deltas, data/code/PNG/PDF hashes and root visual inspection receipt.',
              code_sha256={str(f.relative_to(R)):sha(f) for f in [Path(__file__),S/'review_common.py',R/'stvg/decode.py',R/'studies/shared_corrected_roi_v1/core.py',R/'stvg/metrics_base.py']},
              input_sha256={str(f):sha(f) for f in inputs})
    plan['fingerprint']=digest(plan);atomic(DEST/'plan.json',plan)
    projection={k:v for k,v in plan.items() if k not in ['input_sha256','fingerprint']}
    projection.update(private_plan_fingerprint=plan['fingerprint'],input_sha256_values=list(plan['input_sha256'].values()),input_locations_private=True)
    atomic(PUBLIC/'protocol.json',projection)


def validate_plan():
    plan=read(DEST/'plan.json');assert plan['fingerprint']==digest({k:v for k,v in plan.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in plan[field].items():assert sha(R/f)==h,f
    selected.validate();return plan


def render():
    import numpy as np,torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from stvg.decode import decode
    from stvg.metrics_base import xyxy
    torch.set_num_threads(1);plan=validate_plan()
    assert not(DEST/'render_receipt.json').exists()
    metrics={ds:{x['id']:x for x in read(selected.D/'components'/ds/'metrics.json')['private_rows']} for ds in COUNTS}
    rows={ds:{x['id']:x for x in records(ds)} for ds in COUNTS}
    labels={ds:selected.labels(ds) for ds in COUNTS}
    choices=[]
    specs=[('hc2','Between-observation recovery','Source','unobserved_actual_sIoU'),
           ('vidstg','Temporal boundary update','Source','t'),
           ('vidstg','Final DP joint path selection','FinalIndependent','v')]
    for ds,title,base,field in specs:
        def delta(x):
            a,b=x['values']['FullMAP'].get(field),x['values'][base].get(field)
            return -float('inf') if a is None or b is None else a-b
        ranked=sorted(metrics[ds].values(),key=lambda x:(-delta(x),x['id']))
        for metric in ranked:
            key=metric['id'];v=metric['values']
            if delta(metric)<=0 or v['FullMAP']['v']<=v[base]['v']:continue
            pe,ce=paths(ds,key);gt=labels[ds][key];support={int(f) for f in gt['boxes']}
            with np.load(selected.D/ce['path']) as z:
                ids=z['frame_ids'].copy();pos=z['positions8'].tolist();unobs=[i for i,f in enumerate(ids) if int(f) in support and i not in pos]
                if len(unobs)<3:continue
                if base=='FinalIndependent':
                    dif=np.max(abs(z['FullMAP']-z['FinalIndependent']),axis=1)
                    if not any(dif[i]>0 for i in unobs):continue
                    picked=sorted(sorted(unobs,key=lambda i:(-dif[i],i))[:3])
                else:picked=[unobs[i] for i in np.linspace(0,len(unobs)-1,3,dtype=int)]
            shown=sorted(set([min(pos),max(pos),*picked]));assert len(shown)==5
            choices.append(dict(dataset=ds,id=key,title=title,comparison_base=base,
                                shown_positions=shown,selection_delta_field=field,
                                values=v,selection='Post-hoc positive extreme under frozen rule; not representative'))
            break
        else:raise RuntimeError('No actual case satisfies frozen illustrative rule; no fabricated example')
    atomic(DEST/'selection.json',choices)
    core=selected.parent.load_old('_review_mechanism_selected_core',R/'studies/shared_corrected_roi_v1/core.py')
    proofs=[]
    for ci,case in enumerate(choices,1):
        ds,key=case['dataset'],case['id'];row=rows[ds][key];gt=labels[ds][key]
        pe,ce=paths(ds,key);frames,ids=decode(row['metadata'],ds)
        with np.load(RUN/ds/pe['path']) as old,np.load(selected.D/ce['path']) as current:
            a={name:current[name].copy() for name in ['Source','FullMAP','FinalIndependent','Source_interval','FullMAP_interval','frame_ids','positions8']}
            a['Interpolation']=old['Interpolation'].copy()
        assert ids==a['frame_ids'].tolist()
        cold,provenance=core.derive('components',ds,dict(key=key,row=row))
        for name in ['FullMAP','FinalIndependent','FullMAP_interval']:
            assert np.array_equal(cold[name],a[name])
        fig=plt.figure(figsize=(20,13),layout='constrained')
        grid=fig.add_gridspec(6,5,height_ratios=[.55,2.5,2.5,2.5,2.5,1.2])
        header=fig.add_subplot(grid[0,:]);header.axis('off')
        caption='\n'.join(textwrap.wrap(row['metadata']['caption'],135))
        header.text(0,.95,f'Case {ci}: {case["title"]} | Query: {caption}',va='top',fontsize=12,transform=header.transAxes)
        methods=[('Source','Native Source','#168bd2'),('Interpolation','Eight-frame interpolation','#b45adb'),
                 ('FinalIndependent','Independent candidate choice','#e08a12'),('FullMAP','Ours: final DP','#e84c35')]
        for ri,(name,label,color) in enumerate(methods,1):
            iv=a['Source_interval'] if name=='Source' else a['FullMAP_interval']
            for col,t in enumerate(case['shown_positions']):
                ax=fig.add_subplot(grid[ri,col]);ax.imshow(frames[t]);h,w=frames[t].shape[:2]
                box=xyxy(a[name][t],w,h);active=int(iv[0])<=ids[t]<int(iv[1])
                ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,
                                       edgecolor=color,lw=2.2,linestyle='-' if active else '--'))
                truth=gt['boxes'].get(str(ids[t]))
                if truth is not None:
                    x,y,x1,y1=truth;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor='#20d04a',lw=2))
                else:ax.text(.02,.03,'GT unavailable',color='white',fontsize=8,transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.6))
                if not active:ax.text(.02,.94,'Outside predicted event',color='white',fontsize=8,va='top',transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.6))
                if ri==1:ax.set_title(f'Physical frame {ids[t]} | '+('Expert observed' if t in a['positions8'] else 'Unobserved'),fontsize=10)
                if col==0:ax.text(-.02,.5,label,rotation=90,va='center',ha='right',fontsize=11,color=color,transform=ax.transAxes)
                ax.set_xticks([]);ax.set_yticks([])
        ax=fig.add_subplot(grid[5,:]);fps=row['metadata']['fps'];origin=ids[0]
        convert=lambda x:(np.asarray(x)-origin)/fps
        ax.scatter(convert([ids[int(t)] for t in a['positions8']]),np.full(8,3.4),marker='v',color='purple',s=42,label='Same eight expert observations')
        for iv,y,color,label in [(a['Source_interval'],2.3,'#168bd2','Native Source event'),(a['FullMAP_interval'],1.3,'#e84c35','Current event: interpolation / independent / Ours'),(gt['interval'],.3,'#20b040','GT event')]:
            ax.broken_barh([((iv[0]-origin)/fps,(iv[1]-iv[0])/fps)],(y,.6),facecolors=color,label=label)
        ax.set_ylim(0,4);ax.set_xlim(convert([ids[0],ids[-1]+1]));ax.set_yticks([]);ax.set_xlabel('Seconds from first native sampled physical frame')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.35),ncol=2,fontsize=9)
        fig.suptitle('Actual saved predictions | green boxes: annotated GT | post-hoc illustrative case, not prevalence',fontsize=14)
        png=DEST/f'case_{ci}.png';pdf=DEST/f'case_{ci}.pdf';fig.savefig(png,dpi=130);fig.savefig(pdf);plt.close(fig)
        vals=case['values'];base=case['comparison_base']
        proofs.append(dict(case=ci,dataset=ds,capability=case['title'],
                           selection='Frozen post-hoc positive extreme; not representative prevalence',
                           delta_pp={k:100*(vals['FullMAP'][k]-vals[base][k]) if vals['FullMAP'].get(k) is not None and vals[base].get(k) is not None else None for k in ['t','v','unobserved_actual_sIoU']},
                           parent_prediction_sha256=pe['sha256'],current_prediction_sha256=ce['sha256'],
                           same_candidate_final_independent_cold_arrays_exact=True,
                           same_current_interval_for_spatial_controls=True,GT_overlay_actual_support_only=True,
                           actual_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                           PNG_sha256=sha(png),PDF_sha256=sha(pdf)))
    receipt=dict(status='rendered_private_pending_root_visual_inspection',cases=proofs,
                 code_sha256=sha(Path(__file__)),plan_fingerprint=plan['fingerprint'],
                 new_Source=0,new_expert=0,new_model=0,new_optimizer=0,new_GT_scoring=0,
                 private_GT_overlay=True,actual_SAM2_predictions=False,worker=identity())
    atomic(DEST/'render_receipt.json',receipt);atomic(PUBLIC/'render_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],cases=3,private_RGB_figures=True,new_model=0)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','render'])
    globals()[a.parse_args().command]()
