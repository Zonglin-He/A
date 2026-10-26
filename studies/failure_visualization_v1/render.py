"""Frozen post-hoc failure selection and actual RGB/GT rendering; no inference."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, time, json, hashlib, textwrap, argparse, shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'studies/review_strengthening_v1'))
import review_common as review
from review_common import read,sha,atomic,digest,identity,records,RUN,PROVIDER,selected,COUNTS
S=Path(__file__).parent;D=R/'outputs/failure_visualization_v1';P=R/'results/failure_visualization_v1'
SYNC=R/'outputs/shared_corrected_roi_sync_v1'

def prepare():
    assert not(D/'plan.json').exists();selected.validate()
    inputs=[R/'CURRENT_METHOD_v2.json',R/'profiles/shared_corrected_roi_v1.json',
            R/'outputs/temporal_pooling_isolation_v1/complete_published.json']
    for ds in COUNTS:
        inputs += [RUN/ds/'sealed.json',selected.D/f'components_{ds}_sealed.json',
                   selected.D/f'components_{ds}_reporting_receipt.json',selected.D/'components'/ds/'metrics.json',
                   review.D/'matched_cpu'/ds/'sealed.json',review.D/'matched_cpu'/ds/'reporting_receipt.json',
                   review.D/'matched_cpu'/ds/'metrics.json',SYNC/f'spatial_{ds}_sealed.json',
                   SYNC/f'spatial_{ds}_reporting_receipt.json',SYNC/'spatial'/ds/'metrics.json']
        job=read(RUN/'plan.json')['jobs'][ds];inputs += [PROVIDER/job['manifest'],PROVIDER/job['labels']]
    code=[Path(__file__),R/'studies/review_strengthening_v1/review_common.py',
          R/'studies/shared_corrected_roi_v1/core.py',R/'stvg/decode.py',R/'stvg/metrics.py',
          R/'stvg/metrics_base.py',R/'stvg/paper_panel.py',R/'stvg/prototype.py',R/'stvg/spatial.py']
    plan=dict(schema='actual_failure_visualizations_v1',authorization='Latest direct user: 我还要有失败案例的可视化',
              current_method='Shared-corrected-box appearance ROI, kappa0/beta1, no first auxiliary DP, retained final interval DP.',
              selection='Four deterministic post-hoc negative extremes, not representative prevalence, unique query and parent within a dataset. Break all metric ties by dataset then query ID. (1) HC minimum Full-nativeSource t, both delta t and v<-5pp. (2) Vid minimum Full-nativeSource unobserved fixed-support sIoU, with v<-5pp and EXACT identical Source/current intervals. (3) Either domain minimum Full-FinalIndependent v<-5pp, with exactly shared interval/candidates. (4) Either domain minimum correctedROI-SourceROI t, both t/v<-5pp; exact main geometry/DP parameters fixed. All cases require >=3 GT-supported unobserved native frames. Spatial/DP cases additionally require >=3 active GT-supported unobserved frames with negative per-frame IoU change.',
              frames='Five native sampled RGB frames: first/last of the same8expert positions plus three GT-supported unobserved frames. Spatial/DP select three worst per-frame IoU differences, temporal/ROI prefer GT-supported unobserved frames covered by baseline event but excluded by current event, fill remaining positions by even spread of other supported unobserved frames. This is disclosed GT-based illustrative frame selection, not benchmark scoring or online selection.',
              rows='Cases1-3: native Source/native interval, original same8 center-logsize interpolation/common current interval, same-candidate independent/common current interval, Ours/finalDP/current interval. Case4 replaces independent row with Source-box appearance using full main geometry and its own interval. Green is actual GT only; inactive prediction dashed and labeled; no invented missing annotations. Timeline shows all8anchors and relevant native/SourceROI/current/GT intervals.',
              scientific_scope='Saved full official metrics only. Cold current Full/independent replay and exact original interpolation per case; SourceROI own posterior/MAP/finalDP replay for case4. Offline GT only for annotated-frame support, frame IoU used to select illustrative positions, and overlays; zero new official GT scoring/model/Source/expert/optimizer. General temporal case includes native-to-joint temporal readout change and is not isolated posterior attribution.',
              privacy='New failure PNG/PDF, RGB, caption, IDs and coordinates remain private/local. Earlier user authorized only the existing three mechanism PNG/PDF pairs for GitHub; no automatic extension of that media exception. Public code/protocol/signed scalars/figure hashes/inspection receipts only.',
              limits='Single CPU14-15 nice19 idleIO one-thread consumer, <=2 CPU jobs with CG/gamma; no GPU. RAMavailable>=8GiB/diskfree>=80GiB. Original CG/gamma unmodified and corruption remains paused.',
              code_sha256={str(f.relative_to(R)):sha(f)for f in code},
              input_sha256={str(f):sha(f)for f in inputs})
    plan['fingerprint']=digest(plan);atomic(D/'plan.json',plan)
    atomic(P/'protocol.json',{**{k:v for k,v in plan.items()if k not in ['input_sha256','fingerprint']},
                             'private_plan_fingerprint':plan['fingerprint'],'input_sha256_values':list(plan['input_sha256'].values())})
    for f in code:
        q=D/'source_snapshot'/f.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
    print(plan['fingerprint'],flush=True)

def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    selected.validate()
    for ds in COUNTS:
        for a in [read(selected.D/f'components_{ds}_reporting_receipt.json'),read(review.D/'matched_cpu'/ds/'reporting_receipt.json'),read(SYNC/f'spatial_{ds}_reporting_receipt.json')]:
            assert a['status']=='verified_whole_domain'and a['all_means_recomputed']
    return p

def resources():
    assert shutil.disk_usage(R).free>=80*2**30
    assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
    active=0
    for root in [R/'outputs/shared_corrected_roi_CG_v1',R/'outputs/review_gamma_full_v1']:
        for name in ['hc2_CPU_child.json','vidstg_CPU_child.json']:
            p=root/name
            if p.exists()and not selected.exited(read(p)):active+=1
    assert active<2,'Reserve one CPU slot; no interference with prior workers'

def checked_arrays(ds,key):
    import numpy as np
    pe,ce=review.paths(ds,key)
    with np.load(selected.D/ce['path'])as z:
        a={k:z[k].copy()for k in ['Source','FullMAP','FinalIndependent','Source_interval','FullMAP_interval','FinalIndependent_interval','frame_ids','positions8']}
    me=read(review.D/'matched_cpu'/ds/'sealed.json')['files'][key]
    assert sha(review.D/me['path'])==me['sha256']and sha(review.D/me['receipt_path'])==me['receipt_sha256']
    with np.load(review.D/me['path'])as z:
        a['Interpolation']=z['ExpertInterpolation'].copy();a['Interpolation_interval']=z['ExpertInterpolation_interval'].copy()
        assert np.array_equal(a['FullMAP'],z['Ours'])and np.array_equal(a['FullMAP_interval'],z['Ours_interval'])
    e=read(SYNC/f'spatial_{ds}_sealed.json')['files'][key]
    assert sha(SYNC/e['path'])==e['sha256']and sha(SYNC/e['receipt_path'])==e['receipt_sha256']
    with np.load(SYNC/e['path'])as z:
        a['SourceROI']=z['AuxSourceROI'].copy();a['SourceROI_interval']=z['AuxSourceROI_interval'].copy()
        assert np.array_equal(a['FullMAP'],z['FullMAP'])and np.array_equal(a['FullMAP_interval'],z['FullMAP_interval'])
    return a,dict(parent_prediction_sha256=pe['sha256'],current_prediction_sha256=ce['sha256'],
                  matched_prediction_sha256=me['sha256'],structural_prediction_sha256=e['sha256'])

def select():
    import numpy as np
    from stvg.metrics_base import xyxy
    from stvg.metrics import box_iou
    metrics={ds:{x['id']:x for x in read(selected.D/'components'/ds/'metrics.json')['private_rows']}for ds in COUNTS}
    structural={ds:{x['id']:x for x in read(SYNC/'spatial'/ds/'metrics.json')['private_rows']}for ds in COUNTS}
    matched={ds:{x['id']:x for x in read(review.D/'matched_cpu'/ds/'metrics.json')['private_rows']}for ds in COUNTS}
    rows={ds:{x['id']:x for x in records(ds)}for ds in COUNTS};truth={ds:selected.labels(ds)for ds in COUNTS}
    specs=[dict(kind='temporal',datasets=['hc2'],title='Temporal event update failure',base='Source',field='t'),
           dict(kind='spatial',datasets=['vidstg'],title='Unobserved spatial trajectory failure',base='Source',field='unobserved_actual_sIoU'),
           dict(kind='final_DP',datasets=list(COUNTS),title='Final DP hurts a correct independent choice',base='FinalIndependent',field='v'),
           dict(kind='pooling_ROI',datasets=list(COUNTS),title='Corrected-ROI temporal evidence failure',base='SourceROI',field='t')]
    chosen=[];used=set()
    for spec in specs:
        ranking=[]
        for ds in spec['datasets']:
            for key,m in metrics[ds].items():
                v=m['values'];base=structural[ds][key]['values']['AuxSourceROI']if spec['base']=='SourceROI'else v[spec['base']]
                current=v['FullMAP'];d=current.get(spec['field']);b=base.get(spec['field'])
                if d is None or b is None:continue
                delta=d-b
                if delta>=-.05 or current['v']-base['v']>=-.05:continue
                if spec['kind']=='spatial'and current['t']!=base['t']:continue
                if(ds,m['parent'])in used:continue
                ranking.append((delta,ds,key))
        for delta,ds,key in sorted(ranking):
            m=metrics[ds][key];a,proof=checked_arrays(ds,key);gt=truth[ds][key];row=rows[ds][key]
            ids=a['frame_ids'];pos=set(a['positions8'].tolist());iv=a['FullMAP_interval'];base_iv=a[spec['base']+'_interval']
            if spec['kind']=='spatial'and not np.array_equal(base_iv,iv):continue
            if spec['kind']=='final_DP':assert np.array_equal(base_iv,iv)
            unobs=[i for i,f in enumerate(ids)if str(int(f))in gt['boxes']and i not in pos]
            if len(unobs)<3:continue
            frame_delta={}
            if spec['kind']in ['spatial','final_DP']:
                worst=[]
                for i in unobs:
                    if not(iv[0]<=ids[i]<iv[1]):continue
                    x=xyxy(a['FullMAP'][i],row['metadata']['width'],row['metadata']['height'])
                    y=xyxy(a[spec['base']][i],row['metadata']['width'],row['metadata']['height'])
                    if ds=='hc2':x=np.maximum(x,0);y=np.maximum(y,0)
                    g=gt['boxes'][str(int(ids[i]))];change=box_iou(x,g)-box_iou(y,g)
                    if change<-1e-10:worst.append((change,i))
                if len(worst)<3:continue
                picked=sorted(i for _,i in sorted(worst)[:3]);frame_delta={str(i):float(change)for change,i in worst if i in picked}
            else:
                missed=[i for i in unobs if base_iv[0]<=ids[i]<base_iv[1]and not(iv[0]<=ids[i]<iv[1])]
                pool=missed if len(missed)>=3 else unobs
                picked=[pool[i]for i in np.linspace(0,len(pool)-1,3,dtype=int)]
            shown=sorted({min(pos),max(pos),*picked});assert len(shown)==5
            v=dict(m['values']);v['Interpolation']=matched[ds][key]['values']['ExpertInterpolation'];v['SourceROI']=structural[ds][key]['values']['AuxSourceROI']
            chosen.append(dict(**spec,dataset=ds,id=key,parent=m['parent'],shown_positions=shown,chosen_unobserved_positions=picked,
                               values=v,proof=proof,selected_delta=float(delta),frame_delta_IoU=frame_delta,
                               selection='Frozen deterministic post-hoc negative extreme; not prevalence or an unbiased sample'))
            used.add((ds,m['parent']));break
        else:raise RuntimeError('No actual failure meets frozen rule; preserve, do not fabricate')
    atomic(D/'selection.json',chosen)
    return chosen,rows,truth

def render():
    import numpy as np,torch
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from stvg.decode import decode
    from stvg.metrics_base import xyxy
    from stvg.paper_panel import interpolation
    from stvg.spatial import SpatialGeometry,evidence_dp
    from stvg.prototype import one_step
    from stvg.current_shared_roi import map_interval
    plan=validate();resources();torch.set_num_threads(1);assert not(D/'render_receipt.json').exists()
    who=identity();atomic(D/'worker.json',who);started=time.perf_counter()
    choices,rows,truth=select();core=selected.parent.load_old('_failure_selected_core',R/'studies/shared_corrected_roi_v1/core.py')
    proofs=[]
    for n,case in enumerate(choices,1):
        resources();ds,key=case['dataset'],case['id'];row=rows[ds][key];gt=truth[ds][key];a,proof=checked_arrays(ds,key)
        cold,_=core.derive('components',ds,dict(key=key,row=row))
        for k in ['Source','FullMAP','FinalIndependent','Source_interval','FullMAP_interval','FinalIndependent_interval']:assert np.array_equal(cold[k],a[k])
        pe,_=review.paths(ds,key)
        with np.load(RUN/ds/pe['path'])as old:
            inter=interpolation(old['SharedInit'],old['frame_ids'],old['positions8'],old['native_boxes'])
            assert np.array_equal(inter,old['Interpolation'])and np.array_equal(inter,a['Interpolation'])
        if case['kind']=='pooling_ROI':
            root=selected.PREV/'spatial';e=read(root/f'{ds}_sealed.json')['files'][key];c,_=selected.checked_payload(root,e)
            q,_=one_step(c['features']['AuxSourceROI'],c['q0'],kappa=0,beta=1);iv=map_interval(c['ids'],q)
            g=SpatialGeometry.from_cached_geometry(c['geometries']['main']);b=evidence_dp(g.for_interval(iv))['final'].numpy()
            assert np.array_equal(b,a['SourceROI'])and np.array_equal(iv,a['SourceROI_interval'])
        frames,ids=decode(row['metadata'],ds);assert ids==a['frame_ids'].tolist()
        png=D/f'failure_{n:02d}_{case["kind"]}.png';pdf=png.with_suffix('.pdf')
        methods=[('Source','Native Source','#167fc1'),('Interpolation','Eight-frame interpolation','#9a4aba'),
                 ('SourceROI','Source-box appearance','#d88813')if case['kind']=='pooling_ROI'else('FinalIndependent','Independent candidate choice','#d88813'),
                 ('FullMAP','Ours: final DP','#dc3d30')]
        fig=plt.figure(figsize=(24,15),layout='constrained')
        grid=fig.add_gridspec(7,6,width_ratios=[1.9,3,3,3,3,3],height_ratios=[1,.34,2.2,2.2,2.2,2.2,1.35])
        header=fig.add_subplot(grid[0,:]);header.axis('off')
        v=case['values'];base=case['base'];dt=100*(v['FullMAP']['t']-v[base]['t']);dv=100*(v['FullMAP']['v']-v[base]['v'])
        cap='\n'.join(textwrap.wrap(row['metadata']['caption'],165))
        title=f'Failure {n}: {case["title"]}  |  {ds}\nQuery: {cap}\nOurs minus {base}: delta tIoU {dt:+.2f} pp; delta vIoU {dv:+.2f} pp.  Post-hoc illustrative failure; not representative prevalence.'
        header.text(0,.98,title,va='top',fontsize=13,transform=header.transAxes)
        for col,t in enumerate(case['shown_positions'],1):
            ax=fig.add_subplot(grid[1,col]);ax.axis('off');frame=int(ids[t]);label=f'Frame {frame} | '+('Expert observed'if t in a['positions8']else'Unobserved')
            if str(t)in case['frame_delta_IoU']:label+=f"\nframe IoU change {100*case['frame_delta_IoU'][str(t)]:+.1f} pp"
            ax.text(.5,.5,label,ha='center',va='center',fontsize=10)
        for ri,(name,label,color)in enumerate(methods,2):
            la=fig.add_subplot(grid[ri,0]);la.axis('off');mm=v[name]
            la.text(.98,.5,f'{label}\ntIoU {100*mm["t"]:.2f}%\nvIoU {100*mm["v"]:.2f}%',ha='right',va='center',fontsize=12,color=color)
            iv=a[name+'_interval']
            for col,t in enumerate(case['shown_positions'],1):
                ax=fig.add_subplot(grid[ri,col]);ax.imshow(frames[t]);h,w=frames[t].shape[:2]
                box=xyxy(a[name][t],w,h);active=iv[0]<=ids[t]<iv[1]
                predcolor=color if active else'#6c6c6c'
                ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor=predcolor,lw=2.4,linestyle='-'if active else'--'))
                truthbox=gt['boxes'].get(str(ids[t]))
                if truthbox is not None:
                    x,y,x1,y1=truthbox;ax.add_patch(Rectangle((x,y),x1-x,y1-y,fill=False,edgecolor='#25db4c',lw=2.5))
                else:ax.text(.02,.05,'GT unavailable',color='white',fontsize=9,transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.65))
                if not active:ax.text(.02,.96,'Outside predicted event',va='top',color='white',fontsize=9,transform=ax.transAxes,bbox=dict(facecolor='black',alpha=.65))
                ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_xticks([]);ax.set_yticks([])
        la=fig.add_subplot(grid[6,0]);la.axis('off');la.text(.98,.55,'Event intervals\n+ same eight prompts',ha='right',va='center',fontsize=11)
        ax=fig.add_subplot(grid[6,1:]);fps=row['metadata']['fps'];origin=ids[0]
        spans=[(a['Source_interval'],'Native Source','#167fc1')]
        if case['kind']=='pooling_ROI':spans.append((a['SourceROI_interval'],'Source-box appearance','#d88813'))
        spans += [(a['FullMAP_interval'],'Current event: interpolation / independent / Ours'if case['kind']!='pooling_ROI'else'Corrected-box appearance (Ours)','#dc3d30'),(gt['interval'],'Annotated GT event','#25ad43')]
        for i,(iv,label,color)in enumerate(spans):
            y=len(spans)-1-i;ax.broken_barh([((iv[0]-origin)/fps,(iv[1]-iv[0])/fps)],(y,.55),facecolors=color)
        obs=np.array([ids[int(t)]for t in a['positions8']]);ax.scatter((obs-origin)/fps,np.full(8,len(spans)+.05),marker='v',color='#73349c',s=38)
        ax.set_yticks([i+.27 for i in range(len(spans))],[x[1]for x in spans[::-1]],fontsize=9)
        ax.set_ylim(-.1,len(spans)+.6);ax.set_xlim(0,(ids[-1]+1-origin)/fps);ax.set_xlabel('Seconds from first native sampled physical frame',fontsize=10)
        handles=[Line2D([0],[0],color='#25db4c',lw=2.5,label='Annotated GT box'),Line2D([0],[0],color='#6c6c6c',lw=2,ls='--',label='Prediction inactive outside its event'),Line2D([0],[0],marker='v',color='#73349c',ls='',label='Same eight expert observations')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=11)
        fig.savefig(png,dpi=140);fig.savefig(pdf);plt.close(fig)
        public=dict(case=n,dataset=ds,failure_type=case['kind'],selection='Frozen post-hoc negative extreme, unique parent; GT-supported failure frame selection. Not prevalence.',
                    delta_pp={k:100*(v['FullMAP'][k]-v[base][k])if v['FullMAP'].get(k)is not None and v[base].get(k)is not None else None for k in ['t','v','unobserved_actual_sIoU']},
                    baseline=base,PNG_sha256=sha(png),PDF_sha256=sha(pdf),actual_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                    actual_GT_support_only=True,full_and_independent_cold_replay_exact=True,original_interpolation_exact=True,
                    eight_prompt_markers=True,**proof)
        proofs.append(public);print(json.dumps(dict(case=n,dataset=ds,type=case['kind'],delta_pp=public['delta_pp'])),flush=True)
        del frames
    assert not torch.cuda.is_initialized()
    rec=dict(status='rendered_four_actual_failures_pending_root_image_inspection',cases=proofs,plan_fingerprint=plan['fingerprint'],
             code_sha256=sha(Path(__file__)),worker=who,elapsed_wall_seconds=time.perf_counter()-started,
             new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_official_GT_scoring=0,
             offline_GT_for_illustrative_selection_and_overlay=True,EdgeTAM_predictions_included=False,figure_media_public=False)
    atomic(D/'render_receipt.json',rec);atomic(P/'render_receipt.json',rec)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','render']);globals()[p.parse_args().command]()
