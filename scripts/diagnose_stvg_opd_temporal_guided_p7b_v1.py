"""Postseal, offline support/negative-tail diagnosis; no adaptation or gate selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *
from scripts.score_stvg_opd_temporal_capture_p7a_v1 import rows_read,rows_write,array,near,context_truth

def paired(rr,field):
    groups=collections.defaultdict(list)
    for r in rr:groups[r['parent_id']].append(r[field])
    values=[math.fsum(v for v in xs if v is not None)/sum(v is not None for v in xs)
        for xs in groups.values() if any(v is not None for v in xs)]
    if not values:return dict(parents=0,mean=None,ci95=None,undefined_parents=len(groups))
    x=np.array(values);gen=np.random.default_rng(20261012);indices=gen.integers(len(x),size=(10000,len(x)))
    return dict(parents=len(x),mean=float(x.mean()),ci95=np.quantile(x[indices].mean(1),[.025,.975]).tolist(),
        undefined_parents=len(groups)-len(values),parent_values=x.tolist(),harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)))

def run():
    from scripts.score_stvg_opd_temporal_guided_p7b_v1 import seal
    seal();assert read(B/'ROOT_READBACK.json')['status']=='pass'
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    arows={r['public_cell']:r for r in rows_read(OUT/'P7_A/ROWS.jsonl.gz')}
    rows=rows_read(OUT/'P7_B/ROWS.jsonl.gz');gt,spans=context_truth();d=read(BASE/'DESIGN_LOCK.json');extra=[];extra_checks=0;controls=collections.Counter()
    import torch
    rowmap={r['public_cell']:r for r in rows}
    for r in d['records']:
        ds=r['dataset'];raw=read_row(ds,r['query_ordinal']);truth=gt[ds][r['query_ordinal']];span=spans[ds][r['query_ordinal']]
        fp=ROOT/r['original_Full']['path'];assert sha(fp)==r['original_Full']['sha256'];before=load(fp)['fit']['before']
        old=normalized(load(ROOT/r['original_Full']['input_path']))['expert'];new=load(B/'expert_I1'/f"{r['capture_index']:05}.pt")['expert']
        if old['anchors']['single4']==new['anchors']['single4']:
            prior=load(fp)['fit'];cur=load(B/'formal'/f"{r['public_cell']:05}"/'C_temporal_guided_OPD_repeat0.pt')['fit']
            assert torch.equal(prior['final'],cur['final']) and all(torch.equal(v,cur['state'][k]) for k,v in prior['state'].items())
            controls['exact_same_evidence_controls']+=1;controls['nonempty_same_evidence_controls']+=bool(old['anchors']['single4'])
        saved=load(A/'readouts'/f"{r['public_cell']:05}.pt");metrics,_,_=dense_metric(array(before),raw,truth,span,saved['intervals']['T1'],ds)
        extra_checks+=near(metrics,official(array(before),raw,truth,span,saved['intervals']['T1'],ds));assert sha(fp)==r['original_Full']['sha256']
        rr={k:r[k] for k in ['public_cell','dataset','parent_id','order','previous_P6']}
        for k,v in metrics.items():
            rr['Full_Before_T1_'+k]=v
            for arm,prefix in [('B_posthoc','B'),('C_temporal_guided_OPD','C'),('D_temporal_guided_Direct','D')]:rr[prefix+'_minus_Full_Before_'+k]=rowmap[r['public_cell']][arm+'_'+k]-v
        def quality(ex):
            vals=[]
            for v in ex['anchors']['single4']:
                f=raw['frame_ids'][v['position']]
                if f in truth and span[0]<=f<span[1]:
                    pred=xyxy(np.array([v['box']]),raw['input']['width'],raw['input']['height'])[0]
                    if ds=='hc2':pred=np.maximum(pred,0)
                    vals.append(float(box_iou(pred,np.asarray(truth[f]))))
            return float(np.mean(vals)) if vals else None
        qo,qn=quality(old),quality(new);extra_checks+=near(qn,rowmap[r['public_cell']]['I1_offline_admitted_expert_quality'])
        rr.update(I0_offline_admitted_expert_quality=qo,I1_minus_I0_offline_expert_quality=None if qo is None or qn is None else qn-qo,
            I0_admitted_positions=len(old['anchors']['single4']),I1_minus_I0_admitted_positions=len(new['anchors']['single4'])-len(old['anchors']['single4']))
        extra.append(rr);rowmap[r['public_cell']].update({k:v for k,v in rr.items() if k not in ['public_cell','dataset','parent_id','order','previous_P6']})
    rows_write(OUT/'P7_B/DIAGNOSTIC_EXTRA_ROWS.jsonl.gz',extra)
    predicates={
        'all_added96':lambda r:True,
        'gained_annotated_event_support':lambda r:arows[r['public_cell']]['gained_event_frames']>0,
        'lost_annotated_event_support':lambda r:arows[r['public_cell']]['lost_event_frames']>0,
        'increased_event_outside_duration':lambda r:arows[r['public_cell']]['T1_outside_event_duration']>arows[r['public_cell']]['T0_outside_event_duration'],
        'new_observation_outside_event':lambda r:r['I1_visited_outside_event']>0,
        'posthoc_temporal_harm_over5pp':lambda r:arows[r['public_cell']]['delta_T_OPD_v']<-.05,
        'coupling_harm_over5pp':lambda r:r['coupling_C_minus_B_v']<-.05}
    fields=['coupling_C_minus_B_v','D_minus_B_v','C_minus_D_v','C_minus_B_common_unobserved_real_IoU',
        'C_minus_B_gained_event_unobserved_real_IoU','new_positions_not_visited_before','I1_visited_inside_event',
        'I1_visited_outside_event','I1_offline_admitted_expert_quality','I0_offline_admitted_expert_quality',
        'I1_minus_I0_offline_expert_quality','I1_minus_I0_admitted_positions','Full_Before_T1_s',
        'B_minus_Full_Before_s','C_minus_Full_Before_s','D_minus_Full_Before_s']
    groups={};selected=[];scalar_checks=0
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds and not r['previous_P6']];groups[ds]={}
        for name,predicate in predicates.items():
            chosen=[r for r in rr if predicate(r)]
            groups[ds][name]=dict(cells=len(chosen),distinct_parents=len({r['parent_id'] for r in chosen}),
                statistics={f:paired(chosen,f) for f in fields})
        by=collections.defaultdict(list)
        for r in rr:by[r['parent_id']].append(r)
        order=sorted(by,key=lambda p:np.mean([r['coupling_C_minus_B_v'] for r in by[p]]))
        for label,pid in [('worst_coupling_parent',order[0]),('best_coupling_parent',order[-1])]:
            cell=next(r for r in by[pid] if r['order']=='order1')
            selected.append(dict(dataset=ds,label=label,parent_id=pid,public_cell=cell['public_cell'],
                paired_parent_coupling=float(np.mean([r['coupling_C_minus_B_v'] for r in by[pid]])),row=cell))
    write(OUT/'P7_B/DIAGNOSTIC_GROUPS.json',dict(status='measured_after_complete_B_root',scope='exploratory overlapping offline GT support groups; never select observations or gates',
        groups=groups,bootstrap=10000,seed=20261012,intervals_unadjusted=True,groups_overlap=True,
        selection_grain='offline cell-level predicate; average selected orders per parent; main complete-parent harm remains in SUMMARY',CPU_only=True,GT_read=True))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=ROOT/'output/pdf/stvg_opd_temporal_guided_candidate/P7_B';dest.mkdir(parents=True,exist_ok=True)
    pdfpath=dest/'support_allocation_cases.pdf';assert not pdfpath.exists();private=[]
    with PdfPages(pdfpath) as pdf:
        for chosen in selected:
            r=next(v for v in d['records'] if v['public_cell']==chosen['public_cell']);ds=r['dataset'];q=r['query_ordinal']
            raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q];saved=load(A/'readouts'/f"{r['public_cell']:05}.pt")
            old=normalized(load(ROOT/r['original_Full']['input_path']))
            ep=B/'expert_I1'/f"{r['capture_index']:05}.pt";assert sha(ep)==read(B/'EXPERT_EVIDENCE_BARRIER.json')['files'][str(ep.relative_to(ROOT))];ex=load(ep)
            boxes={'B_posthoc':saved['boxes']['OPD'],'Full_Before':load(ROOT/r['original_Full']['path'])['fit']['before']};files={str(ep.relative_to(ROOT)):sha(ep)}
            for arm in ['C_temporal_guided_OPD','D_temporal_guided_Direct']:
                p=B/'formal'/f"{r['public_cell']:05}"/(arm+'_repeat0.pt');h=sha(p);assert h==read(B/'GLOBAL_PREDICTION_BARRIER.json')['files'][str(p.relative_to(ROOT))]
                boxes[arm]=load(p)['fit']['final'];files[str(p.relative_to(ROOT))]=h
            curves={};metrics={}
            for arm,bb in boxes.items():
                z,fids,iou=dense_metric(array(bb),raw,truth,span,saved['intervals']['T1'],ds)
                scalar_checks+=near(z,official(array(bb),raw,truth,span,saved['intervals']['T1'],ds));metrics[arm]=z;curves[arm]=iou
                for key in ['v','t','s']:scalar_checks+=near(z[key],chosen['row'][('Full_Before_T1' if arm=='Full_Before' else arm)+'_'+key])
            fids=np.array(fids);real=np.isin(fids,raw['frame_ids']);event=(fids>=span[0])&(fids<span[1])
            oldobs=[raw['frame_ids'][p] for p in old['expert']['actual_observation_positions']]
            newobs=[raw['frame_ids'][p] for p in ex['expert']['actual_observation_positions']]
            union=np.isin(fids,oldobs+newobs);mask=event&real&~union
            effect=float((curves['C_temporal_guided_OPD']-curves['B_posthoc'])[mask].mean()) if mask.any() else None
            scalar_checks+=near(effect,chosen['row']['C_minus_B_common_unobserved_real_IoU'])
            I0=saved['intervals']['T0'];I1=saved['intervals']['T1'];fig,axes=plt.subplots(3,1,figsize=(12,7.5),constrained_layout=True,gridspec_kw={'height_ratios':[1,2,2]})
            ax=axes[0]
            for y,iv,label,color in [(0,span,'GT event','#267445'),(1,I0,'Native I0','#68758b'),(2,I1,'Projected I1','#c4812b')]:
                ax.plot(iv,[y,y],lw=8,color=color,solid_capstyle='butt');ax.text(iv[1],y,' '+label,va='center',fontsize=9)
            ax.scatter(oldobs,np.ones(len(oldobs)),marker='v',color='#253d7f',s=50,zorder=4,label='old visited DINO frames')
            ax.scatter(newobs,np.ones(len(newobs))*2,marker='v',color='#ae423c',s=50,zorder=4,label='new visited DINO frames')
            ax.set_ylim(-.5,2.6);ax.set_yticks([]);ax.legend(loc='upper left',fontsize=8);ax.set_xlabel('Physical frame ordinal (private IDs, no video/query identity)')
            ax=axes[1]
            for arm,color,label in [('Full_Before','#9e9e9e','Identical Full Before'),('B_posthoc','#5e6e87','B: same OPD / old observations'),('C_temporal_guided_OPD','#238578','C: OPD / projected observations'),('D_temporal_guided_Direct','#b77237','D: Direct / projected observations')]:
                ax.plot(fids,curves[arm],color=color,lw=1.2,label=label)
                ax.scatter(fids[real],curves[arm][real],color=color,s=6)
            ax.axvspan(span[0],span[1],alpha=.08,color='#237747');ax.set_ylabel('Spatial IoU with official GT');ax.set_ylim(-.02,1.03);ax.legend(fontsize=8);ax.grid(alpha=.2)
            ax=axes[2];delta=curves['C_temporal_guided_OPD']-curves['B_posthoc']
            ax.plot(fids,delta*100,color='#257779',lw=1);ax.scatter(fids[mask],delta[mask]*100,s=24,facecolors='none',edgecolors='#302f40',label='common unobserved real event frames')
            ax.axhline(0,color='.5');ax.axvspan(span[0],span[1],alpha=.08,color='#237747');ax.set_ylabel('C - B spatial IoU (pp)');ax.set_xlabel('Physical frame ordinal');ax.legend(fontsize=8);ax.grid(alpha=.2)
            fig.suptitle(f"{ds} anonymous parent{r['parent_id']} / {chosen['label']} (selected by paired two-order effect)\n"
                f"Paired-parent C-B {chosen['paired_parent_coupling']*100:+.2f} pp; plotted order1 C-B {chosen['row']['coupling_C_minus_B_v']*100:+.2f} pp; output interval shared",fontsize=12)
            pdf.savefig(fig);plt.close(fig)
            private.append(dict(**{k:v for k,v in chosen.items() if k!='row'},source_bindings=files,
                Native_interval=I0,projected_interval=I1,GT_event=span,frame_ids=fids.tolist(),real_decoder=real.tolist(),
                old_visited=oldobs,new_visited=newobs,curves={k:v.tolist() for k,v in curves.items()},metrics=metrics,
                common_unobserved_real_frames=int(mask.sum()),GT_read=True))
    write(B/'PRIVATE_CASE_READBACK.json',dict(status='pass',scope='offline best/worst added96 cases; raw support/framewise geometry stays private',records=private,scalar_checks=scalar_checks,CPU_only=True,GT_read=True))
    write(OUT/'P7_B/CASE_SELECTION_PUBLIC.json',dict(status='measured_after_complete_B_root',selection='each target best/worst added96 paired-parent coupling; order1 shown; not independent prevalence evidence',
        cases=[{k:v for k,v in x.items() if k!='row'} for x in selected],case_count=4,CPU_only=True,GT_read=True))
    write(B/'ROOT_CASE_READBACK.json',dict(status='pass',scope='all anonymous support groups and four offline framewise support-allocation cases/official independent dense equality',
        cases=4,pages=4,scalar_checks=scalar_checks,extra_full_before_and_expert_scalar_checks=extra_checks,extra_diagnostic_cells=512,
        complete_exact_evidence_controls=dict(controls),private_payload_sha256=sha(B/'PRIVATE_CASE_READBACK.json'),
        PDF_file=str(pdfpath.relative_to(ROOT)),PDF_sha256=sha(pdfpath),PDF_bytes=pdfpath.stat().st_size,CPU_only=True,GT_read=True,time=time.time()))
    report=OUT/'P7_B/REPORT.md';assert report.exists()
    text=['','## Support, before/after localization and negative cases','',
        'The offline groups below overlap and select cells by complete sealed GT diagnostics. Selected orders are averaged per parent inside each group. These are exploratory descriptions, not an online gate or a causal decomposition of detector identity versus coverage. Main all-parent harm counts remain in SUMMARY.', '',
        '| Target | Offline group | Cells | Parents | C-B vIoU (pp) | 95% CI (pp) | C-B common unobserved real spatial IoU (pp) |',
        '|---|---|---:|---:|---:|---|---:|']
    for ds,data in groups.items():
        for name,g in data.items():
            c=g['statistics']['coupling_C_minus_B_v'];u=g['statistics']['C_minus_B_common_unobserved_real_IoU']
            mean='undefined' if c['mean'] is None else f"{c['mean']*100:+.3f}";ci='undefined' if c['ci95'] is None else f"[{c['ci95'][0]*100:+.3f},{c['ci95'][1]*100:+.3f}]"
            us='undefined' if u['mean'] is None else f"{u['mean']*100:+.3f}"
            text.append(f"| {ds} | {name} | {g['cells']} | {g['distinct_parents']} | {mean} | {ci} | {us} |")
    budget=read(B/'EXPERT_BUDGET_READBACK.json')
    text+=['',f"Every one of256 query budgets is matched:4 planned positions, the same per-query actual visit count,936 total visits per arm;22 queries have no call under the original parser rule. Visits move for{budget['parents_with_moved_visits']} parents. Original admission accepts812 positions versus814 at projected positions, a downstream consequence reported with quality and event membership rather than assumed to be pure coverage.", '',
        f"Among the complete512 paired cells, {controls['exact_same_evidence_controls']} have exactly identical admitted evidence ({controls['nonempty_same_evidence_controls']} nonempty). New C final boxes and entire final parameter states match original Full bitwise for every such control. Changed-evidence C/D cases keep the same Full Before and original hyperparameters.", '',
        'DIAGNOSTIC_EXTRA_ROWS retains the identical actual Full Before scores, B/C/D after-minus-before spatial localization, old versus new admitted-expert IoU and admission changes. Admitted-reference IoU does not establish object identity. Audited fit timings include recording and CPU math verification and are not isolated deployment latency; the originally authorized isolated cost experiment remains separate.', '',
        'Four support-allocation case pages show each target\'s best and worst added96 paired-parent effect with order1 displayed, including identical Full Before, B/C/D actual curves, Native/projected/GT intervals and old/new visits. These are selected offline illustrations; every parent and severe negative remains in complete scalar outputs. Actual framewise payload and GT geometry stay private.']
    report.write_text(report.read_text()+'\n'.join(text)+'\n')
    print('P7_B_SUPPORT_AND_NEGATIVE_CASES_WRITTEN',4,scalar_checks)

if __name__=='__main__':run()
