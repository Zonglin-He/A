"""Both-order rendering of four already selected offline P7-B parents."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *
from scripts.score_stvg_opd_temporal_capture_p7a_v1 import rows_read,array,near,context_truth

def run():
    assert read(B/'ROOT_READBACK.json')['status']=='pass' and read(B/'ROOT_CASE_READBACK.json')['status']=='pass'
    ns=B/'case_orders_layout002';selected=read(OUT/'P7_B/CASE_SELECTION_PUBLIC.json')['cases'];rows={r['public_cell']:r for r in rows_read(OUT/'P7_B/ROWS.jsonl.gz')}
    extra={r['public_cell']:r for r in rows_read(OUT/'P7_B/DIAGNOSTIC_EXTRA_ROWS.jsonl.gz')};gt,spans=context_truth();d=read(BASE/'DESIGN_LOCK.json')
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.tastvg_oracle_event5_v1 import official
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    target=ROOT/'output/pdf/stvg_opd_temporal_guided_candidate/P7_B/support_allocation_cases_both_orders.pdf';assert not target.exists()
    checked=0;private=[]
    with PdfPages(target) as pdf:
        for z in selected:
            for order in ['order1','order2']:
                r=next(r for r in d['records'] if r['parent_id']==z['parent_id'] and r['order']==order);row=rows[r['public_cell']];before_extra=extra[r['public_cell']]
                ds=r['dataset'];q=r['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q];a=load(A/'readouts'/f"{r['public_cell']:05}.pt")
                op=normalized(load(ROOT/r['original_Full']['input_path']));ep=B/'expert_I1'/f"{r['capture_index']:05}.pt";assert sha(ep)==read(B/'EXPERT_EVIDENCE_BARRIER.json')['files'][str(ep.relative_to(ROOT))];ex=load(ep)
                boxes={'Full_Before':load(ROOT/r['original_Full']['path'])['fit']['before'],'B_posthoc':a['boxes']['OPD']}
                for arm in ['C_temporal_guided_OPD','D_temporal_guided_Direct']:
                    p=B/'formal'/f"{r['public_cell']:05}"/(arm+'_repeat0.pt');assert sha(p)==read(B/'GLOBAL_PREDICTION_BARRIER.json')['files'][str(p.relative_to(ROOT))];boxes[arm]=load(p)['fit']['final']
                curves={}
                for arm,bb in boxes.items():
                    m,fids,curve=dense_metric(array(bb),raw,truth,span,a['intervals']['T1'],ds);checked+=near(m,official(array(bb),raw,truth,span,a['intervals']['T1'],ds));curves[arm]=curve
                    for key in ['v','t','s']:checked+=near(m[key],before_extra['Full_Before_T1_'+key] if arm=='Full_Before' else row[arm+'_'+key])
                oldobs=[raw['frame_ids'][p] for p in op['expert']['actual_observation_positions']];newobs=[raw['frame_ids'][p] for p in ex['expert']['actual_observation_positions']]
                ids=np.array(fids);event=(ids>=span[0])&(ids<span[1]);real=np.isin(ids,raw['frame_ids']);mask=event&real&~np.isin(ids,oldobs+newobs)
                delta=curves['C_temporal_guided_OPD']-curves['B_posthoc'];checked+=near(float(delta[mask].mean()) if mask.any() else None,row['C_minus_B_common_unobserved_real_IoU'])
                fig,axes=plt.subplots(3,1,figsize=(12,7.8),constrained_layout=True,gridspec_kw={'height_ratios':[1,2,2]});ax=axes[0]
                for y,iv,label,col in [(0,span,'GT event','#267445'),(1,a['intervals']['T0'],'Native I0','#68758b'),(2,a['intervals']['T1'],'Projected I1','#c4812b')]:ax.plot(iv,[y,y],lw=8,color=col,solid_capstyle='butt');ax.text(iv[1],y,' '+label,va='center',fontsize=9)
                ax.scatter(oldobs,np.ones(len(oldobs)),marker='v',color='#253d7f',s=50,zorder=4,label='old visited DINO frames');ax.scatter(newobs,np.ones(len(newobs))*2,marker='v',color='#ae423c',s=50,zorder=4,label='new visited DINO frames')
                ax.set_ylim(-.5,2.6);ax.set_yticks([]);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),ncol=2,fontsize=8);ax.set_xlabel('Physical frame ordinal')
                ax=axes[1]
                for arm,color,label in [('Full_Before','#9e9e9e','Identical Full Before'),('B_posthoc','#5e6e87','B: OPD / old observations'),('C_temporal_guided_OPD','#238578','C: OPD / projected observations'),('D_temporal_guided_Direct','#b77237','D: Direct / projected observations')]:ax.plot(ids,curves[arm],color=color,lw=1.1,label=label);ax.scatter(ids[real],curves[arm][real],color=color,s=6)
                ax.axvspan(span[0],span[1],alpha=.08,color='#237747');ax.set_ylabel('Spatial IoU with official GT');ax.set_ylim(-.02,1.03);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=8);ax.grid(alpha=.2)
                ax=axes[2];ax.plot(ids,delta*100,color='#257779',lw=1);ax.scatter(ids[mask],delta[mask]*100,s=24,facecolors='none',edgecolors='#302f40',label='common unobserved real event frames');ax.axhline(0,color='.5');ax.axvspan(span[0],span[1],alpha=.08,color='#237747');ax.set_ylabel('C - B spatial IoU (pp)');ax.set_xlabel('Physical frame ordinal');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=8);ax.grid(alpha=.2)
                fig.suptitle(f"{ds} anonymous parent{r['parent_id']} / {z['label']} / {order}\nPaired-parent C-B {z['paired_parent_coupling']*100:+.2f} pp; this order {row['coupling_C_minus_B_v']*100:+.2f} pp; identical output interval",fontsize=12);pdf.savefig(fig);plt.close(fig)
                private.append(dict(parent_id=r['parent_id'],public_cell=r['public_cell'],order=order,frame_ids=ids.tolist(),curves={k:v.tolist() for k,v in curves.items()},GT_event=span,old_visited=oldobs,new_visited=newobs))
    write(ns/'PRIVATE_BOTH_ORDER_CASES.json',dict(status='pass',records=private,CPU_only=True,GT_read=True))
    write(ns/'ROOT_CASE_ORDERS_READBACK.json',dict(status='pass',scope='same four offline best/worst paired parents, both original orders, all eight actual case pages, uncovered observation legends; independently checked official dense/Before/source outputs',cases=4,orders=2,pages=8,scalar_checks=checked,PDF_file=str(target.relative_to(ROOT)),PDF_sha256=sha(target),PDF_bytes=target.stat().st_size,original_root_sha256=sha(B/'ROOT_CASE_READBACK.json'),private_readback_sha256=sha(ns/'PRIVATE_BOTH_ORDER_CASES.json'),new_GPU=0,new_fit=0,CPU_only=True,GT_read=True,time=time.time()))
    report=OUT/'P7_B/REPORT.md';report.write_text(report.read_text()+'\nActual case-view revision: support_allocation_cases_both_orders.pdf displays both original orders of each of the four paired-parent extremes (eight pages), including opposite signs across orders. The original order1-only export and all original receipts remain preserved privately.\n')
    print('P7_B_BOTH_ORDER_CASES_ACTUAL',8,checked)

if __name__=='__main__':run()
