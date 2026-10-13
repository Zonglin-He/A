"""Full positive/negative/unknown forward-only feasibility and cost report."""
import os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scripts.stvg_opd_forward_candidate_common_v1 import *

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    assert read(BASE/'ROOT_READBACK.json')['status']=='pass'
    rows=read(OUT/'BOUNDARY_ROWS.json');s=read(OUT/'STATISTICS.json');d=read(BASE/'TECHNICAL_DECISION.json')
    assert len(rows)==48
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42})
    folder=OUT/'figures';folder.mkdir(exist_ok=True);p=folder/'FORWARD_CANDIDATE_SCREEN.pdf';assert not p.exists()
    with PdfPages(p) as pdf:
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        for j,ds in enumerate(['hc2','vidstg']):
            rr=[r for r in rows if r['dataset']==ds and r['repeat']==0];positive=[r for r in rr if r['reason']=='positive_original_gain']
            for n,color in [('LN','#d97c27'),('Query','#536eae'),('Half','#3b8c74')]:
                yy=[r['candidates'][n]['G']/r['candidates']['A']['G'] for r in positive]
                ax[0,j].plot(range(len(positive)),yy,marker='o',ms=4,label=n,color=color)
            ax[0,j].axhline(.9,color='black',ls='--',lw=1)
            ax[0,j].set(xlabel='Fixed boundaries with G(A)>0 (repeat 0)',ylabel='Actual expert gain / original Adam gain',title=f'{ds}: local correction retention');ax[0,j].legend()
            yy=[(r['candidates']['A']['D']-r['candidates'][r['selected']]['D'])/r['candidates']['A']['D']*100 if r['candidates']['A']['D']>0 else 0 for r in rr]
            ax[1,j].bar(range(len(rr)),yy,color=['#3b8c74' if r['useful_alternatives'] else '#9aa0aa' for r in rr])
            ax[1,j].set(xlabel='All fixed parent / boundary ordinals (repeat 0)',ylabel='Selected reduction of reference drift (%)',title=f"{s['datasets'][ds]['useful_alternative_boundaries']}/{len(positive)} positive boundaries have useful alternatives")
        fig.suptitle('Real-forward candidates at original OPD states',fontsize=16)
        fig.text(.5,.012,'rho=0.9 prelocked. U excludes all physical Omega0 visits; no GT event or labels select candidates.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.045,1,.95));pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        for j,ds in enumerate(['hc2','vidstg']):
            v=s['datasets'][ds];names=['A','LN','Query','Half']
            ax[0,j].bar(names,[v['selected_mix'].get(n,0) for n in names],color=['#9aa0aa','#d97c27','#536eae','#3b8c74'])
            ax[0,j].set(ylabel='Selected fixed boundaries (repeat 0)',title=f'{ds}: all decisions including fallback and empty')
            costs=[v['measured_repeat1_screen_seconds_median'],v['matched_prior_F_seconds_median']]
            ax[1,j].bar(['Four-candidate screen','Prior F local probe'],costs,color=['#3b8c74','#d97c27'])
            ax[1,j].set(ylabel='Measured local seconds (matched repeat 1)',title='Descriptive comparison; different instrumentation')
            for i,x in enumerate(costs):ax[1,j].text(i,x,f'{x:.3f}',ha='center',va='bottom')
            ax[1,j].set_ylim(0,max(costs)*1.18)
        fig.suptitle('Decision coverage and actual local costs',fontsize=16)
        fig.text(.5,.012,'Screen: states + 4 decoder forwards + output transfers + selection. Capture, hashes and serialization excluded.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.045,1,.95));pdf.savefig(fig);plt.close(fig)
    lines=['# Real-forward finite candidate update: fixed feasibility screen','',
      '**Decision: '+d['status']+'.**','',
      'Eight fixed historical P9/P0 previous-P6 queries, Inherited-order1, two fresh captures each, three saved original-A boundaries: 48 comparisons and 192 complete-grid candidate decoder readouts. These are not 192 fits. No new training, backward, optimizer step, DINO or GT access occurred.','',
      'A copies the exact sealed original Adam after-state directly. LN and Query mask that displacement by parameter block; Half uses half in float32. All original proposal/current/entering outputs reproduce the sealed full OPD endpoints. Adam/noise witnesses remain verified unchanged; hypothetical candidates do not write a new online stream.','',
      'Source/pixels/frames/Native interval/old configuration and admitted Top1 evidence are fixed. Expert gain uses admitted Omega0. U is the full actual decoder grid excluding ALL physical visits, including rejected observations; no GT event restricts U. D is the squared box-coordinate difference per U frame from query-entering Inherited-order1, without the old F objective factor 1/2.','',
      '## Fixed rule and complete outcomes','',
      'rho=0.9 was locked before measurements. If G(A)>0 and U exists, select minimum actual D among A/LN/Query/Half satisfying G>=rho*G(A), with fixed A,LN,Query,Half tie order. Otherwise retain A and explicitly record the reason. No no-update strategy or special beneficial fallback is introduced. Original uninformative/empty paths remain exact no-ops.','',
      '| Dataset | Positive G(A) boundaries | Useful alternatives | Fraction | Useful / positive queries | Exact repeated boundaries |','|---|---:|---:|---:|---:|---:|']
    for ds,v in s['datasets'].items():
        fraction=f"{v['useful_fraction']*100:.1f}%" if v['useful_fraction'] is not None else 'unknown'
        lines.append(f"| {ds} | {v['positive_original_gain']} | {v['useful_alternative_boundaries']} | {fraction} | {v['useful_queries']}/{v['positive_queries']} | {v['repeat_exact']}/12 |")
    lines+=['','Useful means a non-A feasible candidate lowers D by more than 1e-10+64*eps_float32*D(A). The prelocked resource criterion is at least half the positive boundaries in each domain plus exact replay. Repeated boundaries are not independent parents; historical local feasibility is not held-out efficacy.','',
      '| Dataset | Nonpositive A gain | Empty observed | Selected A/LN/Query/Half | Screen median seconds | Prior F median seconds |','|---|---:|---:|---|---:|---:|']
    for ds,v in s['datasets'].items():
        mix='/'.join(str(v['selected_mix'].get(n,0)) for n in ['A','LN','Query','Half'])
        lines.append(f"| {ds} | {v['nonpositive_original_gain']} | {v['empty_observed']} | {mix} | {v['measured_repeat1_screen_seconds_median']:.6f} | {v['matched_prior_F_seconds_median']:.6f} |")
    lines+=['','## Cost and scientific boundaries','',
      'Costs use the same informative fixed repeat1 contexts. The complete screen includes state construction, four real decoder forwards, CPU output transfers, metric/selection arithmetic and final state copy. Capture, current/reference validation, hashes, serialization and independent audits are outside this timer. Prior F includes Jacobian/solver and recording arithmetic; their ratio is descriptive across different instrumentation, not a measured endpoint speedup or full adaptation/deployment latency.','',
      'All 48 boundary records, 192 candidate gains/drifts/costs, nonpositive original gains, rejected alternatives, empty/undefined paths and exact repeat results are published. No GT performance was scored. Smaller drift may preserve an incorrect entering trajectory; expert agreement does not establish target correctness.','',
      'This is a candidate-opportunity screen at saved A states, not a completed adaptive trajectory. Success only supports specifying a separate task-validation protocol. Failure stops this fixed finite-candidate recipe without refuting all functional preservation. No new full candidate fitting, GT validation, expanded candidate set, rho search, extra module, production/main run or paper is automatic.','',
      'The prior high-dimensional projection remains complete with failed numerical qualification. P7-A/CURRENT and cancelled baseline same process/start/optimizer/ordinary-lease T standby remain unchanged. The existing monitor remains ACTIVE while future method/main protocols are unresolved.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(BASE/'FIGURE_EXPORT.json',dict(status='pass',scope='all fixed real-forward opportunities/negatives/unknowns/selection and local costs',PDF_files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,pages=2)},PDFs=1,pages=2,CPU_only=True,GT_read=False,time=time.time()))
    print('REAL_FORWARD_REPORT_TWO_PAGES_EXPORTED')

if __name__=='__main__':report()
