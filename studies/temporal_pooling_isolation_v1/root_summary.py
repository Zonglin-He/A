"""Root-only compact two-domain table from completed audited ROI isolation."""
import run as original
import json, time
from pathlib import Path

def main():
    original.validate()
    rows=[];reports={};bindings={}
    for ds in original.COUNTS:
        d=original.D/ds;p=original.P/ds
        receipt=original.read(d/'root_export_receipt.json')
        assert receipt['status']=='root_complete_full_temporal_ROI_isolation'
        assert receipt['report_sha256']==original.sha(p/'report.json')
        assert receipt['analysis_sha256']==original.sha(p/'ANALYSIS.md')
        assert receipt['audit_seal_sha256']==original.sha(d/'sealed.json')
        exit_receipt=original.read(d/'actual_exit.json')
        assert exit_receipt['code']==0 and exit_receipt['actual_exit']
        assert not original.live(exit_receipt['worker'])
        report=original.read(p/'report.json');assert report['queries']==original.COUNTS[ds]
        assert report['paired']['t']['CI95'][0]<0<report['paired']['t']['CI95'][1]
        assert report['paired']['v']['CI95'][0]<0<report['paired']['v']['CI95'][1]
        reports[ds]=report
        bindings[ds]=dict(root_export_sha256=original.sha(d/'root_export_receipt.json'),
                          audit_seal_sha256=original.sha(d/'sealed.json'),
                          actual_exit_sha256=original.sha(d/'actual_exit.json'),
                          report_sha256=original.sha(p/'report.json'))
    text='# Temporal appearance pooling region: full-domain isolation\n\n'
    text+='The latest direct user request is completed using the exact existing full-domain strict control, plus a new independent all-query numerical-chain verification. Existing AuxSourceROI and selected FullMAP are the requested Source-box and corrected-box appearance arms; BOTH retain the same MAIN spatial candidate geometry. No Source/expert/model/optimizer/GT scoring was repeated. Four actual raw grids additionally verify both ROI pools with the same function and context. Full replay uses the audited sufficient features.\n\n'
    text+='Only the temporal pooling region differs. Same eight expert observations, native Source feature grid, 1.5 context, centering/normalization, one all-video prototype pair, kappa0/beta1 and one posterior update. Each arm selects its own event interval and runs the unchanged final DP on that interval. Full-span candidate boxes, raw unary affinities, box-IoU links and objective coefficients are fixed; interval-dependent eligibility and the original observed/unobserved/edge normalization are recomputed by the same code for each own interval. Same candidate construction does not imply identical final paths after the event interval changes.\n\n'
    text+='|Full target set|Temporal appearance ROI|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---|---:|---:|---:|---:|\n'
    for ds,r in reports.items():
        for a,m in r['means'].items():
            text+=f"|{ds} ({r['queries']} queries / {r['parents']} parents)|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|\n"
    text+='\nCorrected ROI minus Source ROI, percentage points; query-equal differences with parent-video cluster10000bootstrap pointwise95CI:\n\n|Target set|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|Severe t/v/union once|Rescue union|\n|---|---:|---:|---|---:|\n'
    for ds,r in reports.items():
        def f(k):
            v=r['paired'][k];return f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"
        text+=f"|{ds}|{f('t')}|{f('v')}|{r['severe_t']}/{r['severe_v']}/{r['severe_union_once']}|{r['rescue_union_once']}|\n"
    text+='\nBoth full target sets have zero-crossing tIoU and vIoU confidence intervals. The evidence does not establish a stable average temporal-localization benefit specifically from shared correction of the pooling boxes. It is not an equivalence test or proof of zero effect. The supported mechanism claim is limited to object-region appearance supporting temporal correction; a separate causal benefit of corrected ROI is not established. No manuscript/default/parameter edits are made, and no new configurations are selected from these exposed full target sets.\n\n'
    text+='All signed query-level results, duration groups, severe losses/rescues, interval/path change counts, internal fixed-support sIoU/boundary checks and actual cached replay costs remain in the two domain reports. sIoU is intentionally outside the compact table. Inherited Source/teacher costs are nonzero; new cached verification wall is neither process CPU time nor deployment latency.\n\n'
    text+='The first new preflight falsely detected geometry mutation by hashing a Python dictionary as NumPy object-pointer bytes. The preserved science/numerical code is unchanged; independently frozen runtime_v2 canonicalizes only metadata hashing. Four raw-grid cases and both full-domain verifications use this version. Failed-attempt elapsed time was not recorded, rather than fabricated as zero.\n\n'
    text+='Current CG and gamma owners were not signaled or changed. EdgeTAM remains the external mature propagation comparison, and every corruption task remains paused by direct user instruction.\n'
    (original.P/'README.md').write_text(text)
    receipt=dict(status='complete_full_both_domains_root_analyzed_pending_publication',
                 bindings=bindings,summary_sha256=original.sha(original.P/'README.md'),
                 code_sha256=original.sha(Path(__file__)),new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,
                 both_t_v_CIs_cross_zero=True,no_current_method_or_manuscript_change=True,time=time.time())
    original.atomic(original.D/'root_completion.json',receipt)
    original.atomic(original.P/'root_completion.json',receipt)
    print(json.dumps(dict(status=receipt['status'],datasets=list(reports),summary_sha256=receipt['summary_sha256'])),flush=True)

if __name__=='__main__':main()
