"""Post-seal synthesis of all three fixed studies; never chooses a deployed arm."""
import sys,json,shutil,hashlib,importlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from research.joint_event_v1.run import R,read,put,sha,B
torch.set_num_threads(1)
primaries={1:'JointGuard',2:'ExcessGuard',3:'LocalGuard'};reports={};verification={};case_rows=[];interval_risk={}
for version,primary in primaries.items():
    mod=importlib.import_module(f'research.joint_event_v{version}.run');plan=mod.verify();D=mod.D;P=mod.P;seal=read(D/'sealed.json')
    assert seal['count']==64 and not seal['GT_read']and not seal['CUDA_initialized']
    for e in seal['files'].values():assert sha(D/e['path'])==e['sha256']and sha(D/e['trace_path'])==e['trace_sha256']
    report=read(P/'report.json');receipt=read(D/'score_receipt.json');assert receipt['report_sha256']==sha(P/'report.json')
    # Scores are unchanged. Correct only orientation labels for lower-is-better MAE.
    if not report.get('MAE_count_orientation_corrected',False):
        folder=D/'MAE_count_label_correction';folder.mkdir(exist_ok=True)
        shutil.copy2(P/'report.json',folder/'original_report.json');put(folder/'original_receipt.json',receipt)
        before={k:(v['boundary_MAE_seconds']['mean'],v['boundary_MAE_seconds']['CI95'])for k,v in report['paired'].items()}
        for compare in report['paired'].values():
            q=compare['boundary_MAE_seconds'];q['improved'],q['harmed']=q['harmed'],q['improved'];changed=q['improved']+q['harmed']
            q['harmed_fraction_all']=q['harmed']/64;q['harmed_fraction_changed']=q['harmed']/changed if changed else None
        report['MAE_count_orientation_corrected']=True;put(P/'report.json',report)
        assert before=={k:(v['boundary_MAE_seconds']['mean'],v['boundary_MAE_seconds']['CI95'])for k,v in report['paired'].items()}
        receipt.update(report_sha256=sha(P/'report.json'),MAE_count_label_correction='swap improved/harmed labels only for lower-is-better boundary MAE; predictions, scalar scores, means and CIs unchanged',original_report_sha256=sha(folder/'original_report.json'));put(D/'score_receipt.json',receipt)
    reports[version]=report
    from stvg.spatial import SpatialGeometry,evidence_dp
    samples=0
    for key in ['RC009','RC016','RC042']:
        cap=torch.load(B/'captures'/f'{key}.pt',map_location='cpu',weights_only=False);geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry'])
        with np.load(D/'predictions'/f'{key}.npz',allow_pickle=False)as z:
            for a in ['NativeG6',primary]:
                fresh=evidence_dp(geom.for_interval(z[a+'_interval'].tolist()))['final'].numpy();assert np.array_equal(fresh,z[a]);samples+=1
    if version==3:
        for key,e in seal['files'].items():
            with np.load(D/e['path'],allow_pickle=False)as z:
                ids=z['frame_ids'];lookup={int(f):i for i,f in enumerate(ids)}
                def indices(a):span=z[a+'_interval'];return np.array([lookup[int(span[0])],lookup[int(span[1])-1]])
                base=indices('NativeG6');p0=z['positions8'];anchor=ids[p0]+.5
                for a in mod.ARMS:
                    idx=indices(a);assert abs(idx-base).max()<=1
                    def mem(x):return (anchor>=ids[x[0]])&(anchor<ids[x[1]]+1)
                    assert np.array_equal(mem(idx),mem(base))
    changed=0
    for row in report['rows']:
        with np.load(D/'predictions'/f"{row['query']}.npz",allow_pickle=False)as z:
            changed+=int(not np.array_equal(z[primary+'_interval'],z['NativeG6_interval']))
    pair=report['paired'][primary+'-NativeG6']
    interval_risk[version]=dict(changed_intervals=changed,complete_v_harm_fraction_changed_intervals=pair['v']['harmed']/changed if changed else None,
        t_harm_fraction_changed_intervals=pair['t']['harmed']/changed if changed else None)
    verification[version]=dict(queries=64,NativeG6_bitwise_compatible=64,official_dense_checks=report['checks'],max_metric_error=report['max_metric_error'],independent_spatial_replay_checks=samples,spatial_replay_bitwise=True,
        full_paused=True,maintained_runtime_files=len(plan['paper_code']),provider_files=len(plan['provider_sources']),all_frozen_hashes_unchanged=True,
        all64_predictions_before_new_GT=True,CPU_only=True,new_visual_decoder_expert_forwards=0,query_steps=0,
        plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'),MAE_count_label_correction=True)
    if version==3 and (P/'semantic_audit.json').exists():
        verification[version]['postseal_semantic_audit_sha256']=sha(P/'semantic_audit.json')
    put(P/'verification.json',verification[version]);put(D/'verification.json',verification[version])
    for row in report['rows']:
        if row['query']in ['RC009','RC016','RC025','RC029','RC042']:
            v=row['values'][primary];b=row['values']['NativeG6'];case_rows.append(dict(version=version,query=row['query'],delta_t_pp=100*(v['t']-b['t']),delta_v_pp=100*(v['v']-b['v']),boundary_movement_frames=row['movement'][primary],accepted=row['decision'][primary]['accept']))

summaryP=R/'results/joint_event_hc64_v3';summaryD=R/'outputs/joint_event_hc64_v3'
bias_path=R/'outputs/joint_event_hc64_v1/completeness_bias_audit.json'
correlations=[];length_deltas=[]
for row in reports[1]['rows']:
    key=row['query'];tr=torch.load(R/'outputs/joint_event_hc64_v1/traces'/f'{key}.pt',map_location='cpu',weights_only=False)
    with np.load(R/'outputs/joint_event_hc64_v1/predictions'/f'{key}.npz',allow_pickle=False)as z:
        ids=z['frame_ids'];pairs=tr['pairs'];lengths=ids[pairs[:,1]]+1-ids[pairs[:,0]]
        correlations.append(float(np.corrcoef(tr['semantic']['Relative']['score'],np.log(lengths))[0,1]))
        a=z['JointGuard_interval'];b=z['NativeG6_interval']
        if not np.array_equal(a,b):length_deltas.append(int((a[1]-a[0])-(b[1]-b[0])))
recomputed=dict(expanded=sum(x>0 for x in length_deltas),contracted=sum(x<0 for x in length_deltas),mean_loglength_score_correlation=float(np.mean(correlations)))
if bias_path.exists():
    bias=read(bias_path)
    assert bias['expanded']==recomputed['expanded']and abs(bias['mean_loglength_score_correlation']-recomputed['mean_loglength_score_correlation'])<1e-13
else:put(bias_path,recomputed);bias=recomputed
summary=dict(status='completed_all_three_fixed_development_studies_no_promotion',unique_queries=64,unique_parents=64,
             prediction_records=192,independent_sources_not_192=True,official_dense_checks=sum(v['official_dense_checks']for v in verification.values()),
             max_metric_error=max(v['max_metric_error']for v in verification.values()),independent_spatial_replay_checks=sum(v['independent_spatial_replay_checks']for v in verification.values()),
             stages={str(v):dict(primary=primaries[v],means=r['means'][primaries[v]],paired_vs_G6=r['paired'][primaries[v]+'-NativeG6'],groups=r['groups_vs_G6'],mechanism=r['mechanism'],
                                fixed_space_v_gain_pp=100*(r['fixed_space_means'][primaries[v]]['v']-r['fixed_space_means']['NativeG6']['v']),
                                mean_cached_research_panel_feedback_seconds=r['mean_cached_feedback_seconds'],interval_risk=interval_risk[v])for v,r in reports.items()},
             v1_bias=dict(accepted=22,expanded=bias['expanded'],contracted=bias['contracted'],mean_loglength_score_correlation=bias['mean_loglength_score_correlation']),
             cases=case_rows,full_status='paused_by_user',HC1902='sealed_scored',Vid_preserved=1373,
             selection='No study promoted, no full resume, no GT-conditioned mixture; v2 retained only as conservative research candidate',
             scientific_outcome='Better completeness structure and null sanity do not establish reliable description-specific boundary feedback. V3 more corrections also more harms; shifted feedback not clearly beaten.')
put(summaryP/'summary.json',summary)
lines=['# HC64: joint event completeness, matched null, and paired boundary correction','',
 '**Three actual fixed studies completed. No candidate is admitted to the main method.** The conservative matched-null variant produces4 complete-vIoU improvements and2 harms, with0 losses>5pp, but changes only7 intervals and gives+.0129pp tIoU. Directly applying the same feedback to more local boundary pairs increases gains to17 and harms to16 and lowers tIoU. The requested combination of broad improvement and a low harm proportion is not established. Full remains paused at HC1902 scored/Vid1373 saved.','',
 '## What changed in the feedback','',
 'Previous independent hidden-change peaks can cut inside an event. This study evaluates both boundaries as a single interval explaining the predicate evidence, including its interior and what is excluded. It uses the exact saved original8 ReferDINO object-token logits and a fixed selected whole track; no framewise identity switches, new teacher or actionness head.','',
 'A query-only dependency rule extracts main-clause lexical predicates and their clausal/conjoined verbs, excluding noun-attached descriptions when a main verb exists. The target nominal reference follows wh-determiner/wh-person, nominal-subject, then first-noun fallback. Exact old event token mapping reproduces64/64. No raw text is published. Predicate and nominal matching logits are normalized over the same15 tracked queries; their target-track ownership difference is the proposed identity-relative event signal, not an originally calibrated event probability.','',
 'For each query and offset, a disposable ridge1 linear projection maps normalized concatenated six-layer temporal hidden to these pseudo-semantic margins at8 observations. Full8 and8 leave-one projections are fitted. Label centering/scaling in each fold excludes its heldout label; raw video features remain available. Sigmoid intercepts match the native posterior expected occupancy. No GT duration enters this calibration. The original eight teacher logits are temporally fused, so the folds test regression transfer, not independent withheld-teacher inference.','',
 'For every candidate interval I, integrate each predicate soft mask p over physical frame cells and compute','',
 r'\[J_k(I)=\frac{\int_I p_k(t)\,dt}{|I|+\int p_k(t)\,dt-\int_I p_k(t)\,dt},\qquad R_1(I)=\frac1K\sum_k\log J_k(I).\]','',
 'This combines semantic coverage and background spill across both boundaries, while permitting different predicates to occur at different times. No unverified textual event ordering is imposed. J is a ratio of expected intersection/union under an operational soft mask, not expected true tIoU.','',
 'The guard requires positive aggregate leave-one prediction gain at both offsets, positive interval-score gain under all18 full/leave-one views and no loss in any predicate under the combined full views. Otherwise it returns the exact original interval. This is a heuristic against unsupported changes, not a guarantee against shared teacher errors.','',
 'The proposal-level semantic comparison is inspired by [CPL, CVPR2022](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf). Its [official code](https://github.com/minghangz/cpl/blob/main/models/cpl.py) uses trained proposal-conditioned transformers and masked-word reconstruction; those trained capabilities are not reproduced by our per-query regression. [Point-supervised completeness](https://arxiv.org/abs/2108.05029) additionally uses labeled action points and pseudo-background training. Our expert observation positions are not action-boundary labels, so that result cannot validate this feedback.','',
 '## Mechanism-led revisions, with all original failures retained','',
 'V1: joint soft-mask completeness, native anchor-class mass preservation, original physical MBR. Controls remove the guard, use independent semantic endpoint rises/falls with the same guard, or use the old raw event response with the same projection/scoring/guard. It improves average tIoU+.4586pp but all22 accepted intervals expand. Candidate score/log-length correlation is.9353. Even weak nearly uniform masks can reward expansion; semantic agreement does not remove this structural bias.','',
 'V2: subtract a time-uniform mask with identical total predicate evidence before posterior reweighting:','',
 r'\[R_2(I)=\frac1K\sum_k\left[\log J_k(I;p_k)-\log J_k(I;p_k\equiv\bar p_k)\right].\]','',
 'A constant mask now gives zero for every interval; the complete-video interval gives zero for every mask. Both are unit-checked. No strength/temperature scan occurs. All projected masks are reused; no new fit or text pass. Shifted controls move the logit profile by one-quarter timeline and restore its physical occupancy, rather than changing the evidence mass.','',
 'V3: same V2 feedback and guard, but test whether the posterior MBR suppresses useful local corrections. Enumerate at most9 valid native-grid pairs, moving each original boundary by at most one sampled node while preserving native eight-anchor membership. The guarded arm chooses maximum worst-view gain among feasible local pairs; the unguarded arm chooses combined score. Native tie preference is fixed. This is direct pair correction, not another posterior or query-learning claim.','',
 'Every study is locked before its predictions and independently sealed before scoring. V2 and V3 are explicitly adaptive development revisions of already-exposed HC64, not fresh confirmation or independent sources. No GT selects a branch, radius, state or query-specific configuration.','',
 '## Complete comparison','',
 '| Stage / arm | vIoU % | tIoU % | Δv G6 pp | Δt G6 pp | v improved / harmed | t improved / harmed | severe v / t harms >5pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
for v,r in reports.items():
    for a,m in r['means'].items():
        if a in ['NativeG6','Source']:continue
        pair=r['paired'][a+'-NativeG6'];dv=pair['v'];dt=pair['t']
        lines.append(f"| V{v} {a} | {100*m['v']:.4f} | {100*m['t']:.4f} | {dv['mean']:+.4f} | {dt['mean']:+.4f} | {dv['improved']} / {dv['harmed']} | {dt['improved']} / {dt['harmed']} | {dv['severe_below_minus5pp']} / {dt['severe_below_minus5pp']} |")
m=reports[1]['means']['NativeG6'];s=reports[1]['means']['Source']
lines += ['',f"NativeG6 is {100*m['v']:.4f}% vIoU/{100*m['t']:.4f}% tIoU. Original Source is {100*s['v']:.4f}%/{100*s['t']:.4f}%. All NativeG6 interval/box outputs are bitwise identical across studies and to the prior sealed baseline.",'',
          '## Benefit versus risk, not mean gain alone','',
          '| Preregistered stage primary | Harmed /64, complete v | Harmed / changed intervals | Harmed / changed v scores | Δv95% CI pp | Δt95% CI pp |','|---|---:|---:|---:|---|---|']
for v,r in reports.items():
    a=primaries[v];q=r['paired'][a+'-NativeG6'];dv=q['v'];dt=q['t']
    risk=interval_risk[v]
    lines.append(f"| V{v} {a} | {dv['harmed']}/64 ({100*dv['harmed_fraction_all']:.2f}%) | {dv['harmed']}/{risk['changed_intervals']} ({100*risk['complete_v_harm_fraction_changed_intervals']:.2f}%) | {100*dv['harmed_fraction_changed']:.2f}% | [{dv['CI95'][0]:+.4f}, {dv['CI95'][1]:+.4f}] | [{dt['CI95'][0]:+.4f}, {dt['CI95'][1]:+.4f}] |")
lines += ['', 'CIs use10,000 fixed-seed paired parent draws; one query per parent. They are descriptive after iterative development and do not account for method selection/exposure. The narrow positive complete-v interval for V2 is not proof of generalization, and its temporal interval spans zero.','',
          'V2 has58 unchanged complete-v scores, so its3.125% all-query harm rate must be read alongside28.57% harm among its7 changed intervals (33.33% among6 changed-v scores), and only4 improved queries. V3 increases coverage but harms16/37 changed intervals (43.24%;48.48% of changed-v scores), and six tIoU losses exceed5pp. The guard is not a correctness test.','',
          '| Stage primary | Short Δt G6 pp (13) | Long Δt G6 pp (51) | Fixed-native-space Δv pp |','|---|---:|---:|---:|']
for v,r in reports.items():
    a=primaries[v];lines.append(f"| V{v} {a} | {r['groups_vs_G6']['short_le5s'][a]['t']:+.4f} | {r['groups_vs_G6']['long_gt5s'][a]['t']:+.4f} | {summary['stages'][str(v)]['fixed_space_v_gain_pp']:+.4f} |")
lines += ['', 'GT-duration groups are offline descriptions only. No duration-conditioned mixture is evaluated or deployed.','',
          '## Same-case outcomes','',
          '| Query | V1 Δt pp | V2 Δt pp | V3 Δt pp |','|---|---:|---:|---:|']
for key in ['RC009','RC016','RC025','RC029','RC042']:
    ds=[next(x['delta_t_pp']for x in case_rows if x['version']==v and x['query']==key)for v in [1,2,3]]
    lines.append(f'| {key} | {ds[0]:+.4f} | {ds[1]:+.4f} | {ds[2]:+.4f} |')
lines += ['', 'V1 repairs RC016 by expansion but harms RC009 by the same tendency. V2 correctly abstains on these ambiguous cases but supplies no correction. V3 repairs RC016 and RC042 with paired moves, yet again damages RC009. RC025/RC029 lack positive projection transfer and remain native in guarded versions. Local search does not fix misaligned semantic supervision.','',
          'V2 real versus shifted evidence has only+.0015pp tIoU, with a broad CI crossing zero. V3 real versus shifted tIoU is−.0739pp, also inconclusive. These controls do not establish that the learnt pseudo-semantic signal reliably identifies the described event rather than consistent but irrelevant temporal structure.','',
          '## Cost and verification','',
          'No new Source encoder, decoder, visual head, ReferDINO call, RGB decoding, backbone training or query-gradient update. V1 parses64 texts with the existing CPU Stanza parser and fits disposable tiny linear regressions against already saved expert-token responses. V2/V3 reuse those exact projected masks. All stages preserve48 maintained runtime and434 provider source hashes; current full remains paused.','',
          f"Mean cached research-panel feedback costs: V1 {reports[1]['mean_cached_feedback_seconds']*1000:.2f}ms/query (CPU parse and both relative/raw projections), V2 {reports[2]['mean_cached_feedback_seconds']*1000:.2f}ms, V3 {reports[3]['mean_cached_feedback_seconds']*1000:.2f}ms. These include experimental controls, exclude cache creation and initial text-parser loading, and are not measured end-to-end main-method deployment costs.",
          '', 'All192 new query records across three seals precede their corresponding new scoring. They contain64 unique queries/parents, not192 independent samples.2304 official/DenseTube metric checks agree to6.66e-16;18 independent spatial replay checks are bitwise exact. All V3 outputs obey the one-node paired bound and unchanged anchor membership. Scalar MAE values, means and CIs never changed; a post-seal summary corrects only the original scorer’s improved/harmed count orientation for this lower-is-better metric, preserving original reports/receipts privately.','',
          '**Disposition:** retain V2 as a conservative research result only; do not substitute it for the main temporal module. V1 and V3 remain measured negative tradeoffs. Broader gains with low harm are not obtained, and changing reward strength or adding query steps has no support from this experiment. The remaining requirement is a feedback signal calibrated to event semantics and background, not agreement of replicas of the same object-grounding proxy. Full remains paused; no new full or default promotion.','',
          '[Summary and cases](summary.json) · [V1 full results](../joint_event_hc64_v1/report.json) · [V2 full results](../joint_event_hc64_v2/report.json) · [V3 full results](report.json) · [V1 signed](../joint_event_hc64_v1/signed.tsv) · [V2 signed](../joint_event_hc64_v2/signed.tsv) · [V3 signed](signed.tsv) · [Verification](verification.json) · [Semantic/resource audit](semantic_audit.json)']
(summaryP/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
for v in [1,2]:
    dest=R/f'results/joint_event_hc64_v{v}'/'ANALYSIS.md';dest.write_text(f'# Joint event feedback HC64, stage{v}\n\nThis is a completed fixed developmental stage, not a promoted main method.\n\n[Full three-stage analysis](../joint_event_hc64_v3/ANALYSIS.md) · [Protocol](protocol.json) · [Complete results](report.json) · [Signed outcomes](signed.tsv) · [Severe declines](severe_declines.tsv) · [Verification](verification.json)\n')
print(json.dumps({k:v for k,v in summary.items()if k not in ['stages','cases']},indent=2))
