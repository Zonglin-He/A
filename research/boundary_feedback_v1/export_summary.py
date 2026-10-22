"""Report saved results, engineering recovery and exact preservation checks."""
import sys,json,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import R,D,P,ARMS,sha,read,put,verify
import numpy as np
r=read(P/'report.json');f=read(P/'failure_analysis.json');plan=verify()
receipt=read(D/'score_receipt.json');assert receipt['report_sha256']==sha(P/'report.json')
assert read(D/'sealed.json')['count']==64 and read(D/'state.json')['status']=='completed_scored'
failed_log=D/'score.log';assert 'not JSON serializable' in failed_log.read_text()
assert all(sha(R/name)==h for name,h in plan['code'].items())
paper=read(R/'outputs/tastvg_paper_v1/state.json');assert paper['status']=='paused_by_user'
def identity_running(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    return path.exists() and int(path.read_text().rsplit(')',1)[1].split()[19])==identity['startticks']
assert not identity_running(paper['controller'])
assert paper['pause']['preserved_vid_receipts']==1373
verification=dict(status='verified',queries=64,NativeG6_intervals_and_boxes_bitwise_identical=64,
                  new_Source_forwards=0,new_decoder_forwards=0,new_expert_calls=0,query_updates=0,
                  CPU_only=True,all64_sealed_before_labels=True,GT_in_inference=False,
                  official_dense_checks=r['official_dense_checks'],max_metric_error=r['max_error'],
                  max_native_class_mass_error=f['max_class_mass_error'],
                  paper_runtime_files=len(plan['paper_code']),provider_files=len(plan['provider_sources']),
                  all_frozen_hashes_unchanged=True,full_state='paused_by_user',preserved_vid=1373,
                  HC1902='sealed_scored_retained',original_controller_exited=True,
                  serialization_recovery=dict(stage='final JSON only, completed64 scoring iterations',
                      original_log_sha256=sha(failed_log),original_code_retained=True,
                      adapter_sha256=sha(Path(__file__).with_name('score_json_adapter.py')),
                      prediction_seal_unchanged=True,CPU_rescore_only=True),
                  independent_MBR_check=read(D/'independent_numeric_check.json'),
                  plan_sha256=sha(D/'plan.json'),seal_sha256=sha(D/'sealed.json'),report_sha256=sha(P/'report.json'))
put(P/'verification.json',verification);put(D/'verification.json',verification)
arms=ARMS+['Source'];lines=['query\tmethod\tvIoU\ttIoU\tunobserved_sIoU\tdelta_v_vs_G6_pp\tdelta_t_vs_G6_pp\tboundary_MAE_seconds'];severe=[lines[0]]
for x in r['rows']:
    for arm in arms:
        v=x['values'][arm];b=x['values']['NativeG6'];line=f"{x['query']}\t{arm}\t{v['v']:.12g}\t{v['t']:.12g}\t{v['unobserved_s']:.12g}\t{100*(v['v']-b['v']):.12g}\t{100*(v['t']-b['t']):.12g}\t{v['boundary_MAE_seconds']:.12g}"
        lines.append(line)
        if v['v']-b['v']<-.05 or v['t']-b['t']<-.05:severe.append(line)
(P/'signed.tsv').write_text('\n'.join(lines)+'\n');(P/'severe_declines.tsv').write_text('\n'.join(severe)+'\n')
lines=['# HC64: dense boundary feedback with coarse posterior preservation','',
       '**Executed on all64; not admitted to the main method.** The primary has small positive means versus Global6, but paired CIs span zero, long-event tIoU falls, and the protection-only old-token control has higher overall tIoU. Full remains paused at HC1902 sealed/scored and Vid1373 saved.','',
       '## Failure addressed and fixed design','',
       'The previous predictive event feedback cannot distinguish boundaries sharing the same eight-anchor membership. In26 harmed Direct cases,20 contain a better-than-native interval with exactly the selected membership. Increasing query optimization would fit that ambiguous supervision more fully. This experiment instead tests dense changes in the actual text-conditioned temporal hidden already saved at the two native224 offsets. No new query fitting, RGB decoding, encoder/decoder/head execution or expert call occurs.','',
       'For each offset and each of the six layers, normalize hidden vectors and take two frames on either side of a potential cut. The contrast is','',
       r'\[d(t)=\tfrac12\|\mu_{\mathrm{left}}(t)-\mu_{\mathrm{right}}(t)\|_2^2.\]','',
       'Average layers, convert to bounded centered percentile ranks, interpolate in physical frame time, then average offsets. Never interleave offsets before computing changes. A candidate receives its start plus end change score. Unknown clip-edge context is neutral. Window2 and unit reward power are fixed before predictions; no scale/weight sweep.','',
       'The primary preserves native probability of every eight-anchor-membership class C:','',
       r'\[q_B(I)=q_0(C(I))\frac{q_0(I)\exp b(I)}{\sum_{J\in C(I)}q_0(J)\exp b(J)}.\]','',
       'This preserves class probabilities, not the eventual MBR decision or its membership. Original physical MBR and interval-conditioned complete-box DP remain unchanged. Also report the fixed NativeG6 spatial trajectory to isolate temporal effects.','',
       'This borrows the local self-similarity principle of [UBoCo, CVPR2022](https://arxiv.org/abs/2111.14799). Its [released encoder](https://github.com/musicalOffering/uboco/blob/main/network.py) is trained to form useful similarity structure; that supporting condition is not reproduced here. Generic changes and text-conditioned hidden changes are not guaranteed to be the described event boundaries. [DSE-VTG](https://arxiv.org/html/2609.08850v1) combines aligned video-language features, multiple temporal scales and query adaptation; its query optimization does not justify fitting unreliable feedback more strongly in our different interface.','',
       '## Complete fixed comparison','',
       '| Arm | Complete vIoU % | tIoU % | Unobserved sIoU % | Boundary MAE s |','|---|---:|---:|---:|---:|']
for a in arms:
    m=r['means'][a];lines.append(f"| {a} | {m['v']*100:.4f} | {m['t']*100:.4f} | {m['unobserved_s']*100:.4f} | {m['boundary_MAE_seconds']:.4f} |")
lines += ['', 'NativeG6 is bitwise identical to the saved baseline. BoundaryCell is the preregistered primary. BoundaryGlobal removes coarse-mass preservation. ShiftCell retains the same values and budget but cyclically shifts them by one quarter of the timeline. RawCell applies identical mass preservation to the old raw event contrast; it tests whether conservatism alone explains recovery. Source is the original frozen backbone.','',
          '| Primary paired difference | Mean pp | Parent-cluster95% CI pp |','|---|---:|---|']
for comp in ['BoundaryCell-NativeG6','BoundaryCell-ShiftCell','BoundaryCell-RawCell','BoundaryCell-Source']:
    for metric in ['v','t']:
        x=r['paired'][comp][metric];lines.append(f"| {comp}, {metric} | {x['mean']:+.4f} | [{x['CI95'][0]:+.4f}, {x['CI95'][1]:+.4f}] |")
lines += ['', 'One query per parent;10,000 fixed-seed paired parent bootstrap draws. These are descriptive intervals on previously exposed development, not independent confirmation. No arm is selected for full inference from this table.','',
          '## Does the new feedback actually distinguish boundaries?','',
          f"It varies in {f['classes_distinguished']}/{f['multi_candidate_classes']} multi-candidate anchor classes, and changes37/64 final intervals. Native class mass is preserved to {f['max_class_mass_error']:.3g}. Thus the original quantization ambiguity is measurably reduced; discrimination alone is not correct ranking.",
          '', 'Offline posterior-expected tIoU increases +.1956pp versus native (CI[+.0566,+.3375]) and +.2671pp versus the shifted signal (CI[+.0513,+.5039]). This indicates useful local information in this development panel. But final MBR tIoU gains only+.0817pp and its CI crosses zero. The new signal is neither sufficiently reliable nor clearly superior to RawCell. Do not call expectation gains final localization gains.','',
          '| GT-duration group, diagnostic only | Queries | Primary Δt vs G6 pp | RawCell Δt vs G6 pp |','|---|---:|---:|---:|']
for group,n in [('short_le5s',13),('long_gt5s',51)]:
    z=r['groups_vs_NativeG6'][group];lines.append(f"| {group} | {n} | {z['BoundaryCell']['t']:+.4f} | {z['RawCell']['t']:+.4f} |")
lines += ['', 'GT duration never selects a branch or parameter. Complementary group averages do not authorize a GT-conditioned mixture.','',
          '## Concrete corrections and failures','',
          '| Query | Δt vs G6 pp | Δv vs G6 pp | Start MAE change, frames | End MAE change, frames |','|---|---:|---:|---:|---:|']
for x in f['cases']:
    lines.append(f"| {x['query']} | {x['delta_t_pp']:+.4f} | {x['delta_v_pp']:+.4f} | {x['start_MAE_delta_frames']:+d} | {x['end_MAE_delta_frames']:+d} |")
lines += ['', 'RC025/RC029 were harmed by the previous raw-query fit and are improved here without query optimization. RC009 improves by moving the end closer. RC042 contracts both ends toward stronger feature-change evidence but increases both boundary errors; RC016 similarly stops too early. These are measured feature/annotation misalignments, not a visual assertion about the exact action semantics.','',
          'Across all17 temporal harms,10 intervals shorten and7 lengthen;13 move to a higher boundary score. The issue is not only systematic shrinking. The current change score has no learned start-versus-end or specific-event criterion, and separate endpoint peaks can be internal changes. Rank normalization also does not certify evidential strength. The original time posterior anchors semantics only imperfectly.','',
          'Primary complete vIoU improves17/declines15/ties32, with0 losses>5pp. tIoU improves19/declines17/ties28, with2 losses>5pp. All arms and Source retain signed rows and severe declines. RawCell has a larger overall temporal mean but still harms short events; it is not silently selected as the new method.','',
          '## Cost, validation and disposition','',
          f"Cached CPU feedback averages {r['mean_feedback_seconds']*1000:.3f}ms/query (p95 {r['p95_feedback_seconds']*1000:.3f}ms). All64 five-arm readouts/spatial paths/files took {read(D/'sealed.json')['seconds']:.3f}s total. This excludes creation/loading of the backbone cache and is not end-to-end deployment acceleration.",
          '', 'All64 predictions seal before new labels.768 independent official/DenseTube checks agree to6.66e-16. Native64 interval/box controls are bitwise exact, all48 paper runtime and434 provider files match their locked hashes. The scorer first failed only at final NumPy integer JSON serialization; original logs/code retained, an independent CPU serializer adapter completes the same scoring without changing any prediction, plan or fingerprint.','',
          '**Decision:** this bounded candidate adds real dense boundary information and produces actual corrections, but has not established a reliable net boundary improvement. Do not register it, increase query steps, scan weights, or resume full. The unresolved mechanism is semantic validation and joint evaluation of the two event boundaries, not missing optimizer iterations. This result does not rule out trained/appropriately aligned boundary features or all cache-based temporal feedback.','',
          'Reproduce: `python -B research/boundary_feedback_v1/test_core.py`; `run.py prepare`, `run.py predict`, then independent CPU `score_json_adapter.py` after the seal. Assets and prior receipt-bound captures are required. Public files contain no raw captions, media, GT coordinates, weights or features.','',
          '[Protocol](protocol.json) · [All results](report.json) · [Signed rows](signed.tsv) · [Severe declines](severe_declines.tsv) · [Failure analysis](failure_analysis.json) · [Verification](verification.json)']
(P/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(verification))
