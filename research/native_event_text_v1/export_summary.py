"""Summarize fixed sealed outcomes and retain negative evidence."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_text_v1.run import D,P,read,put,sha
r=read(P/'report.json');a=read(P/'information_audit.json');s=read(D/'sealed.json');p=read(D/'plan.json');assert r['seal_sha256']==sha(D/'sealed.json')
rows=[]
for arm,m in r['means'].items():
 t=r['paired'].get(arm+'-NativeG6',{}).get('t',{});rows.append(f"|{arm}|{100*m['v']:.4f}|{100*m['t']:.4f}|{t.get('improved',0)}/{t.get('harmed',0)}/{t.get('unchanged',64)}|{t.get('severe_below_minus5pp',0)}|")
text='''# Native temporal event-text intervention: HC64

Status: completed, fixed primary fails. This is repeatedly exposed development,
not independent confirmation. Spatial method and original8 ReferDINO evidence
are unchanged; paper full stays paused until the separately authorized07:00 decision.

## Hypothesis and historical difference

Old25daec0e already performed genuine full/static text Source forwards, and its
q0²/qStatic lost9.3220pp tIoU on exposedHC128. It is NOT treated as an untried idea.
The new intervention changes native RoBERTa token IDs before text encoding: mask
main-clause predicate BPEs, protect the queried nominal subtree, and mask exactly
the same number of nonprotected function-word BPEs as a perturbation control.
Lengths, attention-mask positions and full subject prefix are preserved. All64
have supported matched interventions. Actual text encoder, multimodal fusion and
native two-pass temporal pipeline execute anew at both original offsets/sixlayers.
Only strictly text-independent visual encoders are memoized by actual tensor bytes.

Primary Selective reweights q0 byqControl/qEvent, the difference between full/event
and full/control log responses. Compare individual event/control ratios, old natural
identity-phrase ratio, sharpening and temporally reversed score placement. Unit
coefficient, no acceptance gates, no weight/MBR sweep, no query learning. All legal
native-supported intervals remain available; original physical MBR and same DP.
Text masking is not claimed a calibrated event absence or a causal identification.
CAEM and D-TSG use trained debiasing/event representations; we do not reproduce
those trained estimators by simply taking a frozen likelihood ratio.

## Results

Percent metrics; temporal improved/harmed/tied versus NativeG6.

|Arm|complete vIoU|tIoU|time +/−/=|time loss>5pp|
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Selective−NativeG6 completev−5.3244pp95%CI[-8.1729,-2.7275],
t−8.7718pp[-13.5705,-4.2212]. Temporal17wins/42harms/5ties,32losses>5pp.
Short<=5s t−2.5665pp, long>5s−10.3535pp. Primary−reversed effectt+.8430pp
with a wide interval crossing0. Source is not exceeded. No tuning or promotion.

## Information and cases

Among721,976 candidate pairs matched for physical duration within10% and native
log posterior within.25, feedback macro ranking accuracy46.7367%95%CI[42.7406,50.5599],
versus native52.1362%. It has not exposed reliable event-ranking information that
merely needs a stronger optimizer or different readout.

Keeping nominal token positions intact is insufficient to guarantee the same object
representation: mean counterfactual Source-box IoU vs full is.7794 for event masking,
.9015 for matched function masking,.7241 for identity-only text. Query-mean IoU falls
below.5 on7/1/11 of64 respectively. These are geometry consistency proxies, NOT GT
identity labels. They demonstrate that event-text changes can also alter grounding;
we do not discard these queries to improve averages or claim pure event causality.

RC001 expands to a later end and tIoU drops34.7458%→21.2987%;RC009 moves both
boundaries earlier and drops88.1356%→65.6000%;RC042 drops85.9375%→80.4688%.
RC016 stays at80.4688%. Complete signed outcomes retain all successes and failures.

## Engineering, verification and cost

All64 native full-text hidden/posterior/Source boxes restore bitwise before
feedback. All64 outputs sealed before new scoring.1024 official/DenseTube checks
max9.99e−16; independent scorer uses original frozen spatial geometry and both
actual per-interval and fixed-space measurements. Frozen48+434 source hashes
remain exact. Three algebra/control tests pass. No GT is read in prediction.

First CPU scorer failed because pre-fusion cache contains the original official
NestedTensor class. score_adapter.py installs only its official module path, then
runs the original pinned scorer unchanged. Original failure log/code/seal/predictions
are retained; no inference rerun, fingerprint waiver or scoring-math change.

Actual256 native query forwards/512 text forwards,256 visual misses/768 exact
visual hits;0 expert calls/extra observations/optimizer steps.428.34s entire worker
including startup and12 seconds class loading; mean measured per-query block5.590s
includes all4 text configurations. This is research-panel cost, not isolated-primary
or end-to-end deployment speed. Newly saved pre-fusion features enable subsequent
native capability tests without another RGB/visual/text encoder pass.

Decision: stop this frozen text-ratio branch; do not continue coefficient scanning.
It does not prove native temporal semantics are absent, but does reject this specific
feedback as effective. Next separately fixed branch tests native motion-vs-appearance
variation with FULL text unchanged, zero new experts and exact native no-op recovery.

References: [CAEM](https://github.com/CFM-MSG/CAEM_Code),
[D-TSG](https://arxiv.org/abs/2207.13457).
[All signed outcomes](signed.tsv),[severe tails](severe_declines.tsv),
[postseal information audit](information_audit.json),[fixed protocol](protocol.json).
'''
(P/'ANALYSIS.md').write_text(text)
put(P/'verification.json',dict(queries=64,full_native_hidden_posterior_boxes_bitwise=64,all64_seal_before_GT=True,official_dense_checks=r['checks'],max_metric_error=r['max_metric_error'],unit_tests=3,paper_sources_unchanged=48,provider_sources_unchanged=434,scorer_module_path_adapter_only=True,seal_sha256=sha(D/'sealed.json'),plan_sha256=sha(D/'plan.json'),new_expert_calls=0,promoted=False))
print('Fixed negative analysis exported; no predictions/labels changed')
