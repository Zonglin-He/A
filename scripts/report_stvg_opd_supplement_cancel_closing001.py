"""Current evidence decision and public supplement report, without a new route."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.audit_stvg_opd_supplement_cancel_public_v1 import REL, read as public_read
BASE = ROOT/'artifacts/stvg_opd_supplement_v1'
OUT = ROOT/REL

def run():
    assert read(BASE/'ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json')['status'] == 'pass'
    references = {
        'C': 'artifacts/stvg_opd_claim_strengthening_v1',
        'N': 'artifacts/stvg_opd_mechanism_review_v1',
        'P7': 'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1',
        'P9': 'artifacts/stvg_opd_evidence_actionability_audit_v1',
        'P0': 'artifacts/stvg_opd_state_conditional_actionability_v1',
        'cost32': 'artifacts/stvg_opd_supplement_v1/cost'}
    bindings = {}
    for key, rel in references.items():
        ns = ROOT/rel; v = read(ns/'FINAL_COMPLETION.json')
        assert v['status'] == 'complete' and all(v[k] for k in ['actual_root','actual_PDF_pixel_view','actual_remote_publication','actual_archive'])
        bindings[key] = dict(final_sha256=sha(ns/'FINAL_COMPLETION.json'), root_closing_sha256=sha(ns/'ROOT_CLOSING.json'))
    p7 = ROOT/references['P7']/'P7_B/CANDIDATE_MECHANISM_DECISION.json'
    p9 = ROOT/references['P9']/'CANDIDATE_DECISION.json'
    p0 = ROOT/references['P0']/'MECHANISM_SCOPE_DECISION.json'
    assert not read(p7)['misallocation_hypothesis_supported_in_both_added96']
    assert not read(p9)['heldout_signal_supports_bounded_counterfactual_selection_test']
    assert not read(p0)['causal_LN_attribution']
    decision = dict(status='existing_evidence_decision_recorded_future_main_protocol_pending',
        scope='record completed evidence constraints after genuine C/N/cost FINAL; no unspecified new experiment launch',
        bindings=bindings, P7_decision_sha256=sha(p7), P9_decision_sha256=sha(p9), P0_decision_sha256=sha(p0),
        fixed_P7_recipe_full_official_expansion_condition_met=False,
        existing_P9_heldout_gate_branch_condition_met=False,
        P0_natural_order_association_is_causal_LN_evidence=False,
        C_Expanded_stable_Full_over_reset_in_both_domains_established=False,
        N_universal_query_component_necessity_established=False,
        original_baseline_remaining_formal_cancelled=True,
        existing_evidence_decision_complete=True, future_method_and_main_protocol_complete=False,
        whole_followup_request_complete=False, future_main_launched=False,
        new_causal_state_intervention_launched=False, new_gate_created=False, new_teacher_added=False,
        production_promoted=False, manuscript_written=False, CPU_only=True, GT_read=False, time=time.time())
    for p in [BASE/'FOLLOWUP_EVIDENCE_DECISION.json', OUT/'FOLLOWUP_EVIDENCE_DECISION.json']:
        if p.exists(): assert {k:v for k,v in read(p).items() if k != 'time'} == {k:v for k,v in decision.items() if k != 'time'}
        else: write(p, decision)
    summary = public_read(OUT/'ORDER_STATISTICS.json.gz')['statistics']
    report = ['# Complete existing S supplement after explicit formal cancellation', '',
        'This closes the current authorized evidence scope. All original 2,304 Full/reset/full-carry order fits, the 18 fixed development configurations and 1,152 development arrivals, saved evidence, six fixed real cases and original 32 audited timing paths are retained. The two HC selected streams are each sealed at 10,446 arrivals and the Vid TENT prefix is preserved at 13,747. The remaining selected formal evaluation was explicitly cancelled; no 82,710-arrival global or missing formal GT evaluation is claimed.', '',
        '## Direct order comparisons', '',
        'Each target has 128 historically exposed parents, one fixed query each and three real online permutations. Parent outcomes average the same three orders; the orders are not independent populations. Intervals use the original 10,000 paired parent bootstrap, seed 20261011, unadjusted for multiple comparisons. Differences are percentage points.', '',
        '| Target | Comparison | Mean (pp) | 95% CI | Negative parents |',
        '|---|---|---:|---|---:|']
    for ds in ['hc2','vidstg']:
        s = summary[ds]['contrasts']['pooled']['metrics']
        for f, label in [('Full_minus_alpha0_v','Full - reset'),('Full_minus_alpha1_v','Full - unattenuated carry')]:
            x = s[f]; report.append(f"| {ds} | {label} | {x['mean']*100:+.3f} | [{x['ci95'][0]*100:+.3f}, {x['ci95'][1]*100:+.3f}] | {x['negative_parent_count']} / 128 |")
    report += ['', 'Both pooled Full-minus-reset intervals cross zero. Full-minus-unattenuated-carry pooled intervals are positive in both these 128-parent three-order cohorts. C Expanded-State uses a different 205/700-parent, two-order population: its reset contrasts cross zero in both domains and its carry advantage is established only in VidSTG. Preserve these distinct populations; do not pool them as independent replications.', '',
        'Before-minus-Frozen is the entering-state readout; After-minus-Before is the current update. Neither replaces the direct final-output Full-minus-reset comparison. All 2,304 positive, negative, neutral, no-information and undefined diagnostics remain in ORDER_ROWS with each-order/pooled absolute, current-update and direct statistics and original sixteen-arrival progression.', '',
        '## Development grid and cancelled formal custody', '',
        'DEVELOPMENT_ROWS preserves every one of the 18 predeclared target/configuration grids and all 1,152 outcomes, including failed-to-improve configurations. The portable public audit independently reproduces every scalar summary and the locked tie rule: highest development parent-macro final vIoU; ties within 1e-12 use fewer parents harmed by more than 20 pp, lower learning rate and then lower margin. These 32 development parents per target are historical development data, not fresh confirmation.', '',
        'All 34,839 already accepted formal artifact bytes and receipts were rehashed without payload deserialization: two HC streams of 10,446 and the Vid TENT prefix of 13,747. All 1,152 development artifact bytes and receipts were also checked. The original stopped baseline PIDs, start identities and commands are retained, along with optimizer memory and the original lease. These formal predictions remain unscored custody evidence; no complete selected-baseline performance table is fabricated.', '',
        '## Saved evidence and figures', '',
        'SAVED_FRAME_EFFECT_ROWS retains all 1,244 historical observed/unobserved scalar outcomes, including unavailable values; their parent bootstrap and every positive/negative parent were independently recomputed. This is within-query After-minus-Before evidence. The original three numerical cohorts are distinct; same-domain clean VidSTG has 732 parents, while each cross-domain cohort has 128. The table-rounding source is independently reaggregated over all 237 parents: 34.229, correcting the old 34.230 display without changing original outcomes.', '',
        'Exact PDF byte hashes bind six existing PDFs and all eleven actually viewed pages: two order pages, two saved numerical/interface pages, six private fixed-case pages and the original cost page. The actual prior view receipts are reused; no new render or model run is claimed. Five numerical/interface PDFs are public. The six-case RGB/query/GT-geometry PDF remains local, with anonymous integrity/provenance metadata only.', '',
        'The original cost32 result remains a separate complete actual measurement: 32 complete Full timed paths, 112 real detector calls and 680 actual backward updates among 800 round records. All slow/no-DINO/no-admitted/no-backward paths are public in cost_complete at its verified closing commit. It is not zero fresh detector cost, a production throughput benchmark or an OOM guarantee.', '',
        '## Current evidence decision and remaining main scope', '',
        'The locked P7 observation-coupling expansion condition was not met. P9 did not establish its heldout gate branch; P0 showed natural order/state association, not causal LN attribution. N found a positive pooled Full-versus-LN-only difference in VidSTG and an HC interval crossing zero. None establishes a universal component requirement, safe gate, new state intervention or an automatic full official main expansion.', '',
        'The existing evidence decision is now recorded in FOLLOWUP_EVIDENCE_DECISION. A future method and main protocol remain a separate pending scope requiring concrete explicit authorization. No new GPU/model/DINO/optimizer/raw-GT scoring, teacher, gate, causal state intervention, parameter search, manuscript or CURRENT promotion occurred in this publication closure. The cancelled formal baseline is not resumed.', '',
        'ROOT_CLOSING / FINAL_COMPLETION require the actual public scalar audit, every remote byte/SHA256/Git blob/mode/tree, both refs and archive check-snapshot-check. This report records the bounded current scope; it does not declare future main work complete.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    stage = read(BASE/'orders/DESIGN_LOCK.json')['stages']
    sources = {}
    for n, d in stage.items():
        b = read(BASE/'orders/stages'/n/'PREDICTION_BARRIER.json')
        sources[n] = dict(target=d['dataset'], source=d['source'], source_model_state_sha256=b['source_model_state_sha256'], expert_model_state_sha256=b['expert_model_state_sha256'])
    write(OUT/'SOURCE_STATE_BINDINGS.json', dict(status='pass', source_state_bindings=sources, runtime_sha256=sha(BASE/'orders/RUNTIME_LOCK.json'), no_weight_payload_exported=True))
    c = read(BASE/'saved_evidence/CASE_COMPLETION.json')
    write(OUT/'FIXED_PRIVATE_CASE_PROVENANCE.json', dict(status='pass', scope='anonymous integrity only; no private RGB/query/GT coordinates exported',
        fixed_cases=6, frame_panels=48, private_PDF_sha256=c['private_PDF_sha256'], case_lock_sha256=c['original_case_lock_sha256'],
        figure1_provenance_sha256=c['figure1_provenance_sha256'], actual_original_pixel_view_sha256=sha(BASE/'saved_evidence/ROOT_VISUAL_REVIEW.json'),
        actual_prior_pixel_view=True, new_model_calls=0, media_exported=False, GT_geometry_exported=False))
    print('S_EVIDENCE_DECISION_RECORDED_MAIN_PROTOCOL_REMAINS_SEPARATE')

if __name__ == '__main__': run()
