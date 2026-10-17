"""Anonymous R1 engineering preference report; no GT helper or model invocation."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_semantic_preference_v1';OUT=ROOT/'results/stvg_semantic_preference/2026-10-17/r1_interface8'
def main():
    from scripts.stvg_semantic_preference_contract_v1 import categorical_from_logprobs,restore_candidate_order
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    d=read(BASE/'teachers/R1/QUALIFICATION_COMPLETION.json');root=read(BASE/'R1_ROOT_READBACK.json')
    assert d['status']=='pass' and root['count']==d['count']==8
    rows=[];checks=0
    for i,f in enumerate(d['files'],1):
        assert sha(ROOT/f['path'])==f['sha256'];r=read(ROOT/f['path']);es=r['evidence'];pp=[]
        for e in es:
            p=categorical_from_logprobs(e['raw_label_logprobs']);assert p==e['conditional_label_posterior']
            mapped=restore_candidate_order(p,e['display_order']);assert mapped==e['candidate_order_posterior'];pp.append(mapped);checks+=2
        avg=[math.fsum(e[k] for e in pp)/len(pp) for k in range(len(pp[0]))] if pp else None
        if avg:
            assert avg==r['permutation_averaged_preference']
            tv=.5*math.fsum(abs(a-b) for a,b in zip(*pp));assert abs(tv-r['permutation_total_variation'])<1e-14
            choices=[max(range(len(p)),key=p.__getitem__) for p in pp];winner=max(range(len(avg)),key=avg.__getitem__)
        else:tv=None;choices=[];winner=None
        rows.append(dict(source_alias=f'R{i:03d}',dataset=r['dataset'],support_count=len(avg)-1 if avg else 0,
            status=r['status'],abstention_probability=None if avg is None else avg[0],averaged_choice_is_abstention=winner==0,
            permutation_total_variation=tv,permutation_top_choice_changed=len(set(choices))>1,
            conditional_entropy_nats=None if avg is None else -math.fsum(x*math.log(x) for x in avg if x>0),
            allowed_label_probability_mass=[e['allowed_label_probability_mass'] for e in es],
            full_vocabulary_top_label_allowed=[e['full_vocabulary_argmax_is_allowed'] for e in es],
            input_tokens=[e['input_audit']['input_tokens'] for e in es],wall_seconds=[e['wall_seconds'] for e in es],
            peak_GPU_allocated_bytes=[e['peak_GPU_allocated_bytes'] for e in es],actual_prefill_calls=len(es),
            semantic_correctness='unknown_no_GT_or_independent_identity_labels',Student_effect='not_measured'))
    summary=[]
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];known=[r for r in rr if r['permutation_total_variation'] is not None]
        summary.append(dict(dataset=ds,parents=len(rr),actual_prefill_calls=sum(r['actual_prefill_calls'] for r in rr),
            mean_permutation_total_variation=math.fsum(r['permutation_total_variation'] for r in known)/len(known),
            maximum_permutation_total_variation=max(r['permutation_total_variation'] for r in known),
            permutation_top_choice_changes=sum(r['permutation_top_choice_changed'] for r in rr),
            averaged_abstentions=sum(r['averaged_choice_is_abstention'] for r in rr),
            allowed_label_mass_min=min(x for r in rr for x in r['allowed_label_probability_mass']),
            allowed_label_mass_max=max(x for r in rr for x in r['allowed_label_probability_mass']),
            all_full_vocabulary_argmax_allowed=all(x for r in rr for x in r['full_vocabulary_top_label_allowed'])))
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'INTERFACE_RESULTS.json',dict(status='complete_engineering_interface_only',count=8,rows=rows,summary=summary,
        raw_responses_and_token_logits_not_exported=True,conditional_probability_not_calibrated_truth=True,GT_read=False))
    write(OUT/'COST.json',dict(actual_prefill_calls=d['actual_prefill_calls'],free_generations=0,new_Student_fits=0,new_Temporal_or_DINO=0,
        actual_model_load_seconds=d['load_seconds'],actual_worker_wall_seconds=read(BASE/'R1_COMPLETION002.json')['compute_seconds'],
        per_request_wall_seconds=math.fsum(x for r in rows for x in r['wall_seconds']),full_model_forward_not_only_last_projection=True,
        original_candidates_and_Source_cost_not_zero=True))
    write(OUT/'ROOT_SCALAR_READBACK.json',dict(status='pass',count=8,checks=checks,no_new_GT_reader=True,
        single_token_IDs_and_restored_permutations_checked=True,all_sources_retained=True))
    lines=['# R1 semantic preference: fixed8 interface feasibility','',
        'HC4/Vid4 first parents of the exposed32, slot0, same frozen weight-free E1 candidate pool (first4 legal existing centers), original4 context frames plus labelled full anchor. Ordinary and reverse arbitrary labels; 0 explicitly permits abstention. Frozen STVG-R1-7B/nativeBF16, exact original checkpoint and clock, no new model or Prompt search. This changes the scheduling of the R1 half only; Qwen download continues and its CPU controller was identity-checked before a bounded priority handoff.', '',
        'Read actual full-vocabulary next-token log probabilities for single-token labels0..K. The reported posterior is conditional on the allowed label vocabulary. Allowed-label probability mass and full-vocabulary argmax are recorded separately, so normalization cannot hide failure to use the requested labels. No verbal confidence is used. These are preferences, not calibrated correctness probabilities.', '',
        '| Domain | Parents | Forward calls | Mean order TV | Max order TV | Changed top choice | Averaged abstentions | Min allowed mass |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summary:lines.append(f"|{x['dataset']}|{x['parents']}|{x['actual_prefill_calls']}|{x['mean_permutation_total_variation']:.6f}|{x['maximum_permutation_total_variation']:.6f}|{x['permutation_top_choice_changes']}|{x['averaged_abstentions']}|{x['allowed_label_mass_min']:.6f}|")
    lines+=['','Engineering qualification passed: the native model can supply finite, aligned candidate/abstention probabilities. Label-order sensitivity remains and is retained; permutation averaging does not prove semantic correctness. Candidate coverage, identity/role quality and Student improvement are untested. No GT, free generation, Student fit, optimizer/response change or official fullset occurred. This report is not a completed2x2 factorial or a method promotion.', '',
        'R1 output hashes, input/template/checkpoint and runtime are sealed privately. Raw query, model token logits, private anchor/context images, GT and weights are excluded. Anonymous scalar records include all eight parents and the abstention. Next: lock same-support geometric/preference Student conversion and same-information propagation; fixed-support weighted Gaussian NLL must not be described as on-policy resampling. Qwen half will reuse these R1 outputs after its own direct32 comparison; no repeated R1 request.']
    doc=ROOT/'docs/STVG_SEMANTIC_PREFERENCE_R1_INTERFACE_002.md';doc.write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));xs=list(range(8))
    axes[0].bar(xs,[r['permutation_total_variation'] for r in rows]);axes[0].set_ylabel('Total variation after restoring candidate identity');axes[0].set_ylim(0,1);axes[0].set_title('Sensitivity to reversed labels')
    axes[1].bar(xs,[r['abstention_probability'] for r in rows]);axes[1].set_ylabel('Mean conditional probability of label0');axes[1].set_ylim(0,1);axes[1].set_title('Abstention retained')
    for ax in axes:ax.set_xticks(xs,[r['source_alias'] for r in rows],rotation=45)
    fig.tight_layout();fig.savefig(OUT/'interface_diagnostics.png',dpi=160);plt.close(fig)
    for n in ['stvg_semantic_preference_contract_v1.py','test_stvg_semantic_preference_contract_v1.py','stvg_semantic_preference_interface_v2.py','continue_stvg_semantic_preference_r1_002.py','continue_stvg_semantic_preference_interface_002.py','report_stvg_semantic_preference_r1_002.py']:
        p=OUT/'code'/n;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/n).read_bytes()))
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT)),'protocols/stvg_semantic_preference_v1.md','protocols/stvg_semantic_preference_r1_priority002.md'}
    verify_privacy(paths);write(BASE/'publication/r1_interface8/EXPORT_MANIFEST.json',dict(status='complete',count=8,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    print('R1_INTERFACE_EXPORT_READY',summary)
if __name__=='__main__':main()
