"""Freeze the bounded actual LLaVA conditional-time adapter qualification."""
import json
from pathlib import Path
from stvg.runtime import ROOT,sha
from stvg.tube_runner import write


def main():
    path=ROOT/'results/simplepath_llava/qualification_plan.json'
    if path.exists():raise ValueError('Qualification already frozen')
    oldpath=ROOT/'results/llava_st_development/plan.json';old=json.loads(oldpath.read_text())
    publication=ROOT/'outputs/maintenance/simplepath_tubedetr_final_publication.json'
    assert json.loads(publication.read_text())['final_full_results_published'] is True
    seal=ROOT/'runtime/llava_st/checkpoint_seal.json';s=json.loads(seal.read_text());assert s['status']=='complete_verified'
    for name,e in s['files'].items():assert (ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B'/name).stat().st_size==e['bytes']
    tokenizer=json.loads((ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B/tokenizer_config.json').read_text())['added_tokens_decoder']
    assert [tokenizer[str(151660+i)]['content'] for i in range(100)]==[f'<TEMP-{i:03d}>' for i in range(100)]
    files=[*sorted((ROOT/'tools/simplepath_llava').glob('*.py')),*sorted((ROOT/'tools/simplepath_tube/core').glob('*.py')),
           *sorted((ROOT/'tools/reuse_cache').glob('*.py')),*[ROOT/f for f in old['code_sha256'] if not f.startswith('tests/')],
           ROOT/'tests/test_simplepath_llava.py']
    for name,h in old['code_sha256'].items():assert sha(ROOT/name)==h,name
    upstream=json.loads((ROOT/'runtime/upstream_main/a238705af9203c4cda407d7d1199ce18b2e6b72f/configs/default.json').read_text())
    for ds,d in old['datasets'].items():
        for key in ['manifest','full_manifest']:assert sha(ROOT/d[key])==d[key+'_sha256']
        d['configuration']['initialization_gamma']=upstream['datasets'][ds]['initialization_gamma']
    write(path,dict(status='frozen_before_new_real_LLaVA_qualification',backbone='LLaVA-ST-Qwen2-7B',
        upstream_commit='a238705af9203c4cda407d7d1199ce18b2e6b72f',method='SimplePath + one all-video Native2D prototype pair',
        datasets=old['datasets'],dataset_order=['vidstg','hc2'],teacher_fingerprint=old['teacher_fingerprint'],
        checkpoint_repo=old['checkpoint_repo'],checkpoint_revision=old['checkpoint_revision'],
        checkpoint_seal_sha256=sha(seal),checkpoint_disk_publisher_hash_verified_previously_no_16GB_rehash=True,
        previous_plan_sha256=sha(oldpath),prior_TubeDETR_final_publication_sha256=sha(publication),
        prior='true autoregressive likelihood: all99 legal start tokens, native separator, separate end distribution conditional on each start; normalize native legal distinct physical endpoints',
        engineering_recovery=dict(reason='actual native tokens use three-digit spelling TEMP-000; v1 guard failed before any generation',archive='outputs/simplepath_llava_engineering_recovery/token_spelling_v1',prior_native_generations=0,prior_development_predictions=0,prior_scores=0),
        prior_temperature=1,Source_generation_unchanged_temperature=.01,
        not_independent_endpoint_outer_product=True,not_single_span_pseudo_distribution=True,
        local_representation='actual first spatial STP100x81x3584 with LAPE before language/text fusion, duplicate physical frames averaged',
        fixed=dict(kappa=64,beta=1,rounds=1,context=1.5,observed_weight=.5,unobserved_weight=.5,link_weight=.25),
        qualification=dict(inputs='first original dev128 query each domain',native_generations=4,
             cold=2,in_memory_SigLIP_cache_replays=2,conditional_start_branches_per_generation=99,
             maximum_separator_tokens=16,maximum_conditional_single_token_forwards=4*99*17,
             new_teacher_calls=0,GT_reads=0,query_substitution=False,
             requirements=['Source/native interval/physical grid/eight observations exact old full',
                'conditional branch at actual generated start matches actual native end logits at atol.002',
                'cold/cache Source/features/q0/conditional rows/Full bitexact',
                'model state hash unchanged; actual local ROI and new auxiliary path finite']),
        cache_policy='old exact teacher cache read-only; qualification visual prefix replay in RAM; no new large disk cache appends',
        reserve_bytes=25<<30,new_teacher_calls=0,network_updates=0,retuning=False,
        claim_scope='shared merged checkpoint, prior Vid-derived instruction exposure and development/full exposure; no strict unseen-domain claim',
        code_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))},
        upstream_code_sha256=old['upstream_code_sha256']))
    print('qualification frozen',sha(path),flush=True)


if __name__=='__main__':main()
