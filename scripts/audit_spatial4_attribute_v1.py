"""F37 independent receipts, arithmetic, aggregation, budget and parent audit."""
import collections
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.prepare_spatial4_attribute_v1 import OUT,F35,F36
from scripts.run_spatial4_attribute_v1 import plan,dest
from scripts.run_spatial10_components_v1 import existing
from scripts.audit_simplification_partial_v1 import audit_fit


def aggregate_check(rows,summary,seed=20260914):
    cells=0;errors=[]
    for typ,groups in [('arms',summary['arms']),('contrasts',summary['contrasts'])]:
        for name,metrics in groups.items():
            for metric,expected in metrics.items():
                values=collections.defaultdict(list)
                for row in rows:
                    if typ=='arms':v=row['arms'].get(name,{}).get(metric)
                    else:
                        a,b=name.split(' - ')
                        av=row['arms'].get(a,{}).get(metric);bv=row['arms'].get(b,{}).get(metric)
                        v=av-bv if av is not None and bv is not None else None
                    if v is not None:values[row['group']].append(v)
                if not values:
                    assert expected['mean'] is None and expected['sources']==0
                    continue
                vector=np.asarray([sum(values[k])/len(values[k]) for k in sorted(values)])
                query_values=[v for vv in values.values() for v in vv]
                rng=np.random.default_rng(seed)
                boot=vector[rng.integers(0,len(vector),(10000,len(vector)))].mean(1)
                ci=np.quantile(boot,[.025,.975])
                err=max(abs(vector.mean()-expected['mean']),max(abs(ci-expected['ci95'])),
                        abs(np.mean(query_values)-expected['query_mean']))
                assert err<1e-12 and len(vector)==expected['sources']
                assert abs(float(np.median(vector))-expected['median'])<1e-12
                assert int((vector>.001).sum())==expected['wins']
                assert int((vector<-.001).sum())==expected['harms']
                assert int((vector<-.05).sum())==expected['negative_gt5pp']
                errors.append(float(err));cells+=1
    return cells,max(errors,default=0.)


def main():
    torch.set_num_threads(2)
    p=plan();source_rows=[r for rr in p['rows'].values() for r in rr]
    ext=[r for rr in p['space_extension'].values() for r in rr]
    allr=source_rows+ext
    assert len(allr)==113==len({r['group'] for r in allr})==len({r['key'] for r in allr})
    assert {r['key'] for r in p['H_rows']}<= {r['key'] for r in allr}
    for c,rr in p['rows'].items():
        dev=[r for r in rr if r['f37_role']=='development']
        ev=[r for r in rr if r['f37_role']=='evaluation']
        assert len(dev)==8 and len(ev)==16
        for field in ('group','source'):
            assert not {r[field] for r in dev}&{r[field] for r in ev}
            assert not {r[field] for r in rr}&{r[field] for r in p['space_extension'][c]}
        assert not {r['input']['video_sha256'] for r in rr}&{r['input']['video_sha256'] for r in p['space_extension'][c]}
    counts=dict(fits=0,backward=0,new_DINO=0);fit_counts=collections.Counter()
    fits=[];errors=[];replay=0;reused=collections.Counter();observations=0
    for stage,rr in [('space',ext),('practical_dev',[r for r in source_rows if r['f37_role']=='development']),
                     ('practical_eval',[r for r in source_rows if r['f37_role']=='evaluation']),('practical_guard',p['H_rows'])]:
        for r in rr:
            x=load(dest(stage,r['key']));assert existing(dest(stage,r['key']))
            assert not x['GT_online'] and x['actual_new_DINO']==0
            assert x['qa']==load(x['parent']['path'])['qa']
            assert sha(x['parent']['path'])==x['parent']['sha256']
            if stage=='space':
                assert x['expert']['new_DINO']==0
                observations+=len(x['expert']['reused'])
                s=x['fits']['S4'];assert s['anchors']==x['expert']['anchors']['single4']
                assert s['planned']==4 and s['gamma']==0 and s['kappa'] is None
                assert all(a['weight']==1 for a in s['anchors'])
                assert s['lr']==(.01 if r['key'].startswith('hcstvg1') else .1)
            for name,z in x['fits'].items():
                if name in x['reuse']:
                    reused[stage+'/'+name]+=1;continue
                errors.append(audit_fit(z));fits.append(z)
                counts['fits']+=1;fit_counts[stage]+=1;counts['backward']+=z['backwards']
                assert not z['GT_online'] and z['student_output_only']
                assert z['parameter_count'] in (1792,2820)
                assert z['steps']==10
                assert z['final']['indices']==x['native_indices']
                assert all(torch.equal(a,b) for a,b in zip(z['final']['logits'],x['native_logits']))
                if stage!='space':
                    assert z['anchors']==x['fits']['S0']['anchors']
                    assert z['planned']==4 and z['gamma']==1e-4
            for a in x['audits'].values():
                assert a['full_model'] and a['source_restored'] and not a['GT_online']
                assert a['box_max_error']==a['logit_max_error']==0;replay+=1
    noops=list((OUT/'noops').glob('*.pt'));assert len(noops)==4
    for path in noops:
        x=load(path);z=x['fit'];assert existing(path)
        counts['fits']+=1;fit_counts['noops']+=1;counts['backward']+=z['backwards']
        assert z['state_delta']==0 and z['restore_exact'] and not z['GT_online']
        assert torch.equal(z['path'][0]['boxes'],z['final']['boxes'])
        assert x['audit']['source_restored']
        assert x['audit']['box_max_error']==x['audit']['logit_max_error']==0;replay+=1
    attr=read(OUT/'ATTRIBUTE_INTERFACE_AUDIT.json');assert attr['no_temporal_fits']
    checks=0;diagnostic=0;time_bias=[];grad=[];visual=[]
    for x in attr['queries']:
        assert x['parameter_fits']==0 and not x['GT_online'] and x['no_labels_assigned']
        assert x['checkpoint_branch']=='model_ema'
        assert sha(x['checkpoint']['path'])==x['checkpoint']['sha256']
        diagnostic+=x['diagnostic_backwards']
        for n in x['native_checks']:
            assert n['native_logit_max_error']==n['native_map_max_error']==n['manual_readout_max_error']==0
            checks+=1
        for n in x['rows']:
            assert n['donor_reencoded_recipient_query'] and n['donor_caption']==x['caption']
            assert n['donor_group']!=x['group']
            assert n['original_memory_sha']!=n['donor_memory_sha']
            assert n['epsilon_clamped_fraction']==0
            time_bias.append(n['proposed_bias_max_logit_change']);grad.append(n['head_gradient_l2'])
            visual.append(n['wrong_visual_probability_max'])
    assert checks==diagnostic==16
    counts['backward']+=diagnostic
    assert counts==read(OUT/'COUNTERS.json'),(counts,read(OUT/'COUNTERS.json'))
    assert all(counts[k]<=p['caps'][k] for k in counts)
    for path in OUT.rglob('*.pt'):
        assert existing(path)
        receipt=read(path.with_suffix('.json'))
        for f,h in receipt.get('code_pins',{}).items():assert sha(ROOT/f)==h,f
    spatial=read(OUT/'SPATIAL4_ALL_SOURCE_RESULTS.json')
    practical=read(OUT/'PRACTICAL_RESULTS.json')
    dev=read(OUT/'PRACTICAL_DEVELOPMENT_RESULTS.json')
    selection=read(OUT/'PRACTICAL_SELECTION.json')
    assert sha(OUT/'PRACTICAL_DEVELOPMENT_RESULTS.json')==selection['dev_results_sha256']
    assert not selection['eval_used']
    cells=0;aggregation_errors=[]
    for key,s in spatial['summary'].items():
        c,role=key.split('/')
        rr=[r for r in spatial['rows'] if r['cohort']==c and
            (r['key'] in p['core_keys'][c] if role=='F35_core' else r['role']==role)]
        n,e=aggregate_check(rr,s);cells+=n;aggregation_errors.append(e)
    for obj in (practical,dev):
        for key,s in obj['summary'].items():
            c,*stage=key.split('/')
            rr=[r for r in obj['rows'] if r['cohort']==c and (not stage or r['stage']==stage[0])]
            n,e=aggregate_check(rr,s);cells+=n;aggregation_errors.append(e)
    old=read(F35/'ALL_SOURCE_RESULTS.json')['summary']['spatial'];parent_error=[]
    for c in p['rows']:
        for new,prior in [('Frozen','A0'),('S0','A3')]:
            for metric in ('vIoU_corrected','sIoU','tIoU'):
                err=abs(spatial['summary'][c+'/F35_core']['arms'][new][metric]['mean']-
                        old[c+'/core']['arms'][prior][metric]['mean'])
                assert err<1e-12;parent_error.append(err)
    duplicates=[]
    for r in p['H_rows']:
        ep=dest('practical_eval',r['key'])
        if not ep.exists():continue
        a,b=load(ep),load(dest('practical_guard',r['key']))
        for name in a['fits']:
            x,y=a['fits'][name],b['fits'][name]
            assert x['best_step']==y['best_step']
            assert torch.equal(x['final']['boxes'],y['final']['boxes'])
            assert all(torch.equal(v,y['state'][k]) for k,v in x['state'].items())
        duplicates.append(r['key'])
    command=['bash','scripts/with_local_cuda.sh','.conda/tubedetr/bin/python','-B','-m','pytest',
             'tests/test_spatial4_attribute_v1.py','tests/test_simplification_partial_v1.py','-q']
    tick=time.time();test=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    write(OUT/'TEST_RECEIPT.json',dict(command=command,exit_code=test.returncode,seconds=time.time()-tick,
          stdout=test.stdout,stderr=test.stderr,test_pins={f:sha(ROOT/f) for f in command[-3:-1]}))
    assert test.returncode==0 and '21 passed' in test.stdout
    output=dict(status='pass',counts=counts,budget=p['caps'],new_fit_stages=dict(fit_counts),
        unique_historical_queries=113,unique_historical_sources=113,untouched=False,
        space_extension=dict(hcstvg1_test=33,vidstg_test=32),practical_dev=16,practical_eval=32,guards=6,
        repeated_guard_sources_not_additive=duplicates,raw_pt_receipts=len(list(OUT.rglob('*.pt'))),
        protected_pins=len(p['protected_pins']),protected_registries_unchanged=True,
        exact_cached_expert_observations=observations,reused_conditions=dict(reused),
        fit_loss_trajectories=len(errors),maximum_loss_error=max(errors),
        actual_final_full_model_replays=replay,full_model_replay_max_error=0,
        actual_space_noops=4,source_and_optimizer_episode_reset_verified=True,
        temporal_endpoints_and_logits_unchanged_all_spatial_fits=True,
        native_ASA_calls_exact=checks,attribute_derivatives=diagnostic,attribute_temporal_fits=0,
        maximum_attribute_time_bias_logit_change=max(time_bias),maximum_attribute_time_head_gradient=max(grad),
        visual_probability_changes=dict(minimum=min(visual),maximum=max(visual)),
        numerical_failed_fits=sum(bool(z['failure']) for z in fits),
        zero_state_new_fits=sum(z['state_delta']==0 for z in fits),
        no_anchor_new_fits=sum(bool(z['skipped']) for z in fits),
        independent_bootstrap_cells=cells,maximum_aggregation_error=max(aggregation_errors),
        F35_core_parent_maximum_error=max(parent_error),tests='21 passed; receipt retained',
        online_GT=False,development_GT='practical LR selection only; fixed grid, 8 sources per direction',
        limitations=['Final actual model replay is checked for each new fit, not every rejected proposal.',
            'No original ordered motion class map; no T1-T5 execution, no temporal semantic efficacy claim.',
            'Floating numerical ASA derivatives are not interpreted as meaningful temporal gradient.',
            'All pools historically exposed; multiple comparisons are descriptive.'])
    write(OUT/'AUDIT.json',output);print(output,flush=True)


if __name__=='__main__':main()
