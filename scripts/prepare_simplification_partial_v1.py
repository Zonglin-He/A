"""F36 evidence freeze and pre-outcome protocol. Never edits a parent."""
import collections
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read, write, sha

OUT = ROOT/'artifacts/decota_simplification_partial_v1'
PARENT = ROOT/'artifacts/decota_spatial10_components_v1'


def evidence():
    completion = read(PARENT/'COMPLETION.json')
    checked = 0
    for f, h in {**completion['files'], **completion['code_pins']}.items():
        assert sha(ROOT/f) == h, f
        checked += 1
    data = read(PARENT/'ALL_SOURCE_RESULTS.json')
    effects = read(PARENT/'MODULE_EFFECTS.json')
    contrasts = [('core','A3','A0','vIoU_corrected'), ('core','A3','A2','vIoU_corrected'),
                 ('core','A3','A2','U_tube_sIoU'), ('panel','C0','E_query','vIoU_corrected'),
                 ('panel','B2','B1','vIoU_corrected'), ('panel','C0','D_gamma0','vIoU_corrected')]
    verified = []
    for c in ('hcstvg1_test','vidstg_test'):
        for role,a,b,m in contrasts:
            rr = [r for r in data['spatial_rows'] if r['key'].startswith(c+':') and role in r['roles']
                  and a in r['arms'] and b in r['arms']]
            groups = collections.defaultdict(list)
            for r in rr:
                x,y=r['arms'][a].get(m),r['arms'][b].get(m)
                if x is not None and y is not None: groups[r['group']].append(x-y)
            x=np.array([sum(groups[g])/len(groups[g]) for g in sorted(groups)])
            rng=np.random.default_rng(20260914)
            boot=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1)
            actual=dict(mean=float(np.mean(x)),ci95=np.quantile(boot,[.025,.975]).tolist(),sources=len(x))
            expected=data['summary']['spatial'][c+'/'+role]['contrasts'][a+' - '+b][m]
            assert abs(actual['mean']-expected['mean']) < 1e-12
            assert np.max(np.abs(np.array(actual['ci95'])-expected['ci95'])) < 1e-12
            assert actual['sources']==expected['sources']
            verified.append(dict(direction=c,role=role,a=a,b=b,metric=m,**actual,
                    keys=[r['key'] for r in rr],source_groups=sorted(groups)))
    write(OUT/'F35_EVIDENCE_AUDIT.json',dict(parent_completion_sha256=sha(PARENT/'COMPLETION.json'),
            raw_path=str(PARENT/'ALL_SOURCE_RESULTS.json'),raw_sha256=sha(PARENT/'ALL_SOURCE_RESULTS.json'),
            protected_files_checked=checked,independent_source_aggregation=True,independent_bootstrap=True,
            bootstrap_seed=20260914,bootstrap_n=10000,contrasts=verified))
    text='# F35 confirmed claims — evidence freeze, not promotion\n\n'
    text+='Scope: TA-STVG, two existing source checkpoints, frozen Grounding DINO-T; historical exposure. '
    text+='The complete spatial pipeline is supported relative to matched FP32 Frozen; parameter-specific benefit relative to interpolation is direction-dependent.\n\n'
    text+='|Cohort / role|Contrast|Metric|Delta pp|95% paired source CI pp|Effective sources|\n|---|---|---|---:|---|---:|\n'
    for r in verified:
        text+=f"|{r['direction']}/{r['role']}|{r['a']} − {r['b']}|{r['metric']}|{100*r['mean']:.3f}|[{100*r['ci95'][0]:.3f}, {100*r['ci95'][1]:.3f}]|{r['sources']}|\n"
    text+='\nA3=Spatial10, A0=matched Frozen, A2=same-anchor absolute interpolation; C0=Spatial10 panel; E_query=query only; B2=weak-view pair, B1=exact duplicate pair. '
    text+='U_tube uses matched actual observation exclusions and valid GT frames. Its smaller effective-source denominator is not the main table denominator.\n\n'
    text+='Current 1/3/5/10 prefix evidence supports increasing mean utility in this range, not per-case monotonicity or a global optimum. '
    text+='Grouped state acceptance reproduces private on the audited separable interface; it does not establish temporal-supervision validity. '
    text+='No claim of spatial-only necessity of the query residual, no two-direction superiority to interpolation, no untouched confirmation, no SOTA claim.\n\n'
    text+=f'Primary rows and every source set: [independent audit]({OUT}/F35_EVIDENCE_AUDIT.json). '
    text+=f'Parent: [F35 final report]({PARENT}/FINAL_REPORT.md), [raw rows]({PARENT}/ALL_SOURCE_RESULTS.json), [module effects]({PARENT}/MODULE_EFFECTS.json).\n'
    (OUT/'CONFIRMED_CLAIMS.md').write_text(text)
    (OUT/'UNCONFIRMED_COMPONENTS.md').write_text('''# F35 unconfirmed components and correction boundaries

- Weak second view and two-view averaging: no independent utility benefit established; same-support teacher IoU and student utility differ.
- Reliability/geometric weights: not stably superior to equal-total/shuffled weights; F36 removal sets accepted weights to one and explicitly changes total supervision.
- Robust loss: directional estimates, not universal necessity. Gamma: negligible only at tested LR/10 steps.
- Query residual's unique necessity is unresolved, but the 1792-parameter interface remains locked; no fresh LN-only search.
- All three old full-posterior warp teachers lack support for adoption under F35 view/objective/interface/budget; no universal temporal TTA impossibility claim.
- Old eight-timepoint arm used denominator four. It cannot substitute F36 A5 with planned K=8.
- Native teacher direct decode being native does not imply a warped-view fit leaves original predictions unchanged.
- HC440 has two scorable high-quality anchors, not verified correctness at all accepted anchors. H must compare the same GT-valid support.
- Historical exposure survives re-splitting; multiple queries from one source are not independent videos.
- Zero-update, fallback-only or interpolation-only systems do not complete the genuine parameter-TTA research objective.

Sources: F35 ALL_SOURCE_RESULTS.json, MODULE_EFFECTS.json, SPATIAL_CASE_REVIEW.md,
TEMPORAL_CASE_REVIEW.md and COMPLETION.json. Hash-linked numeric verification is in F35_EVIDENCE_AUDIT.json.
''')


def main():
    assert not (OUT/'LOCK.json').exists(), 'already locked'
    OUT.mkdir(parents=True,exist_ok=True)
    evidence()
    p=read(PARENT/'LOCK.json')
    rows={c:[copy.deepcopy(r) for r in rr if 'temporal' in r['roles']] for c,rr in p['rows'].items()}
    for rr in rows.values():
        for r in rr: r['f36_role']='development' if 'development' in r['roles'] else 'evaluation'
        assert len(rr)==24 and sum(r['f36_role']=='development' for r in rr)==8
        dev=[r for r in rr if r['f36_role']=='development'];ev=[r for r in rr if r['f36_role']=='evaluation']
        for key in ('group','source'):
            assert not {r[key] for r in dev}&{r[key] for r in ev}
        assert len({r['input']['video_sha256'] for r in rr})==24
    # Two protocol-prescribed cases plus fixed historically reviewed positive/negative pairs.
    hkeys=['hcstvg1_test:000440','vidstg_test:009293', 'hcstvg1_test:000055','hcstvg1_test:000707',
           'vidstg_test:008571','vidstg_test:007868']
    allrows={r['key']:r for rr in p['rows'].values() for r in rr}
    H=[copy.deepcopy(allrows[k]) for k in hkeys]
    protected={**p['protected_pins'], **p['own_pins'],
        str(PARENT.relative_to(ROOT)/'COMPLETION.json'):sha(PARENT/'COMPLETION.json')}
    # Parent amendments are real protected dependencies, not stale initial pins.
    for a in sorted(PARENT.glob('CODE_AMENDMENT_*.json')):
        protected.update(read(a)['new_pins'])
    for f,h in protected.items(): assert sha(ROOT/f)==h, f
    lock=dict(name='DeCoTA_F36_simplification_partial_context',created_unix=time.time(),
        attachments={a:sha(Path('./private_authorization_notes')/a/'pasted-text.txt') for a in
            ['private-authorization-c65216043277','private-authorization-f78e5fa4e9ad']},
        parent_lock_sha256=sha(PARENT/'LOCK.json'),protected_pins=protected,rows=rows,H_rows=H,
        H_selection='Protocol HC440/Vid9293 plus pre-fixed historical good/failure cases HC55/707, Vid8571/7868; not selected from F36 results.',
        caps=dict(fits=650,backward=10000,new_DINO=1024,crop_teacher=192),
        space=dict(lr={'hcstvg1_test':.01,'vidstg_test':.1},steps=10,parameters=1792,
            order=['A0','A1','A2','A3','A4','A5'],candidates=['A4','A5'],planned={'A4':4,'A5':8},
            selection='Per direction: maximum dev source-macro v; within 0.1 pp choose lower planned DINO cost, then better worst10 delta, then A4. Read evaluation only after lock.',
            tie_pp=.1,noninferiority_pp=.5,
            repair='At most ONE integrated candidate globally, only if both A4/A5 dev drop >0.5pp versus A0 in at least one direction. Earliest ladder step whose equal-direction mean delta is below -0.5pp is restored to chosen A4/A5; if no such step, no repair. Evaluate dev once, retain only if mean improves and neither direction worsens >0.5pp. Never use eval to construct repair.'),
        temporal=dict(parameters=66306,steps=5,epsilon=.05,beta=[.1,1.,10.],lr=[1e-4,1e-3,1e-2],
            selection='Per direction maximize dev fixed-Spatial10 source-macro v; ties <=0.1pp lower lr then larger beta. Matched controls use selected beta/lr.',
            planned_crops=2,delta='Median differences within each actual crop offset, concatenated across offsets; report merged-grid median too.',
            crop_call_unit='One complete native two-offset/refinement prediction. Log offset executions separately (2 each).',
            arms=['partial','point','nativeclip','wrong','output'],GT_online=False,student_context='full_original'),
        H=dict(max_sources=6,steps=[10,20],new_bbox_trigger='Matched GT-support 20-step mean anchor IoU <0.95 (nonempty scorable support); not triggered by unlabelled anchors.',
            bbox_limit='Copy final bbox linear only on final layer of second spatial pass; actual shape audit required; diagnostic only.'),
        uncertainty=dict(bootstrap_n=10000,seed=20260914,neutral_pp=.1,historical_exposure=True,
            selection_evaluation_source_disjoint=True,untouched=False),
        no_promotion=True,no_new_models=True,no_corruption=True,no_full_dataset=True,no_automation=True)
    write(OUT/'LOCK.json',lock)
    write(OUT/'LOCKED_IMPLEMENTATION.json',dict(state='read_only_parent_no_promotion',protected_pins=protected,
        registry_values={f:read(ROOT/f) for f in ['methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json']},
        backbone='TA-STVG',expert='frozen Grounding DINO-T',source_checkpoints=read(ROOT/'methods/decota_spatial10_v1/configs.json'),
        scope='query residual plus actual final spatial block norm1/norm3/norm4 affine; 1792',
        input='full original video; original native two offsets; FP32 suffix',time_reference='I_seed = I_out = native I0',
        optimizer='episodic Adam, actual proposals, rollback moments, best real parameter state; max10',
        output='student boxes, no teacher-box replacement/interpolation main output',
        text='complete text context and exact target occurrence tokens; original unresolved fallback',
        unresolved='time policy not solved; all fine-grained heuristics are not independently confirmed innovations'))
    write(OUT/'SOURCE_MANIFEST.json',dict(rows=rows,H_rows=H,historical_exposure=True,overlap_not_additive=True))
    (OUT/'PROTOCOL.md').write_text('''# F36 — spatial subtraction + censored-context temporal supervision

Status: locked, not completed. Execute the two attached finite protocols, not a full-dataset queue.
Parent Spatial10 and both method registries are read-only. No extra models, corruption or scheduling.

48 sources: each direction 8 development and 16 within-round source-isolated evaluation;
all historical exposure. Six named H cases are separate and may overlap, never added to main counts.
Online A/T code is GT-free. H is a separately labelled supervised capacity diagnosis.

Space A0 parent; A1 gamma0; A2 accepted weights1; A3 raw regression loss;
A4 original-view four points; A5 original-view eight points and planned denominator8.
Same-time/same-anchor direct and absolute controls; common actual observed union includes rejected requests.
Dev selection and at most one integrated repair are fixed in LOCK.json, never chosen from eval.

Time: two unique cropped-context frozen predictions; original frame IDs/intervals, no resampling.
Student only uses full original input. Censored compatible sets, epsilon.05 and forward KL to source
posterior; beta/lr 3x3, 5 real steps. Point/nativeclip/wrong and free-simplex same-objective controls.
No empty compatible set may be silently discarded. Full-domain observation is exact zero signal.
Native offset/envelope inference retained, spatial observations and boxes fixed to parent Spatial10.

H: same GT-valid accepted support teacher versus GT, 10/20 steps; unknown frames remain unknown.
Only unresolved fit on correct matched support can trigger final-output bbox-linear copy diagnosis.
The 20-step/new-layer results cannot promote a method or block A/T.

Global caps: 650 new fits, 10000 backward including diagnostics, 1024 new DINO,
192 complete two-offset crop teacher predictions. Cache hits and offset calls counted separately.
Final outputs include complete per-source tables, costs, no-op/reset/reinsertion audits, failure retention,
teacher compatibility/coverage, selection lock, and a single bounded decision. No automatic promotion.
''')
    print('F36 LOCKED', {c:len(rr) for c,rr in rows.items()}, 'H',hkeys,flush=True)


if __name__=='__main__': main()
