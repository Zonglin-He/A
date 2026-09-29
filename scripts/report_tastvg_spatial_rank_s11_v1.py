"""Render fixed S1.1 results without configuration selection."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
from scripts.audit_tastvg_spatial_rank_s11_public_v1 import run as audit
BASE=ROOT/'artifacts/tastvg_spatial_rank_s11_v1';RAW=ROOT/'artifacts/tastvg_spatial_online_opd_s1_v1'

def run():
    s=read(BASE/'SUMMARY.json');d=read(BASE/'DIAGNOSTICS.json');res=read(BASE/'RESOURCES.json');src=read(BASE/'SOURCE_EFFECTS.json');pub=audit(BASE,RAW);write(BASE/'PUBLIC_AUDIT.json',pub)
    def stat(group,subset,key):
        v=s[group][subset]['metrics'][key];return f"{100*v['mean']:+.6f} [{100*v['ci95'][0]:+.6f}, {100*v['ci95'][1]:+.6f}]"
    lines=['# S1.1: Raw-RKL / Rank-RKL / Rank-RKL + Normalized Step','',
    'Completed fixed development experiment. Raw S1 sealed results reused; two new arms each execute six independent16-arrival streams,18 effective updates. No grid or posthoc configuration selection.','',
    '## Primary future nonexpert results','',
    'Paired delta vs Frozen (and budgeted rerank, identical to Frozen at nonexpert arrivals), in percentage points. Source means over five corruption types, then bootstrap10000 seed20260929 over12sources;60corruption cells are not60independent sources. Fixed stream/order conditional intervals.','',
    '| Arm | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] | Clean ΔsIoU [95% CI] | Clean ΔvIoU [95% CI] |','|---|---:|---:|---:|---:|']
    for a,name in [('raw','Raw-RKL'),('rank','Rank-RKL'),('norm','Rank-RKL+Norm')]:lines.append('| '+name+' | '+' | '.join(stat(g,'nonexpert',a+'_minus_frozen_'+m) for g,m in [('corruption','s'),('corruption','v'),('clean','s'),('clean','v')])+' |')
    lines+=['','| Paired change | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] |','|---|---:|---:|']
    for key in ['rank_minus_raw','norm_minus_raw','norm_minus_rank']:lines.append(f'| {key} | '+stat('corruption','nonexpert',key+'_s')+' | '+stat('corruption','nonexpert',key+'_v')+' |')
    lines+=['','## Actuation diagnostics','','Medians over18effective updates per arm. Fixed source probe radius1.5765801926049214; Norm target step.015765801926049215.','', '| Arm | q range | p range | actual step norm | step/probe | KL decreases |','|---|---:|---:|---:|---:|---:|']
    for a,v in d.items():lines.append(f"| {a} | {v['q_range']['median']:.9f} | {v['p_range']['median']:.9f} | {v['step_norm']['median']:.9g} | {v['step_to_probe_ratio']['median']:.9g} | {v['KL_decreased']}/18 |")
    lines+=['','Full KL before/after, gradient ranges, all six final state norms and step ranges are in DIAGNOSTICS.json; every update in per-arm ROWS.json. Rank uses same SGD.005; Norm one fixed global-norm update. Empty expert at arrival0 in every stream means6noops/arm. Nonempty writes at4/8/12. Neither arm uses a GT gate, clipping, rollback, replay, momentum or extra steps. All equal rewards produce uniform teacher, which does not guarantee zero compatibility-RKL gradient if student p is nonuniform.','',
    '## Nonexpert arrivals after the first effective update','',
    'Additional diagnostic9source subset,45corruption cells (arrival>4); the main12source endpoint above retains the initial3nonexpert no-update controls.','',
    '| Arm | Corruption ΔsIoU [95% CI] | Corruption ΔvIoU [95% CI] |','|---|---:|---:|']
    for a in d:lines.append('| '+a+' | '+stat('corruption','nonexpert_after_first_write',a+'_minus_frozen_s')+' | '+stat('corruption','nonexpert_after_first_write',a+'_minus_frozen_v')+' |')
    lines+=['','## Positive and adverse sources','','Source-mean nonexpert corruption Δs/Δv in pp. All sources, including early exact no-op, are retained in SOURCE_EFFECTS.json.','', '| Arm | source | ΔsIoU | ΔvIoU |','|---|---|---:|---:|']
    for a in ['rank','norm']:
        seq=sorted([r for r in src if r['group']=='corruption'],key=lambda r:r[a+'_minus_frozen_s'])
        for r in seq[:2]+seq[-2:]:lines.append(f"| {a} | Q{r['parent']+1:02} | {100*r[a+'_minus_frozen_s']:+.6f} | {100*r[a+'_minus_frozen_v']:+.6f} |")
    lines+=['','## Protocol and verification','',
    'Same TA-STVG Vid-source checkpoint5ab12c86, original16previously exposed VidSTG sources/onequery each, existing clean/5%random-burst frame drop/freeze/motion blur/occlusion/exposure pixels (GT/query-independent). Same Sa2VA-4B revision3fee777d five uniform-frame masks. H and expert caches reused, new expert calls0.1792parameters: second-pass spatial query256 plus final spatial block5 norm1/3/4 weight+bias1536. Four fixed orthogonal antithetic directions plus center, same absolute radius. Current-policy nine candidates regenerated per scheduled expert per arm; same downstream native routing and geometry as S1. Candidate targets/teacher rewards detached. Rank descending0-based average ties1e-12. q=softmax(-rank), student p=softmax(-native L1/GIoU distance), reverse KL(p||q).',
    '', 'Source reset per arm/condition, state inherited across arrivals, report pre-update prediction. Entire192new predictions sealed before old16GT label scoring. Legal GT frame sIoU and correctedvIoU independently dual-scored384times.192state transitions/12resets,36gradient-step/KL reconstructions,64512parameter-coordinate update checks,8learned-state full native reinsertions.48current centers match central predictions; nonexpert frames do not consume expert/candidate data. Exact numerical details in per-arm AUDIT/CURRENT_POLICY_AUDIT files. FP32 KL reconstructed in float64 tolerance1e-6; intended parameter-step reconstruction tolerance1.5e-7.3CPU contract tests plus public scalar reconstruction.',
    '',f"New GPU process total {res['total_GPU_process_seconds']:.6f}s, including loading and validation;432current-policy candidate tubes,36backward steps,0new encoder captures,0new expert calls. Raw reused, no new raw GPU. Public scalar checks: {pub['scalar_checks']}.",
    '', '## Interpretation boundary','',
    'Ranking sharpens teacher q and fixed normalization enforces the requested actuation scale; task utility is assessed separately. Finite-support box compatibility is an output-space surrogate, not a native parameter-policy likelihood. Different inherited parameters produce different later on-policy supports, so this is a comparison of fixed online recipes, not all later gradients on common fixed data. This repeatedly exposed16source development cohort, single stream order and few writes do not establish population equivalence, unseen generalization or universal failure of specialist/OPD methods. No production registration change, temporal OPD rescue or joint evaluation. Parameter-space OPD is only the user-specified possible next mechanism, not executed here.','']
    (BASE/'REPORT.md').write_text('\n'.join(lines))
    (BASE/'REPRODUCE.md').write_text('''# Reproduction

From repository root with the original authorized private caches/model assets:

```bash
.conda/tubedetr/bin/python -B -m pytest -q tests/test_tastvg_spatial_rank_s11_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py prepare rank
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py prepare norm
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py run rank
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py run norm
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_rank_s11_v1.py score rank
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_rank_s11_v1.py score norm
.conda/tubedetr/bin/python -B scripts/analyze_tastvg_spatial_rank_s11_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_spatial_rank_s11_v1.py
```

Artifacts are write-once; preserve completed/failed runs and use a new destination for independent reproduction. Both new predictions must seal before scoring. Raw S1 remains read-only. Public-only scalar verification needs Python/NumPy, no model, media or labels:

```bash
python scripts/audit_tastvg_spatial_rank_s11_public_v1.py results/tastvg_spatial_rank_s11/2026-09-29
```

Public export contains scalar results and code, not private tensors, masks, query text/IDs, GT coordinates or weights. Full inference reproduction requires authorized local inputs; scalar reproduction does not.
''')
    print(BASE/'REPORT.md')

if __name__=='__main__':run()
