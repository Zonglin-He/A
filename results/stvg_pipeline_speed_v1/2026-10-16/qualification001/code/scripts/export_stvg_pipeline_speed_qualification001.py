"""Publish bounded engineering checks, no256 performance claim or private cache."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_pipeline_speed_v1 as speed
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.export_stvg_lite_batch001 import public_source_text
OUT=ROOT/'results/stvg_pipeline_speed_v1/2026-10-16/qualification001'
PUB=speed.BASE/'publication/qualification001'
def main():
    q=read(speed.BASE/'QUALIFICATION_COMPLETION.json');assert q['status']=='pass' and q['engineering_adopted'] and q['count']==4 and not q['GT_read']
    for p,h in read(speed.BASE/'EXECUTION_RUNTIME.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert len(q['records'])==4 and all(sum(r['dataset']==ds for r in q['records'])==2 for ds in ('hc2','vidstg'))
    for r in q['records']:
        assert {t['recipe'] for t in r['tests']}=={'lite_fullstep','lite_backtrack'}
        assert all(t['full_gradient_Adam_solver_acceptance_path_exact'] and t['full_prediction_boxes_logits_interval_exact'] and not t['GT_read'] for t in r['tests'])
    aliases={r['case_id']:f'Q{i:02}' for i,r in enumerate(q['records'],1)}
    def anon(x):
        if isinstance(x,dict):return {k:(aliases[v] if k=='case_id' else anon(v)) for k,v in x.items()}
        if isinstance(x,list):return [anon(v) for v in x]
        return x
    OUT.mkdir(parents=True,exist_ok=False);PUB.mkdir(parents=True,exist_ok=False)
    public=dict(status='pass',count=4,per_domain=2,development_only=True,GT_read=False,engineering_adopted=True,records=anon(q['records']),qualification_wall_seconds=q['wall_seconds'],actual_fit_executions=16,matched_recipe_comparisons=8,scientific_method_parameters_unchanged=True,whole256_Worker_speed_not_yet_measured=True,independent_GPU_reexecution_not_claimed=True,selection='first2 eligible-reference parents per domain in the already locked256 roster; no GT/outcome selection')
    write(OUT/'QUALIFICATION.json',public)
    old=read(speed.BASE/'SAVED_BASELINE_PROFILE.json');write(OUT/'OLD16_ENGINEERING_PROFILE.json',old)
    design=read(ROOT/'artifacts/stvg_spatial_controls256_v1/trials/Full_E1/DESIGN_LOCK.json')
    write(OUT/'FIXED_CONFIG.json',dict(configs=design['configs'],response_settings=design['response_settings'],trainable_parameters=1792,episodic=True,recipes=['lite_fullstep','lite_backtrack'],CG_budget_and_acceptance_unchanged=True,precision_unchanged=True,GT_read=False))
    (OUT/'PROTOCOL.md').write_text((ROOT/'protocols/stvg_pipeline_speed_v1.md').read_text())
    from scripts.prepare_stvg_pipeline_speed_v1 import FILES
    names=[*['scripts/'+n for n in FILES], 'scripts/profile_stvg_pipeline_saved_v1.py','scripts/report_stvg_controls256_speed003.py','scripts/report_stvg_controls256_speed004.py','scripts/export_stvg_pipeline_speed_qualification001.py','scripts/run_stvg_spatial_controls256_student_v1.py','scripts/run_stvg_spatial_controls256_native_v1.py','scripts/run_stvg_global32_student_v1.py','scripts/run_stvg_global32_context_v1.py','scripts/run_stvg_instance_student32_v1.py','scripts/run_stvg_soft_ep_coordinate_trial003.py','scripts/stvg_spatial_lite_fit_v1.py','scripts/stvg_soft_ep_response_v1.py','scripts/run_stvg_adjacent_motion_student_v1.py','methods/decota_final_simplified_v1/replay.py','methods/decota_final_simplified_v1/backbone.py','methods/decota_final_simplified_v1/tensors.py','methods/decota_final_simplified_v1/objectives.py']
    bindings=[]
    for n in names:
        p=ROOT/n;dest=OUT/'code'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(public_source_text(p.read_bytes()));bindings.append(dict(source=n,source_sha256=sha(p),public_sha256=sha(dest)))
    write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',records=bindings,runtime_sha256=sha(speed.BASE/'EXECUTION_RUNTIME.json'),qualification_sha256=sha(speed.BASE/'QUALIFICATION_COMPLETION.json'),GT_read=False))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));warm=[r for r in q['records'] if r['source_profile']['model_load_seconds']==0.];assert len(warm)==2
    for ax in axs:ax.set_yscale('log');ax.set_ylabel('Seconds (log scale)');ax.set_xticks([0,1],['Original','Qualified reuse']);ax.grid(axis='y',alpha=.2)
    for r in warm:
        y=[r['source_capture_seconds'],r['prefix_restore_seconds']];axs[0].plot([0,1],y,'o-',label=r['dataset']);axs[0].annotate(f'{y[0]:.2f}',(0,y[0]),xytext=(4,4),textcoords='offset points');axs[0].annotate(f'{y[1]:.2f}',(1,y[1]),xytext=(4,4),textcoords='offset points')
    axs[0].set_title('Source step: two warm-model checks');axs[0].legend()
    for r in q['records']:
        for t in r['tests']:axs[1].plot([0,1],[t['original_full_readout_seconds'],t['prefix_full_readout_seconds']],'o-',alpha=.6)
    axs[1].set_title('Final all-frame readout: eight checks');fig.suptitle('Engineering qualification: four fixed development parents; no GT')
    fig.text(.5,.015,'Source miss/cache creation, model loads and qualification are not free. These are not whole-Worker speed ratios.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(OUT/'component_times.png',dpi=160);plt.close(fig)
    text='''# Exact pipeline reuse qualification001

Four fixed development parents (HC2 two, VidSTG two), two original recipes, eight paired recipe comparisons and sixteen actual fits. This is an engineering equivalence panel, not a256-source task result, semantic identity evaluation or untouched confirmation. No GT was read. Selection used the first eligible-reference parents in the fixed roster, before any outcome comparison.

The original versus cached Source path matched initial predictions and trainable initialization. Every complete gradient/Adam/solver/proposal/acceptance path matched exactly under full-step and finite-backtrack recipes. Final boxes, temporal logits and intervals over all output frames matched the original complete-model RGB forward. Assertions ran on actual GPU values; saved qualification receipts and pinned code are supplied, not a claim of an additional independent GPU reexecution. The four original Source feature caches remain private.

Warm-model Source checks took3.141 and5.931 seconds originally, versus0.738 and1.019 seconds to restore and verify the exact prefix. Other Source checks include first model loads and are itemized separately. Eight original complete-model final readouts took1.60–2.54 seconds, versus0.0449–0.0595 seconds for complete two-pass Decoder readout from the frozen prefix. Shared qualification itself took78.128 seconds. These component checks do not establish a whole-Worker or whole-study speed ratio; cache creation/misses, IO, model loads, Solver and mandatory roots remain.

Only features before trainable spatial LNs are cached. Every final frame still recomputes the original LNs, both Decoder passes and action gates. Old R1 features are losslessly represented by their original BF16 bits; no model precision is reduced. Original inputs/parameters, Solver budget, step budget, reference admission, rejection, rollback, early stopping and Gaussian sampling remain fixed. All original research step snapshots remain; the bounded asynchronous CPU writer passed real snapshot/state/Adam/SHA readback on the first16 running outputs. That partial engineering check is not a complete256 prediction seal or a task score.

The original Full_E1 Teacher finished1024 requests normally and was adopted without interruption or rerun. The replacement coordinator has already resumed Full_E1 Student256; the other five fixed groups and deferred global32 remain automatically scheduled. Each complete scientific group will publish all same-parent Frozen/Temporal/A comparisons separately. No current256 Spatial performance or global-context benefit is claimed here. CURRENT was not promoted.

The historical ABC16 timing file contains48 logical output reads from the same16-parent panel, not48 independent sources. Solver/Decoder timing is inside fit timing; residual Worker time also includes final full forward, hashing/checks and saving, and must not be called pure IO. Private media, GT coordinates, query/caption prompts, raw Teacher replies, parameter tensors, caches, weights and credentials are excluded.
'''
    (OUT/'REPORT.md').write_text(text)
    doc=ROOT/'docs/STVG_PIPELINE_SPEED_QUALIFICATION001.md';doc.write_text(text+'\n![Matched component timings](../results/stvg_pipeline_speed_v1/2026-10-16/qualification001/component_times.png)\n')
    paths={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}|{str(doc.relative_to(ROOT))}
    write(PUB/'EXPORT_MANIFEST.json',dict(status='complete',count=4,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]))
    print('SPEED_QUALIFICATION_EXPORTED',len(paths))
if __name__=='__main__':main()
