"""Actual fixed P1-P6 closure inventory and source-bound complete result index."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import verify,BASE,NS,PUB,read,write,sha
DEST=ROOT/'results/stvg_opd_paper_suite_complete/2026-10-10'


def run():
    runtime=verify();phases=[];bound=0
    for i in range(1,7):
        label='P'+str(i);close=BASE/(label+'_ROOT_CLOSING_RECEIPT.json');c=read(close)
        assert c['status']=='complete' and not c['paper_suite_complete'] and label in c['scope']
        for rel,expected in c.get('evidence',c.get('receipts',{})).items():
            p=ROOT/rel if rel.startswith(('artifacts/','results/','scripts/','docs/','protocols/','methods/')) else BASE/rel
            assert p.is_file() and sha(p)==expected,(label,rel);bound+=1
        barrier=BASE/(label+'_PREDICTION_BARRIER.json');b=read(barrier)
        assert b['status']=='sealed' and not b.get('GT_read',False)
        github=BASE/(label+'_FINAL_GITHUB_RECEIPT.json');g=read(github)
        assert g['status']=='pass' and g['repository']=='Zonglin-He/A'
        if i==1:assert g['per_remote_file_content_verified'] and set(g['refs'].values())=={g['commit']}
        elif i==2:assert g['all_remote_contents_individually_verified'] and g['branches']==['main','research/stvg-opd-paper-hc2-revision-v2']
        else:assert g['actual_remote_bytes_SHA256_Git_blob_tree_verified'] and set(g['remote_refs_after'].values())=={g['commit']}
        assert g['commit']==c['public_commit'] and len(g['files'])==g['file_count']
        assert sum(r['bytes'] for r in g['files'])==g['bytes']
        for f in g['files']:
            if i==1:assert f['remote_bytes_verified'] and f['remote_blob_verified']
            elif i==2:assert f['remote_bytes_SHA256_gitblob_tree_and_local_exact']
            else:assert all(f[k] for k in ['remote_bytes_exact','remote_SHA256_exact','remote_Git_blob_exact','remote_tree_mode_exact'])
        ar=BASE/(label+'_ARCHIVE_VERIFICATION_RECEIPT.json');assert read(ar)['status']=='pass'
        v=BASE/(label+'_ROOT_VISUAL_REVIEW.json');assert read(v)['status']=='pass'
        ps=BASE/(label+'_PUBLIC_SCALAR_AUDIT.json');assert read(ps)['status']=='pass'
        report=ROOT/'docs'/('STVG_OPD_REVISED_'+label+'_ROOT_REVIEW.md');assert report.is_file()
        output_root=next(str(Path(f['path']).parent) for f in g['files'] if f['path'].endswith('/README.md') and f['path'].startswith('results/stvg_opd_p'+str(i)+'_complete/'))
        item=dict(phase=label,scope=c['scope'],phase_actual_complete=True,root_closing_sha256=sha(close),
            original_public_commit=g['commit'],original_public_tree=g['tree'] if i==2 else g['tree_sha'],original_public_files=g['file_count'],original_public_bytes=g['bytes'],
            complete_original_remote_verification_receipt_sha256=sha(github),actual_view_receipt_sha256=sha(v),actual_archive_receipt_sha256=sha(ar),
            public_scalar_audit_sha256=sha(ps),report_sha256=sha(report),result_directory=output_root,
            logical_arrivals=c.get('actual_GPU_arrivals',c.get('logical_arrivals',c.get('formal_queries'))),
            deployment_outputs=c.get('logical_deployment_outputs'),unit='queries with four deployment arms' if i==6 else 'logical fitted arrivals',
            parent_sources=c.get('actual_root_parent_sources',c.get('parent_sources')),
            new_formal_fits=c.get('new_formal_fits'),exact_complete_original_aliases=c.get('exact_original_complete_stream_reuse'),
            all_negative_results_retained=c['all_negative_results_retained'])
        phases.append(item)
    figures=[]
    for name in ['stvg_motivation_cross_domain_v6','stvg_motivation_backbone_names_v11']:
        folder=ROOT/'artifacts'/name;c=read(folder/'ROOT_CLOSING_RECEIPT.json');g=read(folder/'FINAL_GITHUB_RECEIPT.json')
        assert c['status']=='complete' and g['status']=='pass'
        assert g['main_sha']==g['research_sha']==g['commit'] and len(g['files'])==g['file_count']
        assert sum(f['bytes'] for f in g['files'])==g['bytes']
        for f in g['files']:
            assert len(f['sha256'])==64 and len(f['blob_sha'])==40 and f['bytes']>=0
        figures.append(dict(namespace=name,status='complete',scope=c['scope'],root_sha256=sha(folder/'ROOT_CLOSING_RECEIPT.json'),public_receipt_sha256=sha(folder/'FINAL_GITHUB_RECEIPT.json'),public_commit=g['commit']))
    p0=read(BASE/'P0_ROOT_CLOSING_RECEIPT.json');assert p0['status']=='complete' and not p0['primary_original_efficacy_gate_pass']
    process=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
    own=[]
    for line in process.splitlines():
        args=line.strip().split()
        if len(args)>2 and any(Path(arg).name.startswith(('continue_stvg_opd_','run_stvg_opd_p1_','run_stvg_opd_p2_','run_stvg_opd_p3_','run_stvg_opd_p4_','run_stvg_opd_p5_','run_stvg_opd_p6_')) for arg in args[2:] if not any(x in arg for x in ['\n',';','"',"'"])):
            # Match actual Python script argv, not code strings in an inspection shell.
            if any('python' in Path(arg).name for arg in args[1:3]) and not any(arg in ['-c','-'] for arg in args[1:4]):own.append(line.strip())
    assert not own,own
    inventory=dict(status='pass_pending_final_suite_publication_archive_and_FINAL_COMPLETION',scope='actual fixed P1-P6 phase-contract/evidence assembly, not newly repeated GPU/math/inference',
        phases=phases,figures=figures,phase_count=6,immutable_phase_closing_bindings_checked=bound,
        original_P0_negative_gate_preserved=True,P0_root_sha256=sha(BASE/'P0_ROOT_CLOSING_RECEIPT.json'),
        original_current_method_sha256=runtime['pins']['methods/CURRENT_METHOD.json'],actual_current_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        current_method_changed=False,active_original_suite_model_controllers=[],EATA_and_historical_queues_paused=True,
        historical_rosters_and_different_aggregation_units_not_pooled=True,no_result_driven_tuning=True,
        original_whole_stage_GT_barriers_preserved=True,new_GPU_calls=0,new_GT_arrays_read=False,
        independent_full_decoder_Jacobian=False,future_numerical_OOM_safety=False,paper_suite_complete=False,time=time.time())
    DEST.mkdir(parents=True,exist_ok=True);write(DEST/'PHASE_CLOSURE_INVENTORY.json',inventory)
    lines=['# Original fixed Spatial OPD P1–P6: complete experimental evidence','',
        'The original finite suite has completed all six deployment phases, global prediction seals, postseal scores, actual root math/state/dense reviews, '
        'real report/RGB views, anonymous all-result GitHub verification and phase archive closing. '
        'This final companion assembles their preserved evidence; it does not retune or rerun a phase. '
        'Final suite closure is recorded separately in the actual FINAL_COMPLETION after this package is remotely verified and the final archive is checked.','',
        'The method improves over its frozen source in the full cross-domain primary analysis and in every tested same-domain robustness condition. '
        'The mechanism controls also identify limits: reliable superiority over direct L1/GIoU distillation, two-direction superiority over DINO Refine, '
        'uniform necessity of all active blocks/writeback, and monotonic observation-budget returns are not all established. '
        'Those negative controls are part of the results, not removed or used for another parameter sweep.','',
        '| Phase | Original question | Actual size and completion | Supported result and material limit |','|---|---|---|---|',
        '| P1 | Full cross-domain main comparison | 41,355 OPD arrivals; HC2 3,482 queries/237 parents and VidSTG 10,303 queries/732 parents, three orders; full baseline matched reuse | Excluding the 32 development parents per target, OPD−Source parent vIoU HC2 +2.409 [1.891,2.907] pp and Vid +3.329 [3.017,3.646]. OPD−DINO is positive on HC2; Vid interval contains zero. |',
        '| P2 | Direct L1/GIoU vs Shuffled Feedback vs Fixed Rollout vs Full | 7,168 logical arrivals: 5,632 new fits +1,536 exact complete original aliases; 256 historical parents | Full beats shuffled in all four setting means; Full−Direct intervals all include zero. Some refreshed-rollout contrasts are positive. |',
        '| P3 | Query-only vs LN-only vs alpha0 joint vs Full | 7,168 logical arrivals: 5,376 new fits +1,792 exact complete aliases; 256 historical parents | Full−Query is positive in some settings; Full−LN and Full−alpha0 do not establish uniform superiority. Before−Frozen is its own trajectory, not a causal alpha0 contrast. |',
        '| P4 | Same-domain physical-burst robustness | 15,504 new fits, 969 parents ×16 conditions: clean + five burst families ×2.5/5/10% | All16 Full−Frozen intervals per target are positive. Clean HC +3.766 [2.578,4.901], Vid +2.456 [1.666,3.253] pp. HC current-query means are negative in all16 conditions; severe harms remain. |',
        '| P5 | Uniform K1/2/4/8, actual cost and feedback chain | 2,560 logical arrivals: 2,048 new fits +512 exact original K4 aliases; 256 historical parents/two orders; two pre-GT unified appendices separate | Vid all four budgets have positive total intervals; HC all four contain zero. Vid K4−K2 and K8−K4 contain zero; no K selection follows. |',
        '| P6 | Existing temporal projection/head and postseal offline GT-head failure diagnostic | 64 historical parents/queries, 256 deployment outputs/four arms; separate64 CPU supervised head fits | Actionness projection improves vIoU in both directions; head HC +2.956 [.908,5.188], Vid +.945 [−.174,2.296] pp. Vid head−projection −.689 [−1.610,−.020]; offline GT head +15.764 [10.823,21.225], unavailable at deployment. |','',
        'P4–P6 serve different purposes. P4 tests whether the fixed same-domain method tolerates physical disruptions at three severities. '
        'P5 varies only the declared observation budget and measures actual recorded cost; its pre-GT unified appendix changes several parameters and is separate. '
        'P6 distinguishes existing temporal evidence, an existing head update and supervised capacity/failure after every deployable output is sealed. '
        'It preserves original SpatialOPD longer LN histories, so the temporal/spatial subset is not a state-history-matched causal experiment.','',
        'Every parent bootstrap uses 10,000 paired draws within its declared original target/roster and averages orders/conditions within the parent before sampling when specified. '
        'Intervals are conditional on historically exposed cohorts, source checkpoints and saved histories and are not multiplicity-adjusted. '
        'The full-query P1 primary exclusion does not erase earlier exposure. Counts from different phases/arms/aliases are not added into a purported independent sample size. '
        'The in-domain supervised reference and postseal supervised GT-head are separately labeled; EATA has only the old HC supplement, not a complete paired baseline.','',
        'Original main HC .01/.025/.05/40/LN1/16/M32 and Vid .03/.1/.25/10/LN1/8/M32, Uniform4/admitted Top1 single frozen DINO, '
        '1792 joint parameters, antithetic on-policy Gaussian likelihood, detached IoU, query-residual/Adam reset, independent source resets, LN inheritance and final-round output remain fixed. '
        'Native WHEN is unchanged in main SpatialOPD. The existing finite P6 temporal arms are separately named. '
        'Observation-only separately pinned numerical recorders preserve scientific tensors and original receipts. '
        'A numerical audit passing is not an efficacy result or proof of every decoder/CUDA kernel/future numerical safety.','',
        '| Phase | Actual public files / bytes | Immutable result package |','|---|---:|---|']
    for item in phases:
        link='https://github.com/Zonglin-He/A/tree/'+item['original_public_commit']+'/'+item['result_directory']
        lines.append(f"| {item['phase']} | {item['original_public_files']} / {item['original_public_bytes']} | [{item['original_public_commit'][:12]}]({link}) |")
    lines+=['','Read the full phase root reports below for every absolute score, paired contrast, negative tail, strata, recorded costs and cases. '
        'All original public file contents were actually checked byte-for-byte with SHA256/Gitblob/tree at their immutable commits. '
        'This final assembly checks the saved verification receipts and all immutable phase-closing local bindings; it does not claim to have repeated every model or mathematical scan. '
        'The final companion itself must pass a fresh remote byte verification.','',
        'Figure1 strict cross-domain v6 and final v11 have their own complete root/view/public/archive receipts. '
        'The PTD/Qwen3-VL display row retains joint-training provenance in its caption and is excluded from strict source-only cross-domain statistics. '
        'The fixed plot/cases are not regenerated and no active predictions are read for illustration.','',
        'All original failed guards, incomplete failed-fit witnesses, actual serialized failed fits, old helper errors, negative controls and repaired runtime variants remain preserved. '
        'Where failed process memory was not serialized, comparisons concern real reproductions and repeated fits, not dead memory. '
        'No method promotion, fresh-source stability guarantee, global hyperparameter optimum or future OOM claim is made. '
        'Private RGB/video/query/caption/GT geometry/weights/actions/logits/gradient/Adam/fit/cache assets remain excluded. '
        'EATA and every historical paused best_full/userquick/negative/recovery/IoU-energy queue remain paused.','',
        'The required original P1–P6 experimental suite has no remaining GPU phase. Closure of this suite does not mean that a manuscript is accepted, '
        'that every design component has established superiority, or that paused historical experiments are resumed. '
        'The final actual FINAL_COMPLETION, final publication receipt and final archive determine the monitor stopping condition.']
    (DEST/'ACTUAL_SUITE_REVIEW.md').write_text('\n'.join(lines)+'\n');(ROOT/'docs/STVG_OPD_REVISED_SUITE_ROOT_REVIEW.md').write_text('\n'.join(lines)+'\n')
    for item in phases:
        label=item['phase'];p=ROOT/'docs'/('STVG_OPD_REVISED_'+label+'_ROOT_REVIEW.md');target=DEST/'phase_reports'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        names=[label+'_ROOT_CLOSING_RECEIPT.json',label+'_FINAL_GITHUB_RECEIPT.json',label+'_ARCHIVE_VERIFICATION_RECEIPT.json',label+'_ROOT_VISUAL_REVIEW.json',label+'_PUBLIC_SCALAR_AUDIT.json',label+'_PREDICTION_BARRIER.json']
        final=BASE/(label+'_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json')
        if final.exists():names.append(final.name)
        for name in names:
            target=DEST/'receipts'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(BASE/name,target)
    for item in figures:
        folder=ROOT/'artifacts'/item['namespace']
        for name in ['ROOT_CLOSING_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            target=DEST/'receipts'/item['namespace']/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,target)
    for name in ['P0_ROOT_CLOSING_RECEIPT.json','P5_COMPLETE_CLOSING_GITHUB_RECEIPT.json']:
        shutil.copy2(BASE/name,DEST/'receipts'/name)
    write(BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json',dict(inventory,actual_suite_report_sha256=sha(DEST/'ACTUAL_SUITE_REVIEW.md'),phase_inventory_sha256=sha(DEST/'PHASE_CLOSURE_INVENTORY.json'),
        outputs={str(p.relative_to(ROOT)):sha(p) for p in DEST.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='all_six_phase_contracts_pass',phases=6,immutable_bindings=bound,final_publication_archive_FINAL_COMPLETION_pending=True)))


if __name__=='__main__':run()
