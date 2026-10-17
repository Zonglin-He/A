"""Root scalar export for caption decomposition and fixed noCoverage extension."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_coverage224_v1 import BASE,CORE,EXP,read,sha,put,old
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy
def main(group):
    out=ROOT/'results/stvg_caption_coverage256_v1/2026-10-17'/group;out.mkdir(parents=True,exist_ok=False);doc=ROOT/'docs'/('STVG_CAPTION_COVERAGE_'+group.upper()+'_001.md');paths=set()
    def save(name,d):put(out/name,d)
    if group=='decomposition':
        d=read(BASE/'CAPTION_DECOMPOSITION.json');assert read(BASE/'CAPTION_DECOMPOSITION_COMPLETION.json')['result_sha256']==sha(BASE/'CAPTION_DECOMPOSITION.json');save('RESULTS.json',d)
        # Independently recover contributions from anonymous per-source rows.
        for co in d['records']:
            for s in co['stratification']:
                rr=[r for r in co['records'] if r['dataset']==s['dataset']];control=s['comparison'].split('_minus_')[1];v=[100*(r['arms']['B']['metrics']['m_vIoU']-(r['arms']['A']['metrics'] if control=='A' else r['Temporal'])['m_vIoU']) for r in rr];assert abs(math.fsum(v)/len(rr)-s['overall_delta_pp'])<1e-12
                for sub in s['groups']:
                    vv=[val for r,val in zip(rr,v) if r['caption_available_before_adaptation']==sub['caption_available']];assert len(vv)==sub['count'] and abs(math.fsum(vv)/len(rr)-sub['signed_contribution_to_overall_pp'])<1e-12
        # Bind each pre-Teacher caption flag to the original policy root/barrier.
        root=read(CORE/'POLICY_GLOBAL_ROOT.json');assert root['status']=='pass';ix={(r['case_id'],r['slot']):r['caption_available'] for r in root['records']}
        for j in read(CORE/'TEACHER_JOB_LOCK.json')['records']:assert read(CORE/'native_cache'/j['job_id']/'POLICIES.json')['caption_available']==ix[j['case_id'],j['center_slot']]
        for r in read(old.BASE/'confirmation128/FEATURE_BARRIER.json')['files']:assert sha(ROOT/r['path'])==r['sha256']
        save('ROOT_PRESTATE_AND_ARITHMETIC.json',dict(status='pass',count=384,core_window_states_bound=1024,confirmation_policy_hashes_bound=512,pre_Teacher_input_state_not_update_success=True,all_weighted_contributions_recomputed=True,no_new_GT_reader_or_model=True,GT_read=False))
        lines=['# Caption availability decomposition: core256 and confirmation128','', 'Saved sealed predictions only, no model calls or new GT-reader calls. Core256 and confirmation128 are reported separately. The primary split is the Caption status captured before Spatial Teacher selection; successful update status is descriptive only. All sources and denominators remain. Differences are vIoU percentage points.','', '| Cohort | Domain | Contrast | Subset | N | Subset mean pp | Signed overall contribution pp |','|---|---|---|---|---:|---:|---:|']
        for co in d['records']:
            for r in co['stratification']:
                for sub in r['groups']:lines.append(f"|{co['cohort']}|{r['dataset']}|{r['comparison']}|{'available C' if sub['caption_available'] else 'missing M'}|{sub['count']}|{sub['mean_delta_pp']:+.6f}|{sub['signed_contribution_to_overall_pp']:+.6f}|")
        lines+=['','Confirmation HC: both Caption-present and Caption-missing contributions are negative, so lack of opportunities is not the sole explanation. Confirmation Vid: present-Caption B minus A is +0.091881pp (overall contribution +0.068911), but missing-Caption contribution -0.323335pp reverses the total. This is descriptive attribution of saved outcomes, not causal identification of an optimal selection policy.','', 'B confirmation no-fit20 =19 missing-Caption parents (3 HC/16 Vid,76 missing-window receipts) plus1 Vid source with no legal/parsed non-Source reference. The other14 fitted but finally unchanged cases are different:3 HC no-information/no-proposal and11 Vid all proposed steps rejected. They must not be added to the no-fit20 or equated with76 windows.','', '| Cohort | Domain | Caption | Arm | N | Stage outcomes |','|---|---|---|---|---:|---|']
        for co in d['records']:
            for r in co['stage_counts']:lines.append(f"|{co['cohort']}|{r['dataset']}|{r['caption_available']}|{r['arm']}|{r['count']}|{json.dumps(r['stage_outcomes'],sort_keys=True)}|")
        lines+=['','The JSON retains every window reference admission reason, nonfinite/reference-rule proposal rejection, no-information skipped step and per-source final state. A selected Source candidate is still forbidden by the original contract; this rule is retained. Missing Caption must not be called a model parse error merely because the generic anchor converter emits that reason.','', 'Whole-query missing-Caption fallback-A is an offline scalar diagnostic selecting one complete independent query prediction. It does not splice boxes, states or windows and is not a new independent confirmation. Diagnostic strategy minus A: core HC+0.169818/Vid+0.429831pp; confirmation HC-0.117477/Vid+0.068911pp. It remains separate from noCoverage.','', 'Next model work is only the original noCoverage deletion on the complementary224 core development parents. Primary contrast is against matched A; secondary against B. Original32, additional224 and full256 are separate; all old confirmation conclusions remain. No Global32/C/BT/Prompt/LR/budget search, CURRENT change or official fullset. Semantic512 arm-blind review is separate and only actual viewed pixels may support labels.']
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,co in zip(axs,d['records']):
            r=[x for x in co['stratification'] if x['comparison']=='B_minus_A'];x=[0,1];v=[z['groups'][0]['signed_contribution_to_overall_pp'] for z in r];m=[z['groups'][1]['signed_contribution_to_overall_pp'] for z in r];ax.bar([i-.18 for i in x],v,.36,label='Caption present');ax.bar([i+.18 for i in x],m,.36,label='Caption missing');ax.axhline(0,color='black',lw=.7);ax.set(xticks=x,xticklabels=['HC','Vid'],title=co['cohort'],ylabel='Contribution to full-cohort B minus A (pp)');ax.legend()
        fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig);count=384
    elif group=='coverage256':
        d=read(BASE/'MATCHED_RESULTS256.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'MATCHED_RESULTS256.json');aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(read(CORE/'ROSTER_LOCK.json')['records'],1)}
        def anon(x):
            if isinstance(x,list):return [anon(z) for z in x]
            if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items()}
            return aliases.get(x,x) if isinstance(x,str) else x
        save('RESULTS.json',anon(d));save('ACTUAL_CONFIG.json',read(EXP/'SCOPE_CONFIG.json'));save('STUDY_LOCK.json',read(BASE/'STUDY_LOCK.json'))
        lines=['# Original noCoverage deletion: complementary224 and complete256','', 'Single fixed deletion of the local candidate-priority coverage loop. Original four-frame global64/local96, Prompt, clock, geometry, frame/role quotas, ID/context protection, Student parameters, acceptance/rejection and full readout remain. Caption fallback unchanged. Existing32 is reused exactly;224 are new development executions, not independent confirmation.','', '| Partition | Domain | Contrast | Mean delta pp | Positive / negative / neutral / unknown |','|---|---|---|---:|---|']
        from scripts.stvg_overnight_score_v1 import aggregate
        for part in d['records']:
            ss,pp=aggregate(part['rows'],part['count']);assert ss==part['summary'] and pp==part['paired']
            for r in pp:
                if r['metric']=='m_vIoU' and r['comparison'] in ('B_noCoverage_minus_A','B_noCoverage_minus_B','B_noCoverage_minus_Temporal','B_noCoverage_minus_Frozen'):lines.append(f"|{part['partition']}|{r['dataset']}|{r['comparison']}|{r['mean_delta_pp']:+.6f}|{r['positive']}/{r['negative']}/{r['neutral']}/{r['unknown']}|")
        lines+=['','Primary comparison is noCoverage minus A; secondary is minus B. Improvement over B alone is simplification evidence, not proof of added value over A. No automatic new method/module or reused confirmation128. All negative tails and unknowns remain. Actual-support spatial readouts must not be called fresh Native-clock S-only experiments. Semantic review remains a separate model-assisted pixel audit, not inferred from Student metrics.']
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(11,4))
        for ax,ds in zip(axs,('hc2','vidstg')):
            for k,contrast in enumerate(('B_noCoverage_minus_A','B_noCoverage_minus_B')):
                vals=[next(x['mean_delta_pp'] for x in part['paired'] if x['dataset']==ds and x['comparison']==contrast and x['metric']=='m_vIoU') for part in d['records']];ax.bar([x+(-.18 if k==0 else .18) for x in range(3)],vals,.36,label=contrast)
            ax.set(xticks=range(3),xticklabels=['initial32','new224','all256'],title=ds,ylabel='Student vIoU difference (pp)');ax.axhline(0,color='black',lw=.7);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'paired_effects.png',dpi=150);plt.close(fig);count=256
    else:raise ValueError(group)
    lines+=['','Public exports exclude private media/annotations/GT coordinates, weights/raw tensors/optimizer states, actual query-caption text/Teacher raw responses, credentials and personal conversation. Whole-project FINAL is not claimed.'];doc.write_text('\n'.join(lines)+'\n');paths.add(str(doc.relative_to(ROOT)))
    names=['analyze_stvg_caption_coverage256_v1.py','run_stvg_coverage224_v1.py','continue_stvg_coverage224_v1.py','score_stvg_coverage256_v1.py','report_stvg_caption_coverage_v1.py']
    for name in names:
        t=out/'code'/name;t.parent.mkdir(exist_ok=True);t.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    save('ROOT_PRIVACY_SCALAR_READBACK.json',dict(status='pass',count=count,complete_signed_subgroups=True,actual_plot_pixels_checked_separately=True,GT_read=False))
    paths|={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
    pub=BASE/'publication'/group;put(pub/'EXPORT_MANIFEST.json',dict(status='complete',count=count,paths=sorted(paths),files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in sorted(paths)]));print(json.dumps(dict(status='exported_pending_actual_view_and_remote',group=group,files=len(paths))))
if __name__=='__main__':main(sys.argv[1])
