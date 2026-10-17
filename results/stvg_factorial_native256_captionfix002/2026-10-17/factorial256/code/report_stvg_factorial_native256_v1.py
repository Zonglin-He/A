"""Independently checked, anonymous Native-clock factorial publication."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_factorial_native256_v1 import BASE,CORE,read,sha,put,verify
from scripts.export_stvg_lite_batch001 import public_source_text
from scripts.publish_stvg_soft_ep_v1 import verify_privacy

def main():
    from scripts.audit_stvg_factorial_readout_v1 import main as check_output
    from scripts.stvg_overnight_score_v1 import aggregate
    verify();check_output(True);d=read(BASE/'POSTSEAL_RESULTS.json');assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
    summary,paired=aggregate(d['rows'],256);assert summary==d['summary'] and paired==d['paired']
    ix={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']};rr=read(BASE/'ROSTER_LOCK.json')['records'];aliases={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)}
    formulas=lambda a:{'Temporal_independent':{'Temporal':1,'Frozen':-1},'Spatial_Native_independent':{'S_'+a+'_Native':1,'Frozen':-1},'Spatial_given_T':{'Full_'+a+'_Adapted':1,'Temporal':-1},'factorial_interaction':{'Full_'+a+'_Adapted':1,'Temporal':-1,'S_'+a+'_Native':-1,'Frozen':1},'Adapted_minus_Native_evidence_fixed_T':{'Full_'+a+'_Adapted':1,'T_S_'+a+'_Native_evidence':-1}}
    import numpy as np
    from scipy.stats import t
    for c in d['contrasts']:
        form=formulas(c['spatial_policy'])[c['contrast']];ids=[r['probe_alias'] for r in rr if r['dataset']==c['dataset']];v={cid:100*math.fsum(w*ix[cid,a][c['metric']] for a,w in form.items()) for cid in ids if all(ix[cid,a][c['metric']] is not None for a in form)}
        assert set(v)==set(c['per_source_delta_pp']) and all(abs(x-c['per_source_delta_pp'][cid])<1e-11 for cid,x in v.items())
        assert len(v)==c['count'] and c['unknown']==128-len(v)
        if v:
            mean=math.fsum(v.values())/len(v);assert abs(c['mean_delta_pp']-mean)<1e-11
            if len(v)>1:
                err=float(t.ppf(.975,len(v)-1)*np.std(list(v.values()),ddof=1)/math.sqrt(len(v)));assert np.allclose(c['paired_source_t95_interval_pp'],[mean-err,mean+err],atol=1e-10,rtol=0)
        assert (c['positive'],c['negative'],c['neutral'])==(sum(x>1e-6 for x in v.values()),sum(x< -1e-6 for x in v.values()),sum(abs(x)<=1e-6 for x in v.values()))
    def anon(x):
        if isinstance(x,list):return [anon(z) for z in x]
        if isinstance(x,dict):return {aliases.get(k,k):anon(v) for k,v in x.items() if k not in ('GT_bindings','code_pins')}
        return aliases.get(x,x) if isinstance(x,str) else x
    out=ROOT/'results/stvg_factorial_native256_v1/2026-10-17/factorial256';out.mkdir(parents=True,exist_ok=False);paths=set()
    def save(name,value):put(out/name,anon(value))
    save('RESULTS.json',d);save('STUDY_LOCK.json',read(BASE/'STUDY_LOCK.json'));save('ROOT_FACTORIZED_OUTPUT_READBACK.json',read(BASE/'ROOT_FACTORIZED_OUTPUT_READBACK.json'));save('ROOT_NATIVE_CLOCK_SOURCE_READBACK.json',read(BASE/'ROOT_NATIVE_CLOCK_SOURCE_READBACK.json'))
    usage=[]
    for arm in ('A','B'):
        b=BASE/'arms'/arm;refs=read(b/'trials/G32/REFERENCE_LOCK.json')['records'];reasons=collections.Counter(x['reason'] for r in refs for x in r['decisions']);teacher=read(b/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');qual=read(b/'NATIVE_QUALIFICATION.json');student=[r for r in d['usage'] if r['arm']==arm]
        usage.append(dict(arm=arm,actual_Student_fits=sum(r['actual_fit'] for r in student),no_fit=sum(not r['actual_fit'] for r in student),accepted_steps=sum(r['accepted_steps'] for r in student),proposed_steps=sum(r['proposal_count'] for r in student),reference_admission_reasons=dict(reasons),Teacher={k:v for k,v in teacher.items() if isinstance(v,(int,float,bool)) or k=='status'},qualification={k:v for k,v in qual.items() if isinstance(v,(int,float,bool)) or k=='status'},GT_read=False))
    execution=[]
    for p in sorted((BASE/'execution').glob('*/*/WORKER_EXIT.json')):
        r=read(p);assert r['exit_code']==0;execution.append(dict(group=p.parent.parent.name,stage=p.parent.name,wall_seconds=r['wall_seconds'],GT_read=r['GT_read']))
    save('COST_REFERENCE_ACCEPTANCE.json',dict(status='complete',count=256,groups=usage,stage_costs=execution,feature_build=read(BASE/'FEATURE_BARRIER.json'),Source_and_Temporal_and_Adapted_Full_are_existing_sealed_reuses_not_new_model_work=True,old_control_costs_not_relabelled_as_zero=True,shared_setup_not_double_counted=True,unprofiled_substages='unknown, not estimated',GT_read=False))
    # FEATURE_BARRIER includes private paths; expose only counters/cost scalars.
    costs=read(out/'COST_REFERENCE_ACCEPTANCE.json');costs['feature_build']={k:v for k,v in costs['feature_build'].items() if isinstance(v,(int,float,bool)) or k=='status'};(out/'COST_REFERENCE_ACCEPTANCE.json').write_text(json.dumps(costs,indent=2)+'\n')
    lines=['# True Native-clock Spatial-only and Temporal-conditioned evidence factorial','', 'Already exposed development cohort:128 independent parents per domain,256 total. Confirmation128 remains a completed failed B-versus-A confirmation and is not used for new selection. Native A and B rebuild observations, candidates and Teacher references at the Source Native interval and execute real episodic spatial adaptation. Native-candidate Caption is reused without new Temporal-parameter updates. Existing Frozen, Temporal and Adapted-clock Full outputs are reused exactly.','', '| Domain | Arm | vIoU % | tIoU % | dense known-GT sIoU % | R(vIoU>=.3) % | R(vIoU>=.5) % |','|---|---|---:|---:|---:|---:|---:|']
    def val(ds,a,k):return next(x['mean'] for x in summary if x['dataset']==ds and x['arm']==a and x['metric']==k)
    for ds in ('hc2','vidstg'):
        for a in ('Frozen','Temporal','S_A_Native','S_B_Native','T_S_A_Native_evidence','T_S_B_Native_evidence','Full_A_Adapted','Full_B_Adapted'):
            values=[val(ds,a,k) for k in ('m_vIoU','m_tIoU','m_sIoU_dense_GT','R_vIoU_at_0p3','R_vIoU_at_0p5')];lines.append('|'+ds+'|'+a+'|'+ '|'.join('unknown' if x is None else f'{100*x:.6f}' for x in values)+'|')
    lines+=['','| Domain | Policy | Contrast | vIoU delta pp | Paired source 95% t interval pp | + / - / neutral / unknown |','|---|---|---|---:|---|---|']
    for c in d['contrasts']:
        if c['metric']=='m_vIoU':lines.append(f"|{c['dataset']}|{c['spatial_policy']}|{c['contrast']}|{c['mean_delta_pp']:+.6f}|{c['paired_source_t95_interval_pp']}|{c['positive']}/{c['negative']}/{c['neutral']}/{c['unknown']}|")
    lines+=['','The four-cell interaction is (Full-T)-(S_Native-F). It includes changes to observation positions, candidates, references and Tube-IoU support. It is not proof of optimizer-gradient synergy. The fixed-final-T comparison Full_Adapted minus T_S_Native_evidence uses exactly the same final Temporal interval; dense known-GT spatial IoU also uses the same physical GT frames for all arms. All other scored supports remain explicitly named in JSON.','', 'S-only tIoU equals Frozen and T+S tIoU equals Temporal per source. The complete spatial trajectory is recomputed by the qualified frozen-prefix readout through both native decoder passes. Source analysis establishes that TimeDecoder and action hidden states are independent of the changed PosDecoder parameters and query residual. Existing T head adaptation is an independent sealed branch output. Composing its interval is the original factorized deployment protocol, not a claim of a new jointly trained network forward. Old Adapted-conditioned Spatial readouts are retained with that label and are not renamed true Native-only.','', 'Every per-source signed effect and unknown, paired source-level t interval, reference-admission outcome, actual fit/proposal/acceptance count and observed stage cost is retained. Intervals describe uncertainty for this development cohort; no bootstrap, new selection gate, GT-driven screening, or automatic promotion is performed. Spatial parameter and acceptance mathematics, Source checkpoints, random sampling, visual budget, Prompt and precision remain fixed. Native versus Adapted observation changes are the intended treatment. Semantic512 remains a separate model-assisted pixel review; metric changes do not establish identity/role correctness.','', 'Private media, annotations, GT coordinates, query/Caption/Teacher response text, raw tensors, checkpoints, credentials and conversation records are excluded. No official fullset, CURRENT change or project FINAL.']
    doc=ROOT/'docs/STVG_NATIVE_FACTORIAL256_001.md';doc.write_text('\n'.join(lines)+'\n');paths.add(str(doc.relative_to(ROOT)))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    names=['Spatial_Native_independent','Spatial_given_T','factorial_interaction','Adapted_minus_Native_evidence_fixed_T']
    for ax,ds in zip(axs,('hc2','vidstg')):
        for j,a in enumerate(('A','B')):
            cs=[next(c for c in d['contrasts'] if c['dataset']==ds and c['spatial_policy']==a and c['contrast']==n and c['metric']=='m_vIoU') for n in names];means=[c['mean_delta_pp'] for c in cs];errs=[(c['paired_source_t95_interval_pp'][1]-c['paired_source_t95_interval_pp'][0])/2 for c in cs];ax.bar([i+(-.18 if j==0 else .18) for i in range(4)],means,.36,yerr=errs,label=a,capsize=2)
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(4),xticklabels=['S Native - F','Full - T','Interaction','Evidence clock'],title=ds,ylabel='vIoU effect (percentage points)');ax.tick_params(axis='x',labelsize=8);ax.legend()
    fig.tight_layout();fig.savefig(out/'factorial_effects.png',dpi=160);plt.close(fig)
    for name in ('stvg_factorial_native256_v1.py','continue_stvg_factorial_native256_v1.py','score_stvg_factorial_native256_v1.py','audit_stvg_factorial_readout_v1.py','report_stvg_factorial_native256_v1.py','publish_stvg_factorial_native256_v1.py'):
        p=out/'code'/name;p.parent.mkdir(exist_ok=True);p.write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    save('ROOT_SCALAR_PRIVACY_READBACK.json',dict(status='pass',count=256,summary_cells=len(summary),paired_cells=len(paired),factorial_cells=len(d['contrasts']),source_labels_anonymous=True,complete_signed_outcomes_and_unknowns=True,zero_new_GT_reader_or_model_calls=True,actual_figure_view_separate=True,GT_read=False))
    paths|={str(p.relative_to(ROOT)) for p in out.rglob('*') if p.is_file()};verify_privacy(paths)
    for rel in paths:
        if (ROOT/rel).suffix=='.json':
            raw=(ROOT/rel).read_text()
            for cid in aliases:assert ('"'+cid+'"') not in raw,rel
    put(BASE/'publication/factorial256/EXPORT_MANIFEST.json',dict(status='complete',count=256,paths=sorted(paths),files=[dict(path=p,sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in sorted(paths)]));print('NATIVE_FACTORIAL256_EXPORTED_PENDING_ACTUAL_PIXELS_REMOTE')
if __name__=='__main__':main()
