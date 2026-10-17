"""Saved-scalar-only matched reference-support and tail audit after prediction/GT seal."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_direct_reliable32_v1 import BASE,OLD,read,sha,put
def main():
    from scripts.export_stvg_lite_batch001 import public_source_text
    from scripts.publish_stvg_soft_ep_v1 import verify_privacy
    assert read(BASE/'CPU_COMPLETION.json')['result_sha256']==sha(BASE/'POSTSEAL_RESULTS.json')
    d=read(BASE/'POSTSEAL_RESULTS.json');old=read(OLD/'POSTSEAL_RESULTS.json');rr=read(BASE/'ROSTER_LOCK.json')['records'];alias={r['probe_alias']:f'R{i:03d}' for i,r in enumerate(rr,1)}
    jobs={j['job_id']:j for j in read(BASE/'INPUT_JOB_LOCK.json')['records']};oq={(q['case_id'],q['slot']):q for q in old['reference_quality']};rows=[];domains=[]
    for q in d['reference_quality']:
        j=jobs[q['job_id']];before=oq[q['case_id'],j['slot']];assert before['GT_known']==q['GT_known']
        o=before['reference']['DIRECT'];rows.append(dict(case_id=alias[q['case_id']],dataset=q['dataset'],slot=j['slot'],GT_known=q['GT_known'],
            old_available=o['available'],new_available=q['reference_available'],old_IoU=o['IoU'],new_IoU=q['conditional_IoU'],
            old_success=None if not q['GT_known'] else bool(o['success_at_0p5']),new_success=q['success_at_0p5']))
    def mean(v):return math.fsum(v)/len(v) if v else None
    index={(r['case_id'],r['arm']):r['metrics'] for r in d['rows']}
    for ds in ['hc2','vidstg']:
        qs=[q for q in rows if q['dataset']==ds];known=[q for q in qs if q['GT_known']];common=[q for q in known if q['old_available'] and q['new_available']]
        tail=[dict(case_id=alias[r['probe_alias']],delta_vIoU_pp=100*(index[r['probe_alias'],'R1_FIXED_OPD']['m_vIoU']-index[r['probe_alias'],'R1_DIRECT']['m_vIoU'])) for r in rr if r['dataset']==ds]
        tail.sort(key=lambda q:q['delta_vIoU_pp']);vs=[q['delta_vIoU_pp'] for q in tail]
        domains.append(dict(dataset=ds,planned=64,GT_known=len(known),GT_unknown=64-len(known),
            old_legal=sum(q['old_available'] for q in qs),new_legal=sum(q['new_available'] for q in qs),
            old_success=sum(bool(q['old_success']) for q in known),new_success=sum(bool(q['new_success']) for q in known),
            new_coverage_positions=sum(not q['old_available'] and q['new_available'] for q in qs),lost_coverage_positions=sum(q['old_available'] and not q['new_available'] for q in qs),
            common_available_GT_known=len(common),common_support_mean_IoU_delta_pp=mean([100*(q['new_IoU']-q['old_IoU']) for q in common]),
            gained_GT_known_success=sum(not q['old_success'] and q['new_success'] for q in known),lost_GT_known_success=sum(q['old_success'] and not q['new_success'] for q in known),
            full_all16_mean_delta_vIoU_pp=mean(vs),posthoc_remove_one_largest_positive_mean_pp=mean(vs[:-1]) if vs else None,
            posthoc_sensitivity_not_source_exclusion_or_selection=True,full_signed_tail=tail,
            old_actual_fit_sources=sum(u['actual_fit'] for u in old['usage'] if u['dataset']==ds and u['arm']=='DIRECT'),
            new_actual_fit_sources=sum(u['actual_fit'] for u in d['usage'] if u['dataset']==ds),
            old_accepted_steps=sum(u['accepted_steps'] for u in old['usage'] if u['dataset']==ds and u['arm']=='DIRECT'),
            old_proposed_steps=sum(u['proposed_steps'] for u in old['usage'] if u['dataset']==ds and u['arm']=='DIRECT'),
            new_accepted_steps=sum(u['accepted_steps'] for u in d['usage'] if u['dataset']==ds),new_proposed_steps=sum(u['proposed_steps'] for u in d['usage'] if u['dataset']==ds)))
    obj=dict(status='pass',count=32,domains=domains,rows=rows,new_GT_reader=False,new_model_calls=0,source='sealed old/new saved scalar results only',
        common_available_is_descriptive_not_primary_success_only_cohort=True,all_extremes_preserved=True)
    put(BASE/'MATCHED_REFERENCE_TAIL_READBACK.json',obj)
    out=ROOT/'results/stvg_r1_direct_reliable32/2026-10-17';put(out/'MATCHED_REFERENCE_TAIL_READBACK.json',obj)
    (out/'code'/Path(__file__).name).write_text(public_source_text(Path(__file__).read_bytes()))
    # Include the actual recovery fits/roots, not just the original controller's failed launch.
    cost=read(out/'COST.json');cost['stages']=[dict(stage=str(p.parent.relative_to(BASE/'execution')),**read(p)) for p in sorted((BASE/'execution').rglob('WORKER_EXIT.json'))]
    cost.update(all_actual_recovery_stage_costs_included=True,initial_validation_failure_Student_fits=0,original128_Teacher_reused_no_repeat=True)
    put(out/'COST.json',cost,replace=True)
    put(out/'ENGINEERING_RECOVERY.json',read(BASE/'recovery/STUDENT_RUNTIME_PIN002/COMPLETION.json'))
    name='continue_stvg_r1_direct_reliable32_002.py';(out/'code'/name).write_text(public_source_text((ROOT/'scripts'/name).read_bytes()))
    p=BASE/'publication/reliable32/EXPORT_MANIFEST.json';m=read(p);paths=set(m['paths'])|{str((out/'MATCHED_REFERENCE_TAIL_READBACK.json').relative_to(ROOT)),str((out/'code'/Path(__file__).name).relative_to(ROOT)),str((out/'code'/name).relative_to(ROOT)),str((out/'ENGINEERING_RECOVERY.json').relative_to(ROOT))};verify_privacy(paths)
    m.update(paths=sorted(paths),files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in sorted(paths)]);put(p,m,replace=True)
    print(domains)
if __name__=='__main__':main()
