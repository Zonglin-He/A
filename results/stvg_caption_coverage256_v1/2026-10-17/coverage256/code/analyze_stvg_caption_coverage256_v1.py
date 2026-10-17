"""Saved-only pre-adaptation caption stratification; no models/new GT."""
from pathlib import Path
import collections,gc,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
BASE=ROOT/'artifacts/stvg_caption_coverage256_v1'
read,sha=c.read,c.sha
def put(p,d):c.write(p,d)
def main():
    import torch
    c.verify();out=[];allrows=[]
    for label,core,n in [('core256',c.CORE,256),('confirmation128',c.BASE/'confirmation128',128)]:
        roster=read(core/'ROSTER_LOCK.json')['records'];jobs=read(core/'TEACHER_JOB_LOCK.json')['records'];by=collections.defaultdict(list)
        for j in jobs:by[j['case_id']].append(j)
        if n==256:
            results={a:read(core/'trials'/a/'POSTSEAL_RESULTS002.json') for a in ('A','B')};dirs={a:core/'trials'/a for a in results};metrics={a:{(r['case_id'],r['arm']):r['metrics'] for r in d['rows']} for a,d in results.items()};uses={a:{r['case_id']:r for r in d['usage']} for a,d in results.items()}
        else:
            d=read(core/'POSTSEAL_RESULTS.json');results={'A':d,'B':d};dirs={a:core/'arms'/a/'trials/G32' for a in results};metrics={a:{(r['case_id'],r['arm']):r['metrics'] for r in d['rows']} for a in results};uses={a:{r['case_id']:r for r in d['usage'] if r['arm']==a} for a in results}
        refs={a:{r['case_id']:r for r in read(dirs[a]/'REFERENCE_LOCK.json')['records']} for a in results};files={a:{r['case_id']:r for r in read(dirs[a]/'GLOBAL_PREDICTION_BARRIER.json')['files']} for a in results};rows=[]
        for i,rec in enumerate(roster,1):
            cid=rec['probe_alias'];jj=by[cid];caps=[]
            for j in jj:
                p=core/'native_cache'/j['job_id']/'POLICIES.json';pol=read(p);assert pol['job_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest();caps.append(pol['caption_available'])
            assert len(caps)==4 and len(set(caps))==1
            cap=caps[0];row=dict(case_id=f'{label}_R{i:03d}',dataset=rec['dataset'],cohort=label,caption_available_before_adaptation=cap,arms={})
            for a in results:
                u=uses[a][cid];ref=refs[a][cid];steps=u.get('steps');f=None
                if steps is None and u['actual_fit']:
                    e=files[a][cid];p=ROOT/e['path'];assert sha(p)==e['sha256'];saved=torch.load(p,map_location='cpu',weights_only=False);f=saved['fits']['M2'];assert f is not None
                    steps=[dict(accepted=r['accepted'],skipped=r['skipped'],reason=r.get('reason'),rollback=r['rollback'],candidates=[{k:z[k] for k in ('finite','reference_pass','accepted')} for z in r['candidates']]) for r in f['path']];assert sum(r['accepted'] for r in steps)==u['accepted_steps'];del saved,f;gc.collect()
                steps=steps or [];decisions=ref['decisions'];counts=collections.Counter(x['reason'] for x in decisions);teacher=[]
                for j in jj:
                    p=(core/'teacher'/a/(j['job_id']+'.json')) if n==256 else (core/'arms'/a/'teacher/G32'/(j['job_id']+'.json'));t=read(p);decision=next(x for x in decisions if x['slot']==j['center_slot']);assert decision['teacher_sha256']==sha(p);teacher.append(t)
                calls=sum(t['new_teacher_call'] for t in teacher);missing=sum(t['parsed']['status']=='unavailable_no_caption' for t in teacher)
                if not u['actual_fit']:
                    assert not ref['anchors']
                    reason='caption_unavailable_no_WHO_no_reference' if missing==4 else 'reference_not_admitted' if any(x in counts for x in ('source_candidate_forbidden','presence_below_0_6','candidate_geometry_or_presence_invalid','candidate_center_box_outside_normalized_bounds')) else 'no_legal_or_parsed_nonSource_reference'
                elif not u['accepted_steps']:
                    proposals=[z for r in steps for z in r['candidates']]
                    reason='attempted_all_proposals_rejected' if proposals else 'attempted_no_proposal_no_information_or_weight'
                else:reason='accepted_nonzero_final' if u['nonzero_final'] else 'accepted_no_final_parameter_change'
                row['arms'][a]=dict(actual_fit=u['actual_fit'],accepted_steps=u['accepted_steps'],nonzero_final=u['nonzero_final'],anchor_count=u['anchors'],stage_outcome=reason,window_reference_decisions=dict(counts),WHO_actual_calls=calls,caption_missing_window_receipts=missing,proposal_count=sum(len(r['candidates']) for r in steps),nonfinite_proposals=sum(not z['finite'] for r in steps for z in r['candidates']),reference_rule_rejections=sum(z['finite'] and not z['reference_pass'] for r in steps for z in r['candidates']),skipped_step_reasons=dict(collections.Counter(r['reason'] for r in steps if r['skipped'])),metrics=metrics[a][cid,a])
            for k in ('Frozen','Temporal'):assert metrics['A'][cid,k]==metrics['B'][cid,k]
            row['Temporal']=metrics['B'][cid,'Temporal'];row['Frozen']=metrics['B'][cid,'Frozen'];rows.append(row)
        strat=[];diagnostic=[]
        for ds in ('hc2','vidstg'):
            rr=[r for r in rows if r['dataset']==ds];N=len(rr);assert N==n//2
            for control in ('A','Temporal'):
                vv={r['case_id']:100*(r['arms']['B']['metrics']['m_vIoU']-(r['arms']['A']['metrics'] if control=='A' else r['Temporal'])['m_vIoU']) for r in rr};groups=[]
                for cap in (True,False):
                    ss=[r for r in rr if r['caption_available_before_adaptation']==cap];v=[vv[r['case_id']] for r in ss];mean=math.fsum(v)/len(v) if v else None;contribution=math.fsum(v)/N;groups.append(dict(caption_available=cap,count=len(ss),valid=len(v),unknown=0,mean_delta_pp=mean,signed_contribution_to_overall_pp=contribution,positive=sum(x>1e-6 for x in v),negative=sum(x< -1e-6 for x in v),neutral=sum(abs(x)<=1e-6 for x in v)))
                overall=math.fsum(vv.values())/N;assert abs(overall-math.fsum(x['signed_contribution_to_overall_pp'] for x in groups))<1e-12
                strat.append(dict(dataset=ds,comparison='B_minus_'+control,count=N,overall_delta_pp=overall,groups=groups,decomposition_error=overall-math.fsum(x['signed_contribution_to_overall_pp'] for x in groups)))
            # Prospective query-level fallback is an offline diagnostic, not new
            # inference. It never splices local-window boxes or updated states.
            for name,pick in [('offline_missing_caption_fallback_A',lambda r:'B' if r['caption_available_before_adaptation'] else 'A')]:
                vals=[100*(r['arms'][pick(r)]['metrics']['m_vIoU']-r['arms']['A']['metrics']['m_vIoU']) for r in rr];diagnostic.append(dict(dataset=ds,strategy=name,count=N,minus_A_pp=math.fsum(vals)/N,not_independent_confirmation=True,no_new_model_calls=True,whole_query_only=True))
        stages=[]
        for ds in ('hc2','vidstg'):
            for cap in (True,False):
                ss=[r for r in rows if r['dataset']==ds and r['caption_available_before_adaptation']==cap]
                for a in ('A','B'):stages.append(dict(dataset=ds,caption_available=cap,arm=a,count=len(ss),actual_fits=sum(r['arms'][a]['actual_fit'] for r in ss),WHO_actual_calls=sum(r['arms'][a]['WHO_actual_calls'] for r in ss),missing_caption_windows=sum(r['arms'][a]['caption_missing_window_receipts'] for r in ss),stage_outcomes=dict(collections.Counter(r['arms'][a]['stage_outcome'] for r in ss)),window_decisions=dict(sum((collections.Counter(r['arms'][a]['window_reference_decisions']) for r in ss),collections.Counter())),proposal_count=sum(r['arms'][a]['proposal_count'] for r in ss),reference_rule_rejections=sum(r['arms'][a]['reference_rule_rejections'] for r in ss),nonfinite_proposals=sum(r['arms'][a]['nonfinite_proposals'] for r in ss)))
        result=dict(cohort=label,count=n,stratification=strat,stage_counts=stages,offline_fallback_diagnostic=diagnostic,records=rows);out.append(result);allrows+=rows
    put(BASE/'CAPTION_DECOMPOSITION.json',dict(status='complete',count=384,cohorts_reported_separately=True,pre_adaptation_caption_status=True,new_model_calls=0,new_GT_reader_calls=0,GT_read=False,records=out))
    put(BASE/'CAPTION_DECOMPOSITION_COMPLETION.json',dict(status='complete',count=384,GT_read=False,CPU_only=True,result_sha256=sha(BASE/'CAPTION_DECOMPOSITION.json')))
    print(json.dumps([dict(cohort=x['cohort'],stratification=x['stratification'],offline_fallback=x['offline_fallback_diagnostic']) for x in out]))
if __name__=='__main__':main()
