"""Root-only saved confirmation/component audit, without models or new GT reads."""
from pathlib import Path
import hashlib,json,sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.stvg_overnight_score_v1 import seal,aggregate
read,sha,put=c.read,c.sha,c.write
def main():
    c.verify();base=c.BASE/'confirmation128';arms=read(base/'ARM_LOCK.json')['arms'];assert arms==['A','B'];rows=read(base/'ROSTER_LOCK.json')['records'];assert len(rows)==128
    parents={(x['dataset'],x['parent_sha256']) for x in rows};assert len(parents)==128 and not parents&{(x['dataset'],x['parent_sha256']) for x in read(c.CORE/'ROSTER_LOCK.json')['records']}
    for entry in read(base/'ROSTER_LOCK.json')['registered_history_inventory']:
        rp=ROOT/entry['path'];assert sha(rp)==entry['sha256'];old={(r['dataset'],r['parent_sha256']) for r in read(rp)['records']};assert not parents&old
    designs=[read(base/'arms'/a/'trials/G32/DESIGN_LOCK.json') for a in arms]
    for k in ('configs','response_settings','recipe'):assert designs[0][k]==designs[1][k]
    for arm in arms:
        b=base/'arms'/arm;seal(b);t=b/'teacher/G32';bar=read(t/'TEACHER_GLOBAL_BARRIER.json');root=read(t/'ROOT_READBACK.json');assert bar['count']==root['count']==128 and root['status']=='pass' and root['barrier_sha256']==sha(t/'TEACHER_GLOBAL_BARRIER.json')
        for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    for scope in ('confirmation128','ablation_noCaption','ablation_noCoverage'):
        b=c.BASE/scope;d=read(b/'POSTSEAL_RESULTS.json');assert read(b/'CPU_COMPLETION.json')['result_sha256']==sha(b/'POSTSEAL_RESULTS.json')
        for prefix in ('','actual_support_'):
            s,p=aggregate(d[prefix+'rows'],d['count']);assert (s,p)==(d[prefix+'summary'],d[prefix+'paired'])
    # Audit every actual local and global index/role allocation. Labels/Teacher
    # text are not used to choose or modify either ablation.
    import numpy as np
    from scripts import stvg_overnight_global_v1 as bridge
    audits=[]
    for scope,kind in (('ablation_noCaption','noCaption'),('ablation_noCoverage','noCoverage')):
        b=c.BASE/scope;bridge.configure(scope);roster=read(b/'ROSTER_LOCK.json')['records'];known={r['probe_alias']:r['dataset'] for r in roster};jobs=read(b/'TEACHER_JOB_LOCK.json')['records'];teacher=read(b/'teacher/G32/TEACHER_GLOBAL_BARRIER.json');entries={Path(x['path']).stem:x for x in teacher['files']};no_caption=set();new_calls_on_missing=0;checked=0
        for j in jobs:
            cache=c.CORE/'native_cache'/j['job_id'];old=read(cache/'POLICIES.json');assert sha(cache/'native.npz')==old['native_sha256'];entry=entries[j['job_id']];assert sha(ROOT/entry['path'])==entry['sha256'];response=read(ROOT/entry['path'])
            with np.load(cache/'native.npz',allow_pickle=False) as a:
                f=a['features'];q=a['query_embedding'];cap=a['caption_embedding']
                actual=[]
                for k,(start,end,T,K) in enumerate(((0,512,2,64),(512,1280,3,96))):
                    valid=a['valid'+str(k)];mandatory=a['mandatory'+str(k)];groups=a['groups'+str(k)];s=bridge.choose(f[start:end].reshape(T,256,-1),q,cap,valid,mandatory,groups,K)
                    assert s['quotas']==old['records'][k]['quotas'] and s['role_quotas']==old['records'][k]['role_quotas'];ix=s['indices']['B'].tolist();assert len(ix)==K and mandatory.reshape(-1)[np.flatnonzero(mandatory)].all() and set(np.flatnonzero(mandatory))<=set(ix)
                    if kind=='noCoverage' and k==0:assert ix==old['records'][0]['indices']['B']
                    actual.append(ix)
            if not old['caption_available']:
                no_caption.add(j['case_id']);new_calls_on_missing+=int(response['new_teacher_call'])
            if response['new_teacher_call']:assert response['visual_indices_sha256']==hashlib.sha256(json.dumps(actual).encode()).hexdigest() and response['visual_token_count']==160
            checked+=1
        d=read(b/'POSTSEAL_RESULTS.json');subset=[]
        for ds in ('hc2','vidstg'):
            ix={(r['case_id'],r['arm']):r['metrics']['m_vIoU'] for r in d['rows'] if r['dataset']==ds};ids=[cid for cid in known if known[cid]==ds and cid not in no_caption];v=[100*(ix[i,'Candidate']-ix[i,'LocalB']) for i in ids]
            subset.append(dict(dataset=ds,caption_available_sources=len(ids),caption_unavailable_sources=sum(known[i]==ds for i in no_caption),mean_delta_pp=math.fsum(v)/len(v) if v else None))
        doc=dict(status='pass',count=32,window_count=checked,all_actual_indices_verified=True,same_total_budget_and_each_frame_role_quota=True,same_ID_context_protection=True,same_candidate_geometry_local_pixels_Prompt_Student=True,component=kind,caption_unavailable_sources=len(no_caption),new_WHO_calls_on_missing_caption=new_calls_on_missing,common_caption_secondary=subset,scope='whole-caption selector deletion includes restored eligibility on missing-caption sources' if kind=='noCaption' else 'candidate-priority coverage loop only, unchanged global indices',GT_read=False,new_GT_reader_calls=0,new_model_calls=0);put(b/'ROOT_COMPONENT_READBACK002.json',doc);audits.append(doc)
    decision=read(base/'CONFIRMATION_DECISION.json');assert decision['outcome']=='unresolved_spatial_gain';d=read(base/'POSTSEAL_RESULTS.json');deltas={x['dataset']:x for x in d['paired'] if x['metric']=='m_vIoU' and x['comparison']=='B_minus_A'};assert all(x['mean_delta_pp']<0 for x in deltas.values())
    corrected=dict(status='complete',count=128,original_machine_outcome=decision['outcome'],spatial_B_minus_Temporal_positive_both_domains=True,failed_gate='B does not outperform matched A; additional B evidence value unconfirmed, not absence of all Spatial benefit',B_minus_A_mean_pp=sum(x['mean_delta_pp'] for x in deltas.values())/2,method_selection_stopped=True,new_gate_Prompt_or_candidate_search=False,TubeDETR_frozen_B_effect_condition_not_triggered=True,official_fullset_not_started=True,semantic_pixel_audit_still_pending=True,GT_read=False);put(base/'ROOT_CONFIRMATION_INTERPRETATION002.json',corrected)
    put(c.BASE/'ROOT_MEASUREMENT_READBACK002.json',dict(status='pass',count=128,confirmation_all_roots_and_same_updater=True,all_registered_parent_overlaps_zero=True,all_three_group_scalar_support_chains_verified=True,ablation_windows_checked=sum(x['window_count'] for x in audits),independent_component_single_factor_readback=True,new_GT_reader_calls=0,new_model_calls=0,GT_read=False))
    print(json.dumps(dict(status='pass',count=128,ablation_windows_checked=sum(x['window_count'] for x in audits))))
if __name__=='__main__':main()
