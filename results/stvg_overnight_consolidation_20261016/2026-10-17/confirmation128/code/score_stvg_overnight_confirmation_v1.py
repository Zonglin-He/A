"""Independent128 confirmation, all-arm barriers before any label access."""
from pathlib import Path
import gc,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_overnight_common_v1 as c
from scripts.stvg_overnight_confirmation_v1 import BASE,verify
from scripts.stvg_overnight_score_v1 import seal,aggregate
read,sha,put=c.read,c.sha,c.write
def main():
    import torch,numpy as np
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    verify();lock=read(BASE/'ARM_LOCK.json');names=lock['arms'];assert names[:2]==['A','B'];rr=read(BASE/'ROSTER_LOCK.json')['records'];files={a:seal(BASE/'arms'/a) for a in names}
    # Complete Source/Temporal and every Spatial prediction root verified above.
    # No earlier per-arm scoring path is used for confirmation.
    cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];actual=[];usage=[];references={a:{r['case_id']:r for r in read(BASE/'arms'/a/'trials/G32/REFERENCE_LOCK.json')['records']} for a in names};tick=time.perf_counter()
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];pp=BASE/'spatial/E1/proposals'/(cid+'.json');pool=read(pp) if pp.exists() else dict(windows=[]);common=[x['source_grid_position'] for x in pool['windows']];saved=None
        for a in names:
            d=torch.load(ROOT/files[a][cid]['path'],map_location='cpu',weights_only=False)
            if saved is None:
                saved=d
                for k,name in (('F','Frozen'),('T','Temporal')):
                    vals,_=metric._score_student_arm(d['arms'][k],case,targets[cid],common,official,independent);rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True));actual.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True))
            else:
                assert d['temporal_record']==saved['temporal_record']
                for k in ('F','T'):
                    for field in ('boxes','interval'):assert np.array_equal(d['arms'][k][field],saved['arms'][k][field])
            adopted=[x['position'] for x in references[a][cid]['anchors']]
            for name,k in ((a,'TS_M2'),(a+'_Spatial','S_M2')):
                vals,_=metric._score_student_arm(d['arms'][k],case,targets[cid],common,official,independent);rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True));vals,_=metric._score_student_arm(d['arms'][k],case,targets[cid],adopted,official,independent);actual.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=vals,GT_read=True))
            f=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],arm=a,anchors=len(adopted),actual_fit=f is not None,accepted_steps=f['optimizer_steps'] if f else 0,nonzero_final=any(not torch.equal(v.cpu(),f['final_state'][k].cpu()) for k,v in f['initial_state'].items()) if f else False,GT_read=False))
            if d is not saved:del d
        del saved;gc.collect()
    s,p=aggregate(rows,128);s2,p2=aggregate(actual,128);put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=128,per_domain=64,rows=rows,summary=s,paired=p,actual_support_rows=actual,actual_support_summary=s2,actual_support_paired=p2,usage=usage,GT_bindings=bindings,GT_read=True,bootstrap=False,candidate_lock_sha256=sha(c.BASE/'CANDIDATE_LOCK.json'),semantic_status='pending separate prelocked pixel review; no semantic claim from vIoU'))
    put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=128,GT_read=True,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),wall_seconds=time.perf_counter()-tick));put(BASE/'GROUP_PUBLICATION_HANDOFF.json',dict(status='ready_for_independent_root_view_public_archive',count=128,GT_read=True))
    def delta(a,b):return [next(x['mean_delta_pp'] for x in p if x['dataset']==ds and x['comparison']==a+'_minus_'+b and x['metric']=='m_vIoU') for ds in ('hc2','vidstg')]
    def passes(a):
        ds={b:delta(a,b) for b in ('Frozen','Temporal','A')};return all(math.fsum(v)/2>0 and min(v)>=-.20 for v in ds.values()),ds
    passed,ds=passes('B');corepassed=None;coredata=None
    if not passed and 'core_B' in names:corepassed,coredata=passes('core_B')
    outcome='confirmation_supports_frozen_candidate' if passed else 'confirmation_selected_core_B_fallback_not_unbiased_confirmation' if corepassed else 'unresolved_spatial_gain'
    put(BASE/'CONFIRMATION_DECISION.json',dict(status='locked',count=128,outcome=outcome,candidate_deltas_pp=ds,core_B_deltas_pp=coredata,engineering_domain_tolerance_pp=.20,not_statistical_significance=True,no_per_query_gate_or_parameter_tuning=True,fullset_started=False,CURRENT_changed=False,GT_read=True))
if __name__=='__main__':main()
