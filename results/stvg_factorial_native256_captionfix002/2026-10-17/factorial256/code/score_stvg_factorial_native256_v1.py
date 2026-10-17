"""True branch factorial and fixed-time evidence contrast, all arms sealed first."""
from pathlib import Path
import gc,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_factorial_native256_v1 import BASE,CORE,verify,read,sha,put
from scripts.stvg_overnight_score_v1 import seal,aggregate
def main():
    import torch,numpy as np
    from scipy.stats import t
    from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
    from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
    from scripts.score_stvg_spatial_controls256_v1 import validate_barrier
    verify();files={a:seal(BASE/'arms'/a) for a in ('A','B')};controls={a:validate_barrier(a) for a in ('A','B')};rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,GTbindings=truth._truths_after_seal(cases);official,independent,_=truth._metric_functions();rows=[];usage=[];equiv=[]
    def scored(pred,case,target,common):
        vals,_=metric._score_student_arm(pred,case,target,common,official,independent);vals.update({'R_vIoU_at_0p3':float(vals['m_vIoU']>=.3),'R_vIoU_at_0p5':float(vals['m_vIoU']>=.5)});return vals
    for rec in rr:
        cid=rec['probe_alias'];case=cases[cid];pool=read(CORE/'spatial/E1/proposals'/(cid+'.json'));common=[w['source_grid_position'] for w in pool['windows']];first=None
        for arm in ('A','B'):
            e=files[arm][cid];d=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False);old=torch.load(ROOT/controls[arm][cid]['path'],map_location='cpu',weights_only=False)
            for k in ('F','T'):
                for key in ('boxes','interval'):assert np.array_equal(d['arms'][k][key],old['arms'][k][key])
            assert np.array_equal(d['arms']['S_M2']['boxes'],d['arms']['TS_M2']['boxes']) and d['arms']['S_M2']['interval']==case['Native_interval'] and d['arms']['TS_M2']['interval']==old['arms']['T']['interval']
            preds=[('S_'+arm+'_Native',d['arms']['S_M2']),('T_S_'+arm+'_Native_evidence',d['arms']['TS_M2']),('Full_'+arm+'_Adapted',old['arms']['TS_M2']),('S_'+arm+'_Adapted_conditioned',old['arms']['S_M2'])]
            if first is None:preds=[('Frozen',d['arms']['F']),('Temporal',d['arms']['T'])]+preds
            vals={}
            for name,pred in preds:
                v=scored(pred,case,targets[cid],common);rows.append(dict(case_id=cid,dataset=rec['dataset'],arm=name,metrics=v));vals[name]=v
            if first is None:first=vals
            assert vals['S_'+arm+'_Native']['m_tIoU']==first['Frozen']['m_tIoU']
            for name in ('T_S_'+arm+'_Native_evidence','Full_'+arm+'_Adapted'):assert vals[name]['m_tIoU']==first['Temporal']['m_tIoU']
            assert abs(vals['S_'+arm+'_Native']['m_sIoU_dense_GT']-vals['T_S_'+arm+'_Native_evidence']['m_sIoU_dense_GT'])<2e-12
            f=d['fits']['M2'];usage.append(dict(case_id=cid,dataset=rec['dataset'],arm=arm,actual_fit=f is not None,accepted_steps=f['optimizer_steps'] if f else 0,proposal_count=sum(len(x['candidates']) for x in f['path']) if f else 0,GT_read=False));equiv.append(dict(case_id=cid,arm=arm,spatial_Temporal_logits_Source_exact=True,Native_Adapted_same_final_spatial_trajectory=True,S_tIoU_equals_F=True,Full_tIoU_equals_T=True,same_all_known_GT_geometry_support=True));del d,old;gc.collect()
    summary,paired=aggregate(rows,256);idx={(r['case_id'],r['arm']):r['metrics'] for r in rows};contrasts=[]
    for ds in ('hc2','vidstg'):
        ids=[r['probe_alias'] for r in rr if r['dataset']==ds]
        for arm in ('A','B'):
            formulas={'Temporal_independent':{'Temporal':1,'Frozen':-1},'Spatial_Native_independent':{'S_'+arm+'_Native':1,'Frozen':-1},'Spatial_given_T':{'Full_'+arm+'_Adapted':1,'Temporal':-1},'factorial_interaction':{'Full_'+arm+'_Adapted':1,'Temporal':-1,'S_'+arm+'_Native':-1,'Frozen':1},'Adapted_minus_Native_evidence_fixed_T':{'Full_'+arm+'_Adapted':1,'T_S_'+arm+'_Native_evidence':-1}}
            for name,formula in formulas.items():
                for key in idx[ids[0],'Frozen']:
                    v={cid:100*sum(weight*idx[cid,a][key] for a,weight in formula.items()) for cid in ids if all(idx[cid,a][key] is not None for a in formula)};n=len(v);mean=math.fsum(v.values())/n if n else None;err=float(t.ppf(.975,n-1)*np.std(list(v.values()),ddof=1)/math.sqrt(n)) if n>1 else None
                    contrasts.append(dict(dataset=ds,spatial_policy=arm,contrast=name,metric=key,count=n,unknown=128-n,mean_delta_pp=mean,paired_source_t95_interval_pp=[mean-err,mean+err] if err is not None else None,per_source_delta_pp=v,positive=sum(x>1e-6 for x in v.values()),negative=sum(x< -1e-6 for x in v.values()),neutral=sum(abs(x)<=1e-6 for x in v.values())))
    put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=256,per_domain=128,rows=rows,summary=summary,paired=paired,contrasts=contrasts,usage=usage,invariants=equiv,GT_bindings=GTbindings,GT_read=True,bootstrap=False,interaction_includes_clock_observation_reference_and_metric_coupling=True,not_gradient_synergy_proof=True,geometry_support='same all known GT physical frames; fixed final Temporal comparison isolates spatial trajectories from interval changes'))
    put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=256,GT_read=True,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json')));print('TRUE_NATIVE_FACTORIAL256_ALL_ARMS_POSTSEAL_COMPLETE')
if __name__=='__main__':main()
