"""Independent CPU readback of every cached state's geometry and aggregations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.analyze_stvg_query_e_path64_v1 import D,P,OLD,ARMS,verify


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import box_iou,official
    torch.set_num_threads(4);p=verify();r=read(P/'RESULTS.json');assert sha(P/'RESULTS.json')==read(D/'CPU_COMPLETION.json')['result_sha256']
    by={x['alias']:x for x in r['rows']};c=collections.Counter();maximum=0.;cases=set(p['fixed_cases'])
    targets={ds:truths(ds,dict(parents=[v['query_ordinal']for v in p['records']if v['dataset']==ds and v['alias']in cases]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    for rec in p['records']:
        saved=load(OLD/'predictions'/(rec['alias']+'.pt'));res=by[rec['alias']];teacher=saved['E_training_teacher_float32'].double().numpy();target=np.c_[teacher[:,:2]-teacher[:,2:]/2,teacher[:,:2]+teacher[:,2:]/2]
        for arm,f in saved['fits'].items():
            rr=res['fits'][arm];assert len(rr['states'])==len(f['path'])and len(rr['transitions'])==len(f['rounds'])
            for i,(state,path)in enumerate(zip(rr['states'],f['path'])):
                b=path['boxes'].double().numpy();xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2];io=box_iou(xy,target)
                err=abs(float(io.mean())-state['E_IoU_all']);assert err<1e-12;maximum=max(maximum,err);c['E_all_states']+=1
                assert abs(float(io[f['feedback_positions']].mean())-state['E_IoU_feedback'])<1e-12;c['E_feedback_states']+=1
                assert state['step']==i and state['t']==rr['states'][0]['t'];c['time_supports']+=1
                if i:
                    t=rr['transitions'][i-1];prev=rr['states'][i-1];assert abs(t['E_all_change_pp']-100*(state['E_IoU_all']-prev['E_IoU_all']))<1e-12
                    assert abs(t['v_change_pp']-100*(state['v']-prev['v']))<1e-12;assert t['updated']==f['rounds'][i-1]['updated'];c['step_differences']+=1
                if rec['alias']in cases:
                    ds=rec['dataset'];q=rec['query_ordinal'];met=official(b,plans[ds][q],targets[ds][0][q],targets[ds][1][q],saved['interval'],ds)
                    for ep in ['v','t','s']:assert abs(met[ep]-state[ep])<2e-10;c['fixed_case_metrics']+=1
            last,first=rr['states'][-1],rr['states'][0];diag=rr['diagnosis']
            assert abs(diag['E_change_pp']-100*(last['E_IoU_all']-first['E_IoU_all']))<1e-12
            assert abs(diag['v_change_pp']-100*(last['v']-first['v']))<1e-12;c['case_final_differences']+=1
    for ds,groups in r['summary'].items():
        for arm,g in groups.items():
            for scope,x in g.items():
                rows=[v for v in r['rows']if v['dataset']==ds and(scope=='all'or v['active'])];fs=[v['fits'][arm]for v in rows];diags=[f['diagnosis']for f in fs]
                assert len(rows)==x['queries']and sum(x['final_E_v_sign_counts'].values())==len(rows)
                for a in ['up','down','flat']:
                    for b in ['up','down','flat']:assert sum(d['E_sign']==a and d['v_sign']==b for d in diags)==x['final_E_v_sign_counts'][a+'_'+b];c['sign_cells']+=1
                for key,field in [('mean_final_E_change_pp','E_change_pp'),('mean_final_v_change_pp','v_change_pp'),('mean_final_unobserved_change_pp','unobserved_change_pp')]:
                    vv=[d[field]for d in diags if d[field]is not None];assert abs(sum(vv)/len(vv)-x[key])<1e-12;c['aggregates']+=1
                tr=[t for f in fs for t in f['transitions']if t['updated']];assert len(tr)==x['actual_updates'];c['actual_update_denominators']+=1
    assert r['independent_counts']['states']==c['E_all_states']==6656 and c['step_differences']==6400
    verify();assert not torch.cuda.is_initialized()
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=64,checks=dict(c),max_overlap_difference=maximum,
        all_original_prediction_and_implementation_SHA_unchanged=True,no_model_or_fit_calls=True,no_GT_step_selection=True))
    put(D/'ROOT_READBACK_COMPLETION.json',dict(status='pass',count=64,actual_charts_public_archive_pending=True))


if __name__=='__main__':main()
