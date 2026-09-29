"""All256 predictions sealed, dual geometry; fixed exposed-Dev routing."""
import sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_a03_native import D,N,GAP,ARMS,load,labels_for
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins


def main():
    import numpy as np,torch
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.external_evidence_metrics import tensor_metrics
    torch.set_num_threads(4);start=time.monotonic();out=N/'independent_readback_v1';assert not out.exists()
    try:
        assert read(N/'ROOT_FIELD_READBACK.json')['status']=='passed'
        seal=read(N/'PREDICTIONS_SEAL.json');assert read(N/'COMPLETE.json')['seal_sha']==sha(N/'PREDICTIONS_SEAL.json') and seal['predictions']==256
        check_pins({str(N/p):h for p,h in seal['files'].items()});check_pins(read(N/'LOCK.json')['pins'])
        write(out/'PRE_SCORE_AUDIT.json',dict(status='passed',queries=64,parents=16,predictions=256,
            seal_sha=sha(N/'PREDICTIONS_SEAL.json'),field_readback_sha=sha(N/'ROOT_FIELD_READBACK.json'),
            GT_disclosure='Existing source-GT derived analytic oracle, exposed Dev64. Not label-free or fresh validation.',target_read=False))
        rows=read(N/'INPUTS.json');labels=labels_for(rows,'dev');results={a:[] for a in ARMS};maxerr=0.;reuseerr=0.
        old=read(GAP/'independent_readback_v1/REPORT.json')
        for i,row in enumerate(rows):
            for arm in ARMS:
                pred=load(N/'episodes'/f'{i:04}'/(arm+'.pt'));assert pred['key']==row['key'] and pred['source']==row['source']
                m=score_tube_independently(pred,labels[row['key']]);other=tensor_metrics(pred,labels[row['key']])
                maxerr=max(maxerr,max(abs(m[k]-other[k]) for k in ('tIoU','sIoU','vIoU')));assert maxerr<1e-6
                if arm in ('B1','oracle'):
                    before=old['arms'][arm]['rows'][row['gap_index']];assert before['key']==row['key']
                    reuseerr=max(reuseerr,max(abs(m[k]-before['metrics'][k]) for k in ('tIoU','sIoU','vIoU')));assert reuseerr==0
                results[arm].append(dict(key=row['key'],source=row['source'],metrics=m,interval=pred['interval'],format_ok=pred['format_ok'],invalid_geometry=int((~pred['geometry_valid']).sum())))
        parents={a:summarize_parents(rs) for a,rs in results.items()};metrics=('tIoU','sIoU','vIoU')
        comps={a:{k:paired_parent_bootstrap(parents[a],parents['B1'],k) for k in metrics} for a in ARMS[1:]}
        versus_full={a:{k:paired_parent_bootstrap(parents[a],parents['oracle'],k) for k in metrics} for a in ARMS[2:]}
        tails={};ret={};gates={};gainret={}
        for arm in ARMS[1:]:
            tails[arm]={};ret[arm]={}
            for k in metrics:
                base=np.array([r['metrics'][k] for r in results['B1']]);val=np.array([r['metrics'][k] for r in results[arm]]);diff=(val-base)*100
                tails[arm][k]=dict(query_harm_gt5pp=int((diff< -5).sum()),query_gain_gt5pp=int((diff>5).sum()),worst_pp=float(diff.min()),best_pp=float(diff.max()),positive=int((diff>0).sum()),negative=int((diff<0).sum()),zero=int((diff==0).sum()))
                ret[arm][k]=dict(eligible=int((base>.5).sum()),retained=int(((base>.5)&(val>.5)).sum()))
            collapse=any(comps[arm][k]['mean_delta_pp']<=-1 and comps[arm][k]['negative_parents']>=12 for k in ('tIoU','sIoU'))
            gates[arm]=dict(v_positive=comps[arm]['vIoU']['mean_delta_pp']>0,t_nonnegative=comps[arm]['tIoU']['mean_delta_pp']>=0,s_nonnegative=comps[arm]['sIoU']['mean_delta_pp']>=0,no_systematic_branch_collapse=not collapse)
            gates[arm]['pass']=all(gates[arm].values())
            denom=comps['oracle']['vIoU']['mean_delta_pp'];gainret[arm]=comps[arm]['vIoU']['mean_delta_pp']/denom if denom!=0 else None
        decision='fixed_rank16_factorized_predictor_candidate' if gates['Shared-R16']['pass'] else ('fixed_rank32_factorized_predictor_candidate' if gates['Shared-R32']['pass'] else 'stop_fixed_low_rank_correction')
        report=dict(status='completed_dual_geometry_pending_root_summary',arms={a:dict(rows=rs,summary=summarize_arm(rs,parents[a])) for a,rs in results.items()},comparisons=comps,versus_full=versus_full,
            query_tails=tails,B1_good_retention=ret,gates=gates,oracle_gain_retention=gainret,decision=decision,
            scalar_tensor_max_abs=maxerr,reused_score_max_abs=reuseerr,CPU_seconds=time.monotonic()-start,
            scope='64 already exposed source queries /16 parents; four queries each, fixed GT oracle; descriptive unadjusted paired parent CIs. No learned predictor, no target/fresh claims.')
        write(out/'REPORT.json',report);write(out/'COMPLETE.json',dict(report_sha=sha(out/'REPORT.json')))
        print('A03_SCORED',decision,comps,flush=True)
    except BaseException as e:
        write(N/'SCORER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
