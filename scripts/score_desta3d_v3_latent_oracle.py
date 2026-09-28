"""Seal-first source oracle readback; source GT was already used for latent masks."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,check_pins,verify_seal,local_dependencies
METRICS=('vIoU','sIoU','tIoU');ARMS=('original','temporal','spatial','dual','wrong_temporal','wrong_spatial')

def preflight(name):
    from scripts.score_desta3d_v2_source_task_control_v2 import controls
    paths=local_dependencies([Path(__file__),ROOT/'vg_tta/external_evidence_metrics.py',ROOT/'scripts/score_desta3d_v2_reference_audit.py',
        ROOT/'tests/test_desta3d_v3_latent_oracle_score.py',ROOT/'scripts/crosscheck_desta3d_v3_oracle_summary.py'])
    write(OUT/name/'SCORER_PREFLIGHT.json',{'status':'passed','controls':controls(),'pins':{str(p):sha(p) for p in paths},
        'source_label_sha':sha(PANEL/'SOURCE_RECORDS.json'),'source_GT_already_used_for_oracle':True,'target_read':False})

def score(name):
    import torch
    from vg_tta.external_evidence_metrics import tensor_metrics
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.desta3d_v2_prediction_contract import validate_prediction
    dest=OUT/name;report_dir=dest/'independent_readback_v1';assert not report_dir.exists()
    try:
        pre=read(dest/'SCORER_PREFLIGHT.json');check_pins(pre['pins'])
        done,s=verify_seal(dest);assert done['queries']==16 and done['predictions']==s['predictions']==96
        assert done['extra_native_identity_control']==1 and done['optimizer_steps']==0
        check_pins(read(dest/'LOCK.json')['pins']);cfg=read(dest/'CONFIG.json')
        assert cfg['source_GT_in_worker'] and not cfg['target_input'] and cfg['alpha']==.25 and cfg['optimizer_steps']==0
        rows=read(dest/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
        payloads={a:[] for a in ARMS};identities=[]
        for i,row in enumerate(rows):
            ep=dest/'episodes'/f'{i:02}';ident=read(ep/'INPUT.json');e=read(ep/'COMPLETE.json')
            assert e['key']==ident['key']==row['key'] and e['exact_state'] and e['optimizer_steps']==0
            assert ident['frame_ids']==row['input']['frame_ids'] and not ident['pixel_intervention']
            assert ident['source_GT_mask_sha']==sha(ep/'ORACLE_MASKS.pt') and ident['source_label_sha']==cfg['source_label_sha']
            replay=read(ep/'BASELINE_REPLAY.json');assert len(replay)>=10 and all(replay.values())
            masks=torch.load(ep/'ORACLE_MASKS.pt',map_location='cpu',weights_only=False)
            assert masks['support']['frame_ids']==row['input']['frame_ids']
            for b in ('event','spatial'):
                m=masks[b];assert m.ndim==4 and m.shape[0]==1 and m.shape[1]==len(row['input']['frame_ids'])
                assert torch.isfinite(m).all() and (m>=0).all() and (m<=1).all()
            for arm in ARMS:
                p=torch.load(ep/(arm+'.pt'),map_location='cpu',weights_only=False);validate_prediction(p,len(row['input']['frame_ids']))
                assert p['key']==row['key'] and p['source']==row['source'] and p['adapter_sha']==cfg['adapter_sha']
                assert p['video_sha256']==row['input']['video_sha256'] and p['frame_ids']==row['input']['frame_ids']
                assert p['support']==ident['support'] and p['preprocess']==ident['preprocess']
                assert p['GT_read']==p['oracle_latent']==(arm!='original') and p['worker_source_GT_read']
                assert not p['decoder_GT_prefix'] and not p['target_GT_read'] and p['optimizer_steps']==0
                expected={'temporal':{'event'},'spatial':{'spatial'},'dual':{'event','spatial'},'wrong_temporal':{'event'},'wrong_spatial':{'spatial'},'original':set()}[arm]
                # Native format failure may omit spatial decode; event prefill still executes both adapter readers.
                assert {r['branch'] for r in p['latent_log']}==expected
                payloads[arm].append(p)
            if i==0:
                identity=read(ep/'IDENTITY_CONTROL.json');assert len(identity)>=10 and all(identity.values())
            identities.append({'key':row['key'],'source':row['source'],'oracle_support':masks['support'],
                'original_replay_exact':True,'GT_used_for_latent_only':True,
                'negative_control':torch.load(ep/'WRONG_MASKS.pt',map_location='cpu',weights_only=False)['diagnostic']})
        write(report_dir/'PRE_SCORE_AUDIT.json',{'status':'passed','time':time.time(),'primary_predictions':96,'identity_controls':1,
            'source_GT_already_read_by_worker_for_oracle':True,'source_GT_training_exposed':True,'target_read':False,
            'seal_sha':sha(dest/'PREDICTIONS_SEAL.json'),'no_optimizer_or_forced_GT_prefix':True,'identities':identities})
        assert sha(PANEL/'SOURCE_RECORDS.json')==pre['source_label_sha']==cfg['source_label_sha']
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};results={a:[] for a in ARMS};maxerr=0.
        for i,row in enumerate(rows):
            for arm in ARMS:
                p=payloads[arm][i];lab=labels[row['key']]
                metrics=score_tube_independently(p,lab);other=tensor_metrics(p,lab)
                err=max(abs(metrics[m]-other[m]) for m in METRICS);maxerr=max(maxerr,err);assert err<1e-6
                results[arm].append({'key':row['key'],'source':row['source'],'metrics':metrics,'interval':p['interval'],
                    'format_ok':p['format_ok'],'positions':p['positions']})
        parents={a:summarize_parents(x) for a,x in results.items()}
        comps={a+'_minus_original':{m:paired_parent_bootstrap(parents[a],parents['original'],m) for m in METRICS} for a in ARMS[1:]}
        wrong_comps={}
        for good,bad,branch in [('temporal','wrong_temporal','temporal'),('spatial','wrong_spatial','spatial')]:
            eligible={r['source'] for r in identities if r['negative_control'][branch]['eligible_changed_support']}
            assert eligible,'No discriminating negative-control sources'
            wrong_comps[good+'_minus_'+bad]={'eligible_parents':len(eligible),'excluded_non_discriminating':[r['source'] for r in identities if r['source'] not in eligible],
                'all_parents':{m:paired_parent_bootstrap(parents[good],parents[bad],m) for m in METRICS},
                'eligible_only':{m:paired_parent_bootstrap({p:v for p,v in parents[good].items() if p in eligible},{p:v for p,v in parents[bad].items() if p in eligible},m) for m in METRICS}}
        retention={}
        for m in ('vIoU','tIoU'):
            good={r['key'] for r in results['original'] if r['metrics'][m]>.5}
            retention[m]={'eligible':len(good),'definition':'Original B1 >.5 retained if candidate >.5',
                'arms':{a:{'retained':sum(r['key'] in good and r['metrics'][m]>.5 for r in x),
                    'lost':[r['key'] for r in x if r['key'] in good and r['metrics'][m]<=.5]} for a,x in results.items()}}
        report={'status':'source_GT_branch_latent_oracle_completed','arms':{a:{'rows':x,'summary':summarize_arm(x,parents[a])} for a,x in results.items()},
            'comparisons':comps,'correct_minus_wrong':wrong_comps,'retention':retention,'identities':identities,'scalar_tensor_max_error':maxerr,
            'scope':'16 exposed source-training parents; GT creates latent masks; no optimizer/target/OPD. Not unlabeled TTA or held-out generalization.',
            'configuration':'Frozen official PTD4B+B1; alpha .25 after reader/LN/SiLU before output projection; RGB and gates unchanged',
            'primary_contrasts':['temporal-original tIoU','spatial-original sIoU'],
            'CI':'Descriptive 10000 paired-parent bootstrap, seed20260927, no multiplicity correction','target_read':False}
        write(report_dir/'REPORT.json',report)
        lines=['# Source GT privileged 3D branch-latent oracle','',report['scope'],'',report['configuration'],'',
            '|Latent|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
        for arm in ARMS:
            v=report['arms'][arm]['summary']['parent_macro'];lines.append('|'+arm+'|'+'|'.join(f'{100*v[m]:.6f}' for m in ['tIoU','sIoU','vIoU'])+'|')
        lines+=['','|Versus original|metric|delta pp|95% CI|positive / negative|>5pp losses|','|---|---|---:|---|---|---:|']
        for arm in ARMS[1:]:
            for m in METRICS:
                c=comps[arm+'_minus_original'][m];lines.append(f"|{arm}|{m}|{c['mean_delta_pp']:+.6f}|{c['bootstrap_ci95_pp']}|{c['positive_parents']} / {c['negative_parents']}|{c['severe_loss_below_minus5pp']}|")
        lines+=['','|Correct minus matched wrong (eligible)|metric|parents|delta pp|95% CI|','|---|---|---:|---:|---|']
        for contrast,d in wrong_comps.items():
            for m in METRICS:
                c=d['eligible_only'][m];lines.append(f"|{contrast}|{m}|{d['eligible_parents']}|{c['mean_delta_pp']:+.6f}|{c['bootstrap_ci95_pp']}|")
        lines+=['','S and wrong-S native comparisons share original reference, interval and frame support by structural contract. T/dual may change those conditions; their sIoU is final-tube evidence, not fixed-support spatial gain.','', 'Full per-parent positives, negatives, failures, original-good retention and oracle support remain in REPORT.json.',
            'This one fixed actuation interface does not establish all 3D information sufficiency, an external teacher, OPD benefit or deployment readiness.']
        (report_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write(report_dir/'COMPLETE.json',{'status':'scored','report_sha':sha(report_dir/'REPORT.json'),'geometry_comparisons':288,'max_error':maxerr})
    except BaseException as exc:
        if not (report_dir/'FAILURE.json').exists():write(report_dir/'FAILURE.json',{'error':repr(exc),'traceback':__import__('traceback').format_exc()})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);p.add_argument('--name',required=True)
    a=p.parse_args();globals()[a.action](a.name)
