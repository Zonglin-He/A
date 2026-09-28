"""Seal-first Q0 tables; no source label open until every arm passes identity checks."""
import argparse,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import *
METRICS=('vIoU','sIoU','tIoU');ARMS=('original','temporal','spatial','combined')

def preflight(name):
    from scripts.score_desta3d_v2_source_task_control_v2 import controls
    dest=OUT/name;paths=[Path(__file__),ROOT/'vg_tta/external_evidence_metrics.py',ROOT/'vg_tta/external_privileged_views.py',
        ROOT/'scripts/score_desta3d_v2_reference_audit.py',ROOT/'tests/test_external_qualification_metrics.py']
    write(dest/'SCORER_PREFLIGHT.json',{'status':'passed','controls':controls(),'pins':{str(p):sha(p) for p in paths},
        'source_label_sha':sha(PANEL/'SOURCE_RECORDS.json'),'source_training_exposed':True,'target_read':False})

def score(name):
    import torch,numpy as np
    from vg_tta.external_evidence_metrics import external_prediction,scalar_external,tensor_metrics
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap
    from vg_tta.desta3d_v2_prediction_contract import validate_prediction
    dest=OUT/name;report_dir=dest/'independent_readback_v1';assert not report_dir.exists()
    try:
        pre=read(dest/'SCORER_PREFLIGHT.json');check_pins(pre['pins'])
        done,bseal=verify_seal(dest);assert done['queries']==16 and bseal['predictions']==64
        check_pins(read(dest/'LOCK.json')['pins'])
        cfg=read(dest/'CONFIG.json');teacher=Path(cfg['teacher']);tdone,tseal=verify_seal(teacher)
        assert tdone['queries']==16 and tseal['predictions']==16
        check_pins(read(teacher/'LOCK.json')['pins'])
        rows=read(dest/'INPUTS.json');assert rows==read(teacher/'INPUTS.json') and len({r['source'] for r in rows})==16
        payloads={a:[] for a in (*ARMS,'external')};identities=[]
        for i,row in enumerate(rows):
            ep=dest/'episodes'/f'{i:02}';te=teacher/'episodes'/f'{i:02}'
            ident=read(ep/'INPUT.json');inp=read(te/'INPUT.json');ev=read(te/'EVIDENCE.json')
            assert ident['key']==inp['key']==row['key'] and ident['frame_ids']==inp['frame_ids']==row['input']['frame_ids']
            assert ident['physical_pixel_sha']==inp['pixel_sha256']
            assert ident['evidence_sha']==sha(te/'EVIDENCE.json') and all(read(ep/'BASELINE_REPLAY.json').values())
            payloads['external'].append(external_prediction(ev,row['input']['frame_ids']))
            original_support=None
            for arm in ARMS:
                p=torch.load(ep/(arm+'.pt'),map_location='cpu',weights_only=False);validate_prediction(p,len(row['input']['frame_ids']))
                assert p['key']==row['key'] and p['source']==row['source'] and p['adapter_sha']==cfg['adapter_sha']
                assert p['video_sha256']==row['input']['video_sha256'] and p['frame_ids']==row['input']['frame_ids']
                assert p['evidence_sha']==sha(te/'EVIDENCE.json') and not p['GT_read'] and not p['target_GT_read'] and p['optimizer_steps']==0
                if original_support is None:original_support=p['support'];assert p['preprocess']['pixel_sha']==inp['pixel_sha256']
                else:assert p['support']==original_support
                payloads[arm].append(p)
            identities.append({'key':row['key'],'source':row['source'],'spatial_support':read(te/'SUPPORT.json'),
                'errors':ev['errors'],'temporal_usable':ev['temporal_usable'],'spatial_usable':ev['spatial_usable'],'format_valid':ev['format_valid']})
        write(report_dir/'PRE_GT_AUDIT.json',{'status':'passed','time':time.time(),'teacher_predictions':16,'PTD_predictions':64,
            'source_label_content_not_yet_opened':True,'target_read':False,'identities':identities,
            'teacher_seal_sha':sha(teacher/'PREDICTIONS_SEAL.json'),'PTD_seal_sha':sha(dest/'PREDICTIONS_SEAL.json')})
        assert sha(PANEL/'SOURCE_RECORDS.json')==pre['source_label_sha']==cfg['source_label_sha']
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};results={a:[] for a in payloads};maxerr=0.
        for i,row in enumerate(rows):
            for arm,ps in payloads.items():
                lab=labels[row['key']];p=ps[i]
                metrics=scalar_external(p,lab) if arm=='external' else score_tube_independently(p,lab)
                independent=tensor_metrics(p,lab,external=arm=='external')
                err=max(abs(metrics[m]-independent[m]) for m in METRICS);maxerr=max(maxerr,err);assert err<1e-6
                results[arm].append({'key':row['key'],'source':row['source'],'metrics':metrics})
        parents={a:summarize_parents(x) for a,x in results.items()}
        comps={a+'_minus_original':{m:paired_parent_bootstrap(parents[a],parents['original'],m) for m in METRICS} for a in results if a!='original'}
        retention={}
        for m in ('vIoU','tIoU'):
            good={r['key'] for r in results['original'] if r['metrics'][m]>.5}
            retention[m]={'eligible':len(good),'definition':'B1 original >.5, retained if candidate >.5',
                'arms':{a:{'retained':sum(r['key'] in good and r['metrics'][m]>.5 for r in x),
                    'lost':[r['key'] for r in x if r['key'] in good and r['metrics'][m]<=.5]} for a,x in results.items()}}
        report={'status':'Q0_source_engineering_qualification','arms':{a:{'rows':x,'summary':summarize_arm(x,parents[a])} for a,x in results.items()},
            'comparisons':comps,'retention':retention,'identities':identities,'scalar_tensor_max_error':maxerr,
            'scope':'16 exposed source training parents; external teacher pretraining overlap unresolved; not unseen generalization',
            'PTD_state':'official PTD4B + frozen B1, same state across all four views; cancelled v3 excluded',
            'external_scoring':'continuous physical inclusive endpoints converted to half-open +1; local valid/median/interpolated boxes on exact common grid; missing boxes zero; format errors retained',
            'CI':'descriptive 10000 paired-parent bootstrap, seed20260927, no multiplicity correction','target_read':False}
        write(report_dir/'REPORT.json',report)
        lines=['# Q0 Teacher Evidence Qualification','',report['scope'],'','|Input|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
        row=report['arms']['external']['summary']['parent_macro'];lines.append('|LLaVA-ST direct|'+'|'.join(f'{100*row[m]:.6f}' for m in ['tIoU','sIoU','vIoU'])+'|')
        lines+=['','# Q0 Privileged Policy Qualification','',report['PTD_state'],'','|Input|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
        for a in ARMS:
            r=report['arms'][a]['summary']['parent_macro'];lines.append('|'+a+'|'+'|'.join(f'{100*r[m]:.6f}' for m in ['tIoU','sIoU','vIoU'])+'|')
        lines+=['','|Compared with original|metric|delta pp|95% CI|>5pp losses|','|---|---|---:|---|---:|']
        for a in ARMS[1:]:
            for m in METRICS:
                c=comps[a+'_minus_original'][m];lines.append(f"|{a}|{m}|{c['mean_delta_pp']:+.6f}|{c['bootstrap_ci95_pp']}|{c['severe_loss_below_minus5pp']}|")
        lines+=['','All raw errors, local fallbacks, coverage/gap diagnostics and per-parent positive/negative cases remain in REPORT.json.',
            'A Q0 mean advantage alone does not authorize OPD or establish held-out teacher superiority. Native controls remain a separate stage.']
        (report_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write(report_dir/'COMPLETE.json',{'status':'scored','report_sha':sha(report_dir/'REPORT.json'),'geometry_comparisons':240,'max_error':maxerr})
    except BaseException as e:
        if not (report_dir/'FAILURE.json').exists():write(report_dir/'FAILURE.json',{'error':repr(e),'traceback':__import__('traceback').format_exc()})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);p.add_argument('--name',required=True)
    a=p.parse_args();globals()[a.action](a.name)
