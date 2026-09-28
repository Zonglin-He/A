"""Independent CPU fixed-endpoint scoring, raw update and objective readback."""
import argparse,sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,verify_seal,check_pins
from vg_tta.external_evidence_metrics import tensor_metrics
from scripts.score_desta3d_v2_reference_audit import score_tube_independently

def cpu_ce(x,y,valid):
    z=x.double().numpy()[valid.numpy()];t=y.numpy()[valid.numpy()];z=z-z.max(-1,keepdims=True)
    return float((np.log(np.exp(z).sum(-1))-z[np.arange(len(t)),t]).mean())

def audit(name):
    dest=OUT/name;assert not (dest/'ROOT_DECISION.json').exists();done,_=verify_seal(dest)
    check_pins(read(dest/'LOCK.json')['pins']);cfg=read(dest/'CONFIG.json');rows=read(dest/'INPUTS.json')
    assert done['actual_steps']==60 and done['predictions']==62
    labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};cases=[];geometry_error=ce_error=norm_error=delta_error=0.;cnt=0
    for i,row in enumerate(rows):
        ep=dest/'episodes'/f'{i:02}';assert all(read(ep/'ZERO_NATIVE_REPLAY.json').values())
        history=read(ep/'TRAJECTORY.json');assert len(history)==31
        values=[];deltas=None;steps=[]
        for step,h in enumerate(history):
            p=torch.load(ep/f'NATIVE_{step:02}.pt',weights_only=False,map_location='cpu');lab=labels[row['key']]
            one=score_tube_independently(p,lab);two=tensor_metrics(p,lab)
            geometry_error=max(geometry_error,max(abs(one[k]-two[k]) for k in ['vIoU','sIoU','tIoU']))
            assert geometry_error<1e-6
            values.append({'step':step,'metrics':one,'interval':p['interval'],'CE':h['CE'],'GT_margin':h['GT_margin'],
                'native_argmax':h['native_argmax'],'format_ok':bool(p['format_ok']),'reference':p['readout']['spatial_reference_token_ids'],'parameter_norm':h['parameter_norm']})
            from vg_tta.desta3d_v3_free_actuation import native_targets
            from vg_tta.desta3d_v3_actuation_support import spatial_positions
            tr=torch.load(ep/f'TRACE_{step:02}.pt',weights_only=False,map_location='cpu')
            branch=row['diagnostic_branch'];proxy={**p,'positions':spatial_positions(tr)} if branch=='spatial' else p
            tar,val=native_targets(lab,proxy,branch);ex=tr['branches'][0 if branch=='event' else 1]['logits']['time' if branch=='event' else 'coordinate']
            ce_error=max(ce_error,abs(cpu_ce(ex,tar,val)-h['CE']));assert ce_error<1e-5
            if branch=='spatial':
                baseline_pred=torch.load(ep/'NATIVE_00.pt',weights_only=False,map_location='cpu')
                assert spatial_positions(tr)==baseline_pred['positions']
                assert p['event_completion']==baseline_pred['event_completion']
                assert torch.equal(p['time_distribution']['endpoint_logits'],baseline_pred['time_distribution']['endpoint_logits'])
            if step<30:
                f=torch.load(ep/f'FORWARD_{step:02}.pt',weights_only=False,map_location='cpu')
                assert f['exact'] and torch.equal(f['native'],f['replay'])
                loss=cpu_ce(f['native'],f['targets'],f['valid']);ce_error=max(ce_error,abs(loss-h['CE']))
                assert ce_error<1e-5
                r=torch.load(ep/f'STEP_{step+1:02}.pt',weights_only=False,map_location='cpu')
                g=r['gradient'].double().numpy().reshape(-1);d=r['actual_delta'].double().numpy().reshape(-1)
                norm=float(np.sqrt(np.dot(g,g)));norm_error=max(norm_error,abs(norm-r['gradient_norm']))
                assert abs(norm-r['gradient_norm'])<max(2e-5,1e-4*norm)
                assert np.isfinite(g).all() and np.isfinite(d).all() and r['counter']==step+1 and r['live_parameter_binding']
                if deltas is None:deltas=np.zeros_like(d)
                deltas+=d;cnt+=1
                steps.append({'counter':step+1,'grad_norm':norm,'delta_norm':float(np.sqrt(np.dot(d,d))),
                       'gradient_dot_actual_delta':float(np.dot(g,d)),'clip':r['clip_triggered']})
        f=torch.load(ep/'FINAL.pt',weights_only=False,map_location='cpu');assert list(f['optimizer']['state'])==[0]
        assert int(f['optimizer']['state'][0]['step'])==30
        err=float(np.max(np.abs(deltas-f['parameter'].double().numpy().reshape(-1))));delta_error=max(delta_error,err);assert err<1e-5
        if cfg['mode']=='span':
            q=torch.load(ep/'BASIS.pt',weights_only=False,map_location='cpu');arr=f['parameter'].double().numpy();basis=q['basis'].double().numpy()
            projected=(arr@basis.T)*q['scale'];assert np.allclose(projected,f['token_delta'].double().numpy(),rtol=2e-5,atol=2e-5)
        baseline=values[0];final=values[-1];branch=row['diagnostic_branch'];delta={k:100*(final['metrics'][k]-baseline['metrics'][k]) for k in ['tIoU','sIoU','vIoU']}
        if branch=='event':
            import re
            m=re.search(r'<\|time_start\|><t(\d+)><t(\d+)>',lab['response']);target=[int(m[1])-1,int(m[2])-1]
            success=delta['tIoU']>0 and final['native_argmax']==target
        else:
            success=delta['sIoU']>0 and final['CE']<baseline['CE']
            assert all(v['reference']==baseline['reference'] and v['interval']==baseline['interval'] for v in values)
        cases.append({'key':row['key'],'branch':branch,'registered_fixed_endpoint_success':success,'delta_pp':delta,
             'baseline':baseline,'final':final,'all_steps':values,'raw_steps':steps,
             'format_failures':sum(not v['format_ok'] for v in values),'clip_count':sum(s['clip'] for s in steps),'grad_dot_actual_delta_negative':sum(s['gradient_dot_actual_delta']<0 for s in steps)})
    decision={'status':'completed_independently_audited','mode':cfg['mode'],'cases':cases,
       'proceed_span':cfg['mode']=='free' and all(c['registered_fixed_endpoint_success'] for c in cases),'fixed_endpoint_only':True,
       'geometry_comparisons':186,'geometry_scalar_tensor_max_error':geometry_error,'CPU_CE_max_error':ce_error,
       'raw_steps':cnt,'gradient_norm_max_error':norm_error,'delta_sum_final_max_error':delta_error,
       'source_GT_training_and_outcome_selected':True,'target_read':False,
       'scope':'Two exposed diagnostic cases, not generalization. Failure of a finite optimizer run is not proof of impossible actuation.',
       'receipt':read(dest/'RECEIPT.json'),'pins':{str(p):sha(p) for p in [Path(__file__),dest/'COMPLETE.json',dest/'PREDICTIONS_SEAL.json',PANEL/'SOURCE_RECORDS.json']}}
    write(dest/'ROOT_DECISION.json',decision)
    lines=['# Source native actuation control','',decision['scope'],'', '|branch|baseline t/s/v %|final t/s/v %|CE before/after|fixed endpoint success|','|---|---|---|---|---|']
    for c in cases:
        vals=lambda r:'/'.join(f"{100*r['metrics'][k]:.6f}" for k in ['tIoU','sIoU','vIoU'])
        lines.append(f"|{c['branch']}|{vals(c['baseline'])}|{vals(c['final'])}|{c['baseline']['CE']:.6f}/{c['final']['CE']:.6f}|{c['registered_fixed_endpoint_success']}|")
    lines+=['','All31 states per case, exact replay checks, actual30 Adam counters, raw gradients/deltas and positive/negative native outcomes retained. Intermediate steps are not selected as final.',
        '',f"Proceed to separate frozen-output-span registration: {decision['proceed_span']}"]
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k not in ['cases','pins']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);audit(p.parse_args().name)
