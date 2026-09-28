"""Sealed PANEL16 readback: independent raw algebra and native geometry."""
import sys,time,argparse,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,check_pins,verify_seal,local_dependencies,tensor_sha
from scripts.desta3d_v3_decomposition_oracle import ARMS,OLD,B1,ADAPTER_SHA

def preflight(name):
    from scripts.score_desta3d_v2_source_task_control_v2 import controls
    paths=local_dependencies([Path(__file__),ROOT/'scripts/crosscheck_desta3d_v3_decomposition.py'])
    write(OUT/name/'SCORER_PREFLIGHT.json',dict(status='passed',controls=controls(),
        pins={str(p):sha(p) for p in paths},source_labels=sha(PANEL/'SOURCE_RECORDS.json'),
        tolerances={'raw_scalar_rtol':2e-7,'raw_scalar_atol':2e-8,'direction_relative_L2':3e-6,
                    'budget_relative':2e-6,'CE_atol':2e-5,'geometry_atol':1e-6},
        no_new_GT_until_complete=True,source_GT_already_used_by_oracle=True))

def score(name):
    import numpy as np,torch
    from scripts.desta3d_v3_privileged_ptd_qualification import equal
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,summarize_arm,paired_parent_bootstrap,iou_xyxy
    from vg_tta.external_evidence_metrics import tensor_metrics
    from vg_tta.desta3d_v3_actuation_support import spatial_positions
    from vg_tta.desta3d_v3_free_actuation import native_targets
    torch.set_num_threads(4);d=OUT/name;o=d/'independent_readback_v1';assert not o.exists()
    start=time.monotonic();pre=read(d/'SCORER_PREFLIGHT.json');tol=pre['tolerances']
    def load(p):return torch.load(p,map_location='cpu',weights_only=False)
    def arr(t):return t.detach().float().numpy().astype(np.float64)
    def dot(a,b):
        aa=a.reshape(-1);bb=b.reshape(-1)
        return math.fsum(float(np.dot(aa[k:k+262144],bb[k:k+262144])) for k in range(0,len(aa),262144))
    def norm(a):return math.sqrt(max(0.,dot(a,a)))
    def cosine(a,b):return dot(a,b)/(norm(a)*norm(b)) if norm(a)*norm(b) else None
    scalar_errors=[];direction_errors=[];ces=[]
    def check(a,b):
        assert np.allclose(a,b,rtol=tol['raw_scalar_rtol'],atol=tol['raw_scalar_atol']),(a,b)
        scalar_errors.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    def ce(logits,targets,valid):
        x=arr(logits)[valid.numpy()];y=targets.numpy()[valid.numpy()];m=x.max(-1)
        return float(np.mean(m+np.log(np.exp(x-m[:,None]).sum(-1))-x[np.arange(len(y)),y]))
    try:
        check_pins(pre['pins']);done,sealed=verify_seal(d);check_pins(read(d/'LOCK.json')['pins'])
        assert done['predictions']==sealed['predictions']==96 and done['queries']==16 and done['backward_calls']==32 and done['finite_reads']==128
        assert done['optimizer_steps']==0 and not done['target_read']
        # Root artifact was outside the worker's episode-only seal. Bind it here
        # before scoring, then verify its subspaces against exact B1 weights.
        write(d/'ROOT_BASIS_SEAL.json',dict(basis_sha=sha(d/'BASIS.pt'),time=time.time(),
            disclosure='Worker seal enumerates episodes; root BASIS independently bound and B1 span-checked before scoring'))
        qb=load(d/'BASIS.pt');q={k:arr(qb[k]) for k in ['T','S','J']}
        state=load(B1)['adapter'];span_audit={}
        for k,b in [('T','event'),('S','spatial')]:
            w=arr(state['out_proj_'+b+'.weight']);res=norm(w-q[k]@(q[k].T@w))/norm(w)
            orth=float(np.max(np.abs(q[k].T@q[k]-np.eye(q[k].shape[1]))));assert res<2e-6 and orth<2e-6
            span_audit[k]=dict(rank=q[k].shape[1],relative_range_residual=res,orthogonal_error=orth)
        for k in ['T','S']:assert norm(q[k]-q['J']@(q['J'].T@q[k]))/norm(q[k])<2e-6
        assert np.max(np.abs(q['J'].T@q['J']-np.eye(q['J'].shape[1])))<2e-6
        cfg=read(d/'CONFIG.json');rows=read(d/'INPUTS.json');assert len(rows)==len({r['source'] for r in rows})==16
        assert sha(PANEL/'SOURCE_RECORDS.json')==pre['source_labels']==cfg['source_label_sha']
        identities=[];geometries=[];finite_all=[];payload={a:[] for a in ARMS}
        for i,row in enumerate(rows):
            ep=d/'episodes'/f'{i:02}';comp=read(ep/'COMPLETE.json');ident=read(ep/'INPUT.json');z=load(ep/'GEOMETRY.pt')
            assert comp['key']==ident['key']==row['key'] and comp['exact_state'] and comp['optimizer_steps']==0
            assert all(read(ep/'BASELINE_REPLAY.json').values())
            f=arr(z['stock']);g={b:arr(t) for b,t in z['gradients'].items()};delta={k:arr(t) for k,t in z['deltas'].items()}
            assert tensor_sha(z['stock'])==ident['common_F_sha']==ident['support']['visual_grid']==comp['common_F_sha']
            assert g['event'].shape==g['spatial'].shape==f.shape
            log=z['geometry'];check(cosine(g['event'],g['spatial']),log['gradient_cosine']);check(norm(f),log['stock_norm'])
            for k,b in [('T','event'),('S','spatial')]:
                check(norm(g[b]),log['gradient_norms'][k]);projected=g[b]@q[k]@q[k].T
                check(norm(projected)/norm(g[b]),log['projection_fraction'][b])
                target=cfg['ratios'][b]*norm(f);expected=-projected*target/norm(projected) if norm(projected) else projected*0
                err=norm(delta[k]-expected)/max(norm(expected),1e-30);direction_errors.append(err);assert err<tol['direction_relative_L2']
            balance=g['event']/norm(g['event'])+g['spatial']/norm(g['spatial']);proj=balance@q['J']@q['J'].T
            budget=math.sqrt(norm(delta['T'])**2+norm(delta['S'])**2);expected=-proj*budget/norm(proj) if norm(proj) else proj*0
            err=norm(delta['J']-expected)/max(norm(expected),1e-30);direction_errors.append(err);assert err<tol['direction_relative_L2']
            assert norm(delta['J_pass']-delta['J']/math.sqrt(2))/max(norm(delta['J_pass']),1e-30)<tol['direction_relative_L2']
            check(norm(delta['J'])**2,norm(delta['T'])**2+norm(delta['S'])**2)
            check(2*norm(delta['J_pass'])**2,norm(delta['T'])**2+norm(delta['S'])**2)
            for k in delta:
                check(norm(delta[k]),log['norms'][k])
                for b,n in [('event','T'),('spatial','S')]:check(-dot(g[b],delta[k]),log['descent_dot'][n][k])
            check(cosine(delta['T'],delta['S']),log['direction_cosine'])
            initial={};fin={}
            trace=load(ep/'BASE_TRACE.pt')
            for b in ['event','spatial']:
                v=load(ep/(b+'_INITIAL_FORWARD.pt'));ob=z['objectives'][b]
                assert v['exact'] and torch.equal(v['native'],v['replay']) and v['F_sha']==ident['common_F_sha']
                assert torch.equal(v['targets'],ob['targets']) and torch.equal(v['valid'],ob['valid'])
                t=trace['branches'][0 if b=='event' else 1]['logits'][ob['kind']]
                assert torch.equal(v['native'],t)
                val=ce(t,v['targets'],v['valid']);ces.append(abs(val-ob['CE']));assert ces[-1]<tol['CE_atol'];initial[b]=val
                for k in delta:
                    x=load(ep/f'FINITE_{k}_{b}.pt');assert torch.equal(x['targets'],ob['targets']) and torch.equal(x['valid'],ob['valid'])
                    assert x['baseline_trace_sha']==sha(ep/'BASE_TRACE.pt') and x['F_sha']==tensor_sha(z['stock']+z['deltas'][k])
                    val=ce(x['logits'],x['targets'],x['valid']);ces.append(abs(val-x['CE']));assert ces[-1]<tol['CE_atol']
                    fin[k+'_'+b]=dict(CE=val,delta_CE=val-initial[b],valid_actions=int(x['valid'].sum()))
            geometries.append(dict(index=i,key=row['key'],source=row['source'],**log))
            finite_all.append(dict(index=i,key=row['key'],source=row['source'],initial=initial,finite=fin))
            for a in ARMS:
                p=load(ep/(a+'.pt'));assert p['key']==row['key'] and p['source']==row['source'] and p['support']==ident['support']
                assert p['preprocess']==ident['preprocess'] and p['adapter_sha']==ADAPTER_SHA and not p['target_read'] and p['optimizer_steps']==0
                payload[a].append(p)
                if a!='original':
                    for b,v in read(ep/(a+'_INJECTION.json')).items():
                        if 'missing' in v:assert not p['format_ok'];continue
                        assert v['exact_expected'];k={'T_only':('T',None),'S_only':(None,'S'),'decomposed':('T','S'),
                            'joint':('J','J'),'joint_pass_matched':('J_pass','J_pass')}[a][0 if b=='event' else 1]
                        check(v['delta_norm'],0 if k is None else norm(delta[k]))
                        actual=z['stock']+(z['deltas'][k] if k else 0);assert tensor_sha(actual)==v['new_F_sha']
                        check(norm(arr(actual)-f),v['realized_FP32_F_delta_norm'])
            assert equal(payload['S_only'][-1]['time_distribution'],payload['original'][-1]['time_distribution'])
            identities.append(dict(index=i,key=row['key'],common_F_sha=ident['common_F_sha']))
            del z,trace,f,g,delta
        write(o/'PRE_SCORE_AUDIT.json',dict(status='passed',time=time.time(),predictions=96,gradient_pairs=16,
            raw_numeric_checks=len(scalar_errors),raw_scalar_max_abs_error=max(scalar_errors),direction_max_relative_L2=max(direction_errors),
            full_vocab_CE_checks=len(ces),full_vocab_CE_max_error=max(ces),identities=identities,spans=span_audit,
            union_rank=q['J'].shape[1],source_GT_already_used_for_oracle=True,target_read=False,
            injection_limit='Actual postcast endpoint equals independently evaluated frozen-adapter result asserted and hashed by worker; full postcast tensors not stored'))
        # Native utility starts only after complete/seal/raw/identity readback.
        labels={r['key']:r for r in read(PANEL/'SOURCE_RECORDS.json')};results={a:[] for a in ARMS};mx=0.;fixed=[]
        for i,row in enumerate(rows):
            ep=d/'episodes'/f'{i:02}';lab=labels[row['key']];trace=load(ep/'BASE_TRACE.pt');base=payload['original'][i]
            for b in ['event','spatial']:
                support={**base,'positions':spatial_positions(trace)} if b=='spatial' else base
                y,v=native_targets(lab,support,b)
                if b=='spatial':y=torch.tensor(trace['coordinate_ids'])[y]
                old=load(ep/(b+'_INITIAL_FORWARD.pt'));assert torch.equal(y,old['targets']) and torch.equal(v,old['valid'])
            pos=spatial_positions(trace);ids={int(x):j/1000. for j,x in enumerate(trace['coordinate_ids'])}
            diag={}
            for k in ['original','T','S','J','J_pass']:
                ll=trace['branches'][1]['logits']['coordinate'] if k=='original' else load(ep/f'FINITE_{k}_spatial.pt')['logits']
                tokens=ll.argmax(-1).tolist();values=[];invalid=0
                for n,p in enumerate(pos):
                    if not lab['box_valid'][p]:continue
                    if any(t not in ids for t in tokens[n]):values.append(0.);invalid+=1;continue
                    box=[ids[t] for t in tokens[n]];values.append(iou_xyxy(box,lab['boxes_xyxy'][p]))
                diag[k]=dict(annotated_anchors=len(values),full_vocab_argmax_coordinate_IoU_mean=float(np.mean(values)),
                    noncoordinate_argmax_frames=invalid,scope='fixed baseline anchors; coordinate distribution only, not native grammar/tube')
            fixed.append(dict(index=i,key=row['key'],diagnostics=diag))
            for a in ARMS:
                p=payload[a][i];v=score_tube_independently(p,lab);vv=tensor_metrics(p,lab)
                err=max(abs(v[m]-vv[m]) for m in ['vIoU','sIoU','tIoU']);mx=max(mx,err);assert err<tol['geometry_atol']
                results[a].append(dict(key=row['key'],source=row['source'],metrics=v,format_ok=p['format_ok'],interval=p['interval'],
                    same_reference=p['readout']['spatial_reference_token_ids']==base['readout']['spatial_reference_token_ids'],
                    same_interval=p['interval']==base['interval']))
        parents={a:summarize_parents(r) for a,r in results.items()};ms=['vIoU','sIoU','tIoU']
        contrasts=[(a,'original') for a in ARMS[1:]]+[('decomposed','joint'),('decomposed','joint_pass_matched'),('decomposed','T_only'),('decomposed','S_only')]
        comparisons={a+'_minus_'+b:{m:paired_parent_bootstrap(parents[a],parents[b],m) for m in ms} for a,b in contrasts}
        retention={}
        for m in ['vIoU','tIoU']:
            good={r['key'] for r in results['original'] if r['metrics'][m]>.5}
            retention[m]=dict(eligible=len(good),arms={a:dict(retained=sum(r['key'] in good and r['metrics'][m]>.5 for r in rr),
                lost=[r['key'] for r in rr if r['key'] in good and r['metrics'][m]<=.5]) for a,rr in results.items()})
        cos=np.array([x['gradient_cosine'] for x in geometries]);dots={};finite_summary={}
        for b in ['T','S']:
            for k in ['T','S','J','J_pass']:
                v=np.array([x['descent_dot'][b][k] for x in geometries]);dots[b+'_on_'+k]=dict(mean=float(v.mean()),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()))
        for k in finite_all[0]['finite']:
            v=np.array([x['finite'][k]['delta_CE'] for x in finite_all]);finite_summary[k]=dict(mean_delta_CE=float(v.mean()),decreased=int((v<0).sum()),increased=int((v>0).sum()),unchanged=int((v==0).sum()))
        candidate=comparisons['decomposed_minus_original']['vIoU'];gate=candidate['mean_delta_pp']>0 and candidate['severe_loss_below_minus5pp']==0
        gate=gate and all(comparisons['decomposed_minus_'+b]['vIoU']['bootstrap_ci95_pp'][0]>0 for b in ['joint','joint_pass_matched'])
        report=dict(status='source_decomposition_completed_independently_audited',arms={a:dict(rows=r,summary=summarize_arm(r,parents[a])) for a,r in results.items()},
            comparisons=comparisons,retention=retention,geometry=geometries,finite_objectives=finite_all,fixed_spatial=fixed,
            gradient_summary=dict(count=16,mean=float(cos.mean()),median=float(np.median(cos)),minimum=float(cos.min()),maximum=float(cos.max()),
                quantiles=np.quantile(cos,[0,.25,.5,.75,1]).tolist(),negative=int((cos<0).sum()),positive=int((cos>0).sum()),near_orthogonal_abs_le_point1=int((abs(cos)<=.1).sum()),
                histogram_edges=[-1,-.1,0,.1,1],histogram_counts=np.histogram(cos,bins=[-1,-.1,0,.1,1])[0].tolist()),
            local_descent_summary=dots,finite_objective_summary=finite_summary,registered_practical_gate=bool(gate),
            geometry_scalar_tensor_max_error=mx,scope='Same16 exposed Vid training parents; one analytic GT-gradient correction, no optimizer/GTprefix/target. Query fixed, full shared F gradient; source-derived locked large radii. Joint union and unit-gradient balance; both parameter and two-pass energy budgets. Descriptive unadjusted CI; no claim of universal decomposition need, latent sufficiency, learned-reader or OPD success.',
            CPU_seconds=time.monotonic()-start)
        write(o/'REPORT.json',report)
        lines=['# PANEL16 shared-F decomposition oracle','',report['scope'],'','|arm|tIoU %|sIoU %|vIoU %|','|---|---:|---:|---:|']
        for a in ARMS:lines.append('|'+a+'|'+'|'.join(f"{100*report['arms'][a]['summary']['parent_macro'][m]:.6f}" for m in ['tIoU','sIoU','vIoU'])+'|')
        lines+=['','Gradient summary: '+str(report['gradient_summary']),'','Local descent dots: '+str(dots),'','Finite fixed-support CE: '+str(finite_summary),
            '',f'Registered practical gate: {gate}. All96 predictions,16 gradient pairs, signs, zero effects, tails, failures and retention are in REPORT.json.']
        (o/'REPORT.md').write_text('\n'.join(lines)+'\n');write(o/'COMPLETE.json',dict(status='scored',report_sha=sha(o/'REPORT.json'),geometry_values=288,CPU_seconds=time.monotonic()-start))
    except BaseException as e:
        write(o/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','score']);p.add_argument('--name',required=True)
    a=p.parse_args();globals()[a.action](a.name)
