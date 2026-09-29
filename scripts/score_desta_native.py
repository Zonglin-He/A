"""Exposed-dev hyperparameter selection; labels never reach the update worker."""
import argparse
from functools import cmp_to_key
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.desta_native_common import D,A3,read,write,load,sha,verified,check_pins


def score(phase):
    import numpy as np
    import torch
    from scripts.desta3d_v3_a0_fast_screen import labels_for
    from scripts.score_desta3d_v2_reference_audit import score_tube_independently,summarize_parents,paired_parent_bootstrap
    from vg_tta.external_evidence_metrics import tensor_metrics
    torch.set_num_threads(4);start=time.monotonic();out=D/'scores'/phase
    if (out/'COMPLETE.json').exists():
        check_pins(read(out/'COMPLETE.json')['pins']);return
    seal=read(D/(phase+'_SEAL.json'));audit=read(D/'audits'/phase/'REPORT.json');assert audit['status']=='passed'
    check_pins({str(D/p):h for p,h in seal['files'].items()})
    rows=read(D/'DEV64.json');indices=seal['indices'];keys=list(seal['configurations'])
    for i in indices:
        assert verified(D/'native'/f'{i:02}'/'identity')
        for key in keys:assert verified(D/'native'/f'{i:02}'/key)
    write(out/'PRE_SCORE_AUDIT.json',dict(seal_sha=sha(D/(phase+'_SEAL.json')),raw_audit_sha=sha(D/'audits'/phase/'REPORT.json'),
        labels='only requested exposed Dev indices',source_GT_chooses_config=True,GT_in_update=False,fresh_read=False,target_read=False))
    labels=labels_for([rows[i] for i in indices],'dev')
    arms=['B1','GT-R16',*keys];scores={a:[] for a in arms};maxerr=0.
    for i in indices:
        row=rows[i]
        for key in arms:
            if key=='B1':path=D/'native'/f'{i:02}'/'identity/B1.pt'
            elif key=='GT-R16':
                path=A3/'native/episodes'/f'{i:04}'/'Shared-R16.pt'
                oldseal=read(A3/'native/PREDICTIONS_SEAL.json')['files'];assert sha(path)==oldseal[str(path.relative_to(A3/'native'))]
            else:path=D/'native'/f'{i:02}'/key/'PREDICTION.pt'
            pred=load(path);assert pred['key']==row['key'] and pred['source']==row['source']
            m=score_tube_independently(pred,labels[row['key']]);other=tensor_metrics(pred,labels[row['key']])
            maxerr=max(maxerr,max(abs(m[k]-other[k]) for k in ('tIoU','sIoU','vIoU')));assert maxerr<1e-6
            scores[key].append(dict(key=row['key'],source=row['source'],metrics=m))
    parents={a:summarize_parents(s) for a,s in scores.items()};metrics=('tIoU','sIoU','vIoU')
    summary={a:{m:float(np.mean([p[m] for p in ps.values()])*100) for m in metrics} for a,ps in parents.items()}
    diffs={a:{m:paired_parent_bootstrap(ps,parents['B1'],m) for m in metrics} for a,ps in parents.items() if a!='B1'}
    tails={};retention={}
    for a in arms[1:]:
        tails[a]={};retention[a]={}
        for m in metrics:
            base=np.array([x['metrics'][m] for x in scores['B1']]);value=np.array([x['metrics'][m] for x in scores[a]])
            delta=(value-base)*100
            tails[a][m]=dict(harm_gt5pp=int((delta < -5).sum()),positive=int((delta>0).sum()),negative=int((delta<0).sum()),
                             zero=int((delta==0).sum()),worst_pp=float(delta.min()),best_pp=float(delta.max()))
            retention[a][m]=dict(eligible=int((base>.5).sum()),retained=int(((base>.5)&(value>.5)).sum()))
    tolerance=read(D/'CONFIG.json')['tie_tolerance_pp']
    def compare(a,b):
        delta=summary[a]['vIoU']-summary[b]['vIoU']
        if abs(delta)>tolerance:return -1 if delta>0 else 1
        ha=tails[a]['vIoU']['harm_gt5pp'];hb=tails[b]['vIoU']['harm_gt5pp']
        return (-1 if ha<hb else 1) if ha!=hb else (-1 if a<b else int(a>b))
    ranked=sorted(keys,key=cmp_to_key(compare));selected=ranked[:6] if phase=='dev16' else ranked[:1]
    factor={}
    if phase=='dev16':
        grid=seal['configurations']
        for name in ('steps','radius','temporal_weight'):
            levels=sorted({c[name] for c in grid.values()});marginals={}
            for level in levels:
                subset=[key for key,c in grid.items() if c[name]==level]
                marginals[str(level)]={m:float(np.mean([summary[k][m] for k in subset])) for m in metrics}
            span={m:max(v[m] for v in marginals.values())-min(v[m] for v in marginals.values()) for m in metrics}
            # Matched contexts, paired at the parent level, highest vs lowest setting.
            parent_diff={}
            for source in parents['B1']:
                lo=np.mean([parents[k][source]['vIoU'] for k,c in grid.items() if c[name]==levels[0]])
                hi=np.mean([parents[k][source]['vIoU'] for k,c in grid.items() if c[name]==levels[-1]])
                parent_diff[source]=(hi-lo)*100
            factor[name]=dict(marginal_parent_macro_percent=marginals,marginal_span_pp=span,
                              highest_minus_lowest_v_pp=float(np.mean(list(parent_diff.values()))),
                              matched_parent_deltas_pp=parent_diff)
    report=dict(phase=phase,summary_percent=summary,comparisons=diffs,query_tails=tails,B1_good_retention=retention,
                ranked=ranked,selected=selected,factorial_dev16=factor,rows=scores,
                scalar_tensor_max_abs=maxerr,CPU_seconds=time.monotonic()-start,
                scope='exposed source development; configuration selected by GT offline; descriptive unadjusted parent CIs; not fresh evaluation')
    write(out/'REPORT.json',report);write(out/'SELECTION.json',dict(selected=selected,ranked=ranked,primary='parent_macro_vIoU'))
    lines=[f'# DESTA {phase}: actual native results','', '| configuration | tIoU % | sIoU % | vIoU % | >5pp v harm |', '|---|---:|---:|---:|---:|']
    for a in arms[:2]+ranked:
        s=summary[a];harm=tails.get(a,{}).get('vIoU',{}).get('harm_gt5pp',0)
        lines.append(f'| {a} | {s["tIoU"]:.6f} | {s["sIoU"]:.6f} | {s["vIoU"]:.6f} | {harm} |')
    lines+=['', 'Selected: '+', '.join(selected),'',report['scope'], '', 'All failures, missing pseudo-support and harms are retained. GT-R16 is diagnostic.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'COMPLETE.json',dict(pins={str(out/n):sha(out/n) for n in ('REPORT.json','SELECTION.json','REPORT.md')}))
    print('SCORED',phase,summary,'SELECTED',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev16','dev64','ablations']);a=p.parse_args();score(a.phase)
