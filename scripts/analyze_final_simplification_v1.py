"""Offline paired scoring and predeclared sequential decisions; never a teacher."""
import argparse
import collections
import copy
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_final_simplification_v1 import OUT,verify,dev_rows,episode_path,existing
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_decota_final_freeze_v1 import aggregate,METRICS


def label_payload(p):
    assert sha(p['labels'])==p['labels_sha256']
    return read(p['labels'])


def measurements(data,labels,names):
    rows=[]
    for x in data:
        gt=labels[x['key']];row=dict(key=x['key'],group=x['group'],cohort=x['cohort'],arms={})
        for name in names:
            pred=x['predictions'][name]
            row['arms'][name]=checked_score(pred['boxes'],gt,x['input']['frame_ids'],pred['indices'])[0]
        rows.append(row)
    return rows


def tables(rows,names,reference):
    out={}
    for cohort in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==cohort];groups=[r['group'] for r in rr]
        out[cohort]=dict(queries=len(rr),sources=len(set(groups)),arms={},deltas={})
        for name in names:
            out[cohort]['arms'][name]={m:aggregate([r['arms'][name][m] for r in rr],groups) for m in METRICS}
            out[cohort]['deltas'][name]={m:aggregate([None if r['arms'][name][m] is None or r['arms'][reference][m] is None
                else r['arms'][name][m]-r['arms'][reference][m] for r in rr],groups) for m in METRICS}
    return out


def gate(rows,candidate,parent,baseline, *, require_cost=False,costs=None):
    pair=tables(rows,[candidate,parent],parent);directions={};dvs=[]
    for c,v in pair.items():
        rr=[r for r in rows if r['cohort']==c]; dv=v['deltas'][candidate]['vIoU_corrected']['mean']
        assert dv is not None;dvs.append(dv)
        counts={}
        for name in [candidate,parent]:
            dd=[r['arms'][name]['vIoU_corrected']-r['arms'][baseline]['vIoU_corrected'] for r in rr
                if r['arms'][name]['vIoU_corrected'] is not None and r['arms'][baseline]['vIoU_corrected'] is not None]
            counts[name]={str(t):sum(z < -t/100 for z in dd) for t in (5,10)}
        directions[c]=dict(delta_v_pp=100*dv,tails=counts,
            pass_tail=counts[candidate]['5']<=counts[parent]['5']+1 and counts[candidate]['10']<=counts[parent]['10'],
            pass_direction=dv>=-.01)
    avg=float(np.mean(dvs));cost_pass=True;cost_summary=None
    if require_cost:
        cost_summary={}
        for name in [candidate,parent]:
            cost_summary[name]={k:float(np.mean([x[name][k] for x in costs]))
                                for k in ['measured_forwards','measured_fit_seconds']}
        cost_pass=(cost_summary[candidate]['measured_forwards'] <= .9*cost_summary[parent]['measured_forwards']
                   and cost_summary[candidate]['measured_fit_seconds'] < cost_summary[parent]['measured_fit_seconds'])
    passed=avg>=-.005 and all(v['pass_tail'] and v['pass_direction'] for v in directions.values()) and cost_pass
    return dict(passed=bool(passed),mean_delta_v_pp=avg*100,directions=directions,costs=cost_summary,
        cost_pass=bool(cost_pass),rule='PROTOCOL.md predeclared practical gate; not a significance test')


def markdown_table(t):
    lines=['| Cohort | Arm | source vIoU | source sIoU | source tIoU | query vIoU | Δv vs reference |',
           '|---|---|---:|---:|---:|---:|---:|']
    for c,section in t.items():
        for arm,x in section['arms'].items():
            vals=[100*x[m]['mean'] for m in ['vIoU_corrected','sIoU','tIoU']]
            lines.append(f'|{c}|{arm}|'+ '|'.join(f'{v:.3f}' for v in vals)+
                f"|{100*x['vIoU_corrected']['query_mean']:.3f}|{100*section['deltas'][arm]['vIoU_corrected']['mean']:+.3f}|")
    return '\n'.join(lines)


def space():
    p=verify();labels=label_payload(p);data=[];cost=[]
    for c in p['rows']:
        for row in dev_rows(p,c):
            path=episode_path('space',row);assert existing(path);x=load(path)
            preds={'Frozen':x['native']}
            for n,f in x['fits'].items():preds[n]=f['final']
            data.append({**x,'predictions':preds})
            cost.append({n:{k:f.get(k) for k in ['measured_forwards','measured_fit_seconds',
                'trial_forwards','accepted_updates','backtracking_trigger_steps','optimizer_steps','backwards',
                'numerical_failure','best_step']} for n,f in x['fits'].items()})
    rows=measurements(data,labels,['Frozen','A','B','C'])
    gates={n:gate(rows,n,'A','Frozen',require_cost=True,costs=cost) for n in ['B','C']}
    good=[n for n in ['B','C'] if gates[n]['passed']]
    chosen='A'
    if good:
        chosen=max(good,key=lambda n:gates[n]['mean_delta_v_pp'])
        if set(good)=={'B','C'} and abs(gates['B']['mean_delta_v_pp']-gates['C']['mean_delta_v_pp'])<=.1:chosen='B'
    selection='best';last=None;stability=None
    if chosen!='A':
        active=[x['fits'][chosen] for x in data if not x['fits'][chosen]['skipped']]
        stable=[not f.get('failure') and (f['path'][-1]['loss']-min(s['loss'] for s in f['path']))<=.01*max(abs(min(s['loss'] for s in f['path'])),1e-12) for f in active]
        stability=dict(active=len(active),stable=sum(stable),fraction=float(np.mean(stable)) if stable else 1.)
        if stability['fraction']>=.9 and not any(f.get('failure') for f in active):
            from vg_tta.final_simplification_v1 import use_last
            for x in data:x['predictions']['D']=use_last(x['fits'][chosen])['final']
            rows=measurements(data,labels,['Frozen','A','B','C','D'])
            last=gate(rows,'D',chosen,'Frozen')
            if last['passed']:selection='last'
    spec=dict(base_arm=chosen,backtracking=chosen=='A',selection=selection,
              lr_multiplier=.5 if chosen=='C' else 1.,steps=10,planned=4,gamma=0.,kappa=None,
              gates=gates,last_gate=last,stability=stability,development_only=True,
              sources=64,query_count=64,historical_exposure=True,full_pool_used_for_selection=False)
    write(OUT/'SPATIAL_SELECTION.json',spec)
    t=tables(rows,list(rows[0]['arms']),'A')
    write(OUT/'S1_RESULTS.json',dict(summary=t,rows=rows,selection=spec,costs=cost))
    text=('# S1 Spatial Optimization Subtraction\n\n'+markdown_table(t)+'\n\n'+
        f"Selected {chosen}, {selection}; only the prelocked development list was used.\n\n"+
        'Cost/tail gates and all exact trajectories are in S1_RESULTS.json. This is not an independent test.\n')
    _text(OUT/'S1_RESULTS.md',text)
    print('SPATIAL_LOCK',chosen,selection,gates,flush=True)


def _text(path,text):
    assert not path.exists(),path
    path.write_text(text)


def time_data(stage,p):
    rows=[]
    for c in p['rows']:
        for r in dev_rows(p,c):
            path=episode_path('time_'+stage,r);assert existing(path);rows.append(load(path))
    return rows


def temporal(stage):
    p=verify();labels=label_payload(p);data=time_data('AB',p)
    for x in data:
        x['predictions']={'Baseline':x['native'],**{n:q['parameter'] for n,q in x['points'].items()}}
    if stage=='AB':
        rows=measurements(data,labels,['Baseline','A','B']);g=gate(rows,'B','A','Baseline')
        write(OUT/'TEMPORAL_B_DECISION.json',dict(try_C=g['passed'],gate=g,development_only=True))
        if not g['passed']:
            write(OUT/'TEMPORAL_OPTIMIZER_SELECTION.json',dict(arm='A',stage='AB',
                config=dict(backtracking=True,selection='best',beta=.1,gamma=1.),gate=g))
        selected='B' if g['passed'] else 'A'
    else:
        # Append each earlier arm so every chosen-parent comparison is paired.
        stages=['C'] if stage=='C' else [s for s in ['C','D','E'] if s==stage or (OUT/('time_'+s)).exists()]
        for s in stages:
            by={x['key']:x for x in time_data(s,p)}
            for x in data:x['predictions'].update({n:z['parameter'] for n,z in by[x['key']]['points'].items()})
        names=list(data[0]['predictions']);rows=measurements(data,labels,names)
        if stage=='C':
            parent='B';g=gate(rows,'C',parent,'Baseline');selected='C' if g['passed'] else parent
            conf=dict(backtracking=False,selection='last' if selected=='C' else 'best',beta=.1,gamma=1.)
            write(OUT/'TEMPORAL_OPTIMIZER_SELECTION.json',dict(arm=selected,stage='C' if selected=='C' else 'AB',config=conf,gate=g))
        elif stage=='D':
            old=read(OUT/'TEMPORAL_OPTIMIZER_SELECTION.json');parent=old['arm'];g=gate(rows,'D',parent,'Baseline')
            selected='D' if g['passed'] else parent
            write(OUT/'TEMPORAL_KL_SELECTION.json',dict(arm=selected,stage='D' if g['passed'] else old['stage'],
                config={**old['config'],'beta':0. if g['passed'] else .1},gate=g))
        else:
            old=read(OUT/'TEMPORAL_KL_SELECTION.json');parent=old['arm'];g=gate(rows,'E',parent,'Baseline')
            selected='E' if g['passed'] else parent
            write(OUT/'TEMPORAL_SELECTION.json',dict(arm=selected,stage='E' if g['passed'] else old['stage'],
                config={**old['config'],'gamma':0. if g['passed'] else 1.},gate=g,eta=.25,
                development_only=True,full_pool_used_for_selection=False))
    t=tables(rows,list(rows[0]['arms']),'Baseline')
    write(OUT/f'T1_{stage}_RESULTS.json',dict(rows=rows,summary=t,gate=g,selected=selected))
    _text(OUT/f'T1_{stage}_RESULTS.md','# T1 '+stage+'\n\n'+markdown_table(t)+f'\n\nSelected: {selected}. Gate: {g["passed"]}.\n')
    print('TEMPORAL',stage,selected,g,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['space','time']);ap.add_argument('--stage',default='AB');a=ap.parse_args()
    if a.action=='space':space()
    else:temporal(a.stage)


if __name__=='__main__':main()
