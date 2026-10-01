"""Reconstruct the locked screen/focus decisions without selecting new parameters."""
import json,math
from pathlib import Path
from scripts.prepare_tastvg_coordinate_tuning_v2 import refine

def validate(b,screen,anchor):
    read=lambda p:json.loads(p.read_text());trials=read(b/'TRIALS.json');sel=read(b/'SELECTION.json')
    assert 0<len(trials)<=100 and [r['number'] for r in trials]==list(range(len(trials)))
    assert all(t['state'] in ['COMPLETE','FAIL'] for t in trials)
    choose=lambda ts:min([t for t in ts if t['state']=='COMPLETE'],key=lambda t:(-t['objective'],t['number']))
    coords=['rho','steps','student_temperature','direction_count'];expected=[('screen_anchor',anchor)]
    for key in coords:
        expected += [(f'screen_{key}_{i:03}',{**anchor,key:v}) for i,v in enumerate(screen[key]) if v!=anchor[key]]
    actual=[t for t in trials if t['stage']=='screen'];assert len(actual)==len(expected)==24
    assert [(t['tag'],t['params']) for t in actual]==expected
    origin=actual[0];assert origin['state']=='COMPLETE';alloc=[]
    for i,key in enumerate(coords):
        best=choose([t for t in actual if t['tag']=='screen_anchor' or t['tag'].startswith('screen_'+key+'_')])
        if best['objective']>origin['objective']:alloc.append(dict(key=key,improvement=best['objective']-origin['objective'],winner=best,index=i))
    alloc=sorted(alloc,key=lambda x:(-x['improvement'],x['index']))[:2]
    assert read(b/'FOCUS_ALLOCATION.json')==dict(coordinates=alloc,anchor_tag=origin['tag'],anchor_objective=origin['objective'],confirmation_used=False)
    current=anchor;decisions=[]
    for j,item in enumerate(alloc):
        key=item['key'];stage=f'focus{j+1}'
        if key in ['rho','student_temperature']:
            lo,hi=(1e-5,2.) if key=='rho' else (.01,100.)
            vals=sorted(set([float(f'{10**(math.log10(lo)+(math.log10(hi)-math.log10(lo))*i/20):.14g}') for i in range(21)]+[current[key],item['winner']['params'][key]]))
        else:vals=sorted(set(([1,2,3,4,5,6,8,10] if key=='steps' else [1,2,3,4,6,8,12,16])+[current[key]]))
        assert read(b/f'{stage}_GRID.json')==dict(parameter=key,fixed=current,values=vals)
        tt=[t for t in trials if t['stage']==stage];coarse=[t for t in tt if '_coarse_' in t['tag']]
        assert [t['params'] for t in coarse]==[{**current,key:v} for v in vals]
        best=choose(coarse);fine=refine(vals,best['params'][key]) if key in ['rho','student_temperature'] else []
        assert read(b/f'{stage}_REFINEMENT.json')==dict(parameter=key,fixed=current,best_coarse=best['tag'],values=fine)
        assert [t['params'] for t in tt if '_fine_' in t['tag']]==[{**current,key:v} for v in fine]
        best=choose(tt);decision=read(b/f'{stage}_SELECTION.json')
        assert {k:v for k,v in decision.items() if k!='time'}==dict(parameter=key,params=best['params'],tag=best['tag'],objective=best['objective'],result_dir=best['result_dir'],confirmation_used=False)
        assert decision['time']<sel['time'];decisions.append(decision);current=best['params']
    assert set(t['stage'] for t in trials)=={'screen'}|{f'focus{i+1}' for i in range(len(alloc))}
    assert sel['params']==current and sel['decisions']==decisions and not sel['confirmation_used']
    for t in trials:
        if t['state']=='FAIL':assert t['objective'] is None
        else:assert math.isfinite(t['objective'])
        if t.get('reused_from'):
            prior=next(x for x in trials if x['tag']==t['reused_from']);assert prior['number']<t['number'] and prior['state']=='COMPLETE' and prior['params']==t['params'] and prior['objective']==t['objective']
    return trials,sel
