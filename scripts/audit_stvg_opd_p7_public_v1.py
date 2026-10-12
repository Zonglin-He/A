"""Public anonymous P7 statistics/contrasts audit without model or GT access."""
import collections,gzip,json,math,sys
from pathlib import Path
import numpy as np

def read(p):
    if p.exists():return json.loads(p.read_text())
    with gzip.open(p.with_suffix(p.suffix+'.gz'),'rt') as f:return json.load(f)
def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f]
def near(a,b):
    if isinstance(a,dict):assert set(a)==set(b);return sum(near(v,b[k]) for k,v in a.items())
    if isinstance(a,list):assert len(a)==len(b);return sum(near(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<2e-10*(1+abs(a))
    return 1

def run(root,phase):
    p=Path(root)/'results/stvg_opd_temporal_guided_candidate/2026-10-12'/('P7_'+phase);rr=rows(p/'ROWS.jsonl.gz');s=read(p/'SUMMARY.json');assert len(rr)==512 and len({r['parent_id'] for r in rr})==256
    checks=0
    for ds,data in s.items():
        for subset,blocks in data.items():
            views=([('both',blocks['paired_both_orders'])]+list(blocks['orders'].items())) if phase=='A' else list(blocks.items())
            for order,b in views:
                chosen=[r for r in rr if r['dataset']==ds and (subset=='all128' or r['previous_P6']==(subset=='previous_P6_32')) and (order=='both' or r['order']==order)];parents=collections.defaultdict(list)
                for r in chosen:parents[r['parent_id']].append(r)
                assert len(parents)==b['parents']=={'all128':128,'previous_P6_32':32,'added96':96}[subset]
                for field,v in b['statistics'].items():
                    vals=[math.fsum(r[field] for r in rs if r[field] is not None)/sum(r[field] is not None for r in rs) for rs in parents.values() if any(r[field] is not None for r in rs)]
                    if not vals:
                        checks+=near(dict(parents=0,mean=None,ci95=None,undefined_parents=len(parents)),v);continue
                    x=np.array(vals);gen=np.random.default_rng(20261012);idx=gen.integers(len(x),size=(10000,len(x)));boot=x[idx].sum(1)/len(x)
                    expected=dict(mean=math.fsum(x)/len(x),ci95=np.quantile(boot,[.025,.975]).tolist(),parent_values=x.tolist(),
                        harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)),positive_parents=int(sum(x>1e-12)),negative_parents=int(sum(x< -1e-12)))
                    if phase=='A':expected.update(min=float(min(x)),max=float(max(x)))
                    else:expected.update(parents=len(x),undefined_parents=len(parents)-len(x))
                    checks+=near(expected,v)
    for r in rr:
        if phase=='A':
            for arm in ['Native','OPD','Direct','Capped_Direct']:
                for m in ['v','t','s','R30','R50']:checks+=near(r[f'{arm}_T1_{m}']-r[f'{arm}_T0_{m}'],r[f'delta_T_{arm}_{m}'])
                assert r[f'{arm}_T0_s']==r[f'{arm}_T1_s']
            for m in ['v','t','s','R30','R50']:checks+=near(r[f'delta_T_OPD_{m}']-r[f'delta_T_Native_{m}'],r[f'interaction_OPD_Native_{m}'])
        else:
            for name,left,right in [('coupling_C_minus_B','C_temporal_guided_OPD','B_posthoc'),('D_minus_B','D_temporal_guided_Direct','B_posthoc'),('C_minus_D','C_temporal_guided_OPD','D_temporal_guided_Direct'),('B_minus_A','B_posthoc','A_original')]:
                for m in ['v','t','s','R30','R50']:checks+=near(r[f'{left}_{m}']-r[f'{right}_{m}'],r[f'{name}_{m}'])
    if phase=='B':
        aa={r['public_cell']:r for r in rows(p.parent/'P7_A/ROWS.jsonl.gz')}
        for extra in rows(p/'DIAGNOSTIC_EXTRA_ROWS.jsonl.gz'):
            r=next(r for r in rr if r['public_cell']==extra['public_cell']);assert all(r[k]==extra[k] for k in ['dataset','parent_id','order','previous_P6'])
            for arm,prefix in [('B_posthoc','B'),('C_temporal_guided_OPD','C'),('D_temporal_guided_Direct','D')]:
                for m in ['v','t','s']:checks+=near(r[arm+'_'+m]-extra['Full_Before_T1_'+m],extra[prefix+'_minus_Full_Before_'+m])
            r.update({k:v for k,v in extra.items() if k not in ['public_cell','dataset','parent_id','order','previous_P6']})
        predicates={
            'all_added96':lambda r:True,
            'gained_annotated_event_support':lambda r:aa[r['public_cell']]['gained_event_frames']>0,
            'lost_annotated_event_support':lambda r:aa[r['public_cell']]['lost_event_frames']>0,
            'increased_event_outside_duration':lambda r:aa[r['public_cell']]['T1_outside_event_duration']>aa[r['public_cell']]['T0_outside_event_duration'],
            'new_observation_outside_event':lambda r:r['I1_visited_outside_event']>0,
            'posthoc_temporal_harm_over5pp':lambda r:aa[r['public_cell']]['delta_T_OPD_v']<-.05,
            'coupling_harm_over5pp':lambda r:r['coupling_C_minus_B_v']<-.05}
        gd=read(p/'DIAGNOSTIC_GROUPS.json');assert gd['GT_read'] and gd['groups_overlap']
        for ds,groups in gd['groups'].items():
            population=[r for r in rr if r['dataset']==ds and not r['previous_P6']]
            for name,g in groups.items():
                chosen=[r for r in population if predicates[name](r)];by=collections.defaultdict(list)
                for r in chosen:by[r['parent_id']].append(r)
                assert len(chosen)==g['cells'] and len(by)==g['distinct_parents']
                for f,v in g['statistics'].items():
                    values=[math.fsum(r[f] for r in rs if r[f] is not None)/sum(r[f] is not None for r in rs) for rs in by.values() if any(r[f] is not None for r in rs)]
                    if not values:checks+=near(dict(parents=0,mean=None,ci95=None,undefined_parents=len(by)),v);continue
                    x=np.array(values);gen=np.random.default_rng(20261012);idx=gen.integers(len(x),size=(10000,len(x)))
                    expected=dict(parents=len(x),mean=math.fsum(x)/len(x),ci95=np.quantile(x[idx].mean(1),[.025,.975]).tolist(),
                        undefined_parents=len(by)-len(x),parent_values=x.tolist(),harm_5pp=int(sum(x<-.05)),harm_20pp=int(sum(x<-.2)))
                    checks+=near(expected,v)
        for z in read(p/'CASE_SELECTION_PUBLIC.json')['cases']:
            population=[r for r in rr if r['dataset']==z['dataset'] and not r['previous_P6']];by=collections.defaultdict(list)
            for r in population:by[r['parent_id']].append(r)
            order=sorted(by,key=lambda k:math.fsum(r['coupling_C_minus_B_v'] for r in by[k])/len(by[k]))
            assert z['parent_id']==order[0 if z['label']=='worst_coupling_parent' else -1]
            assert z['public_cell']==next(r['public_cell'] for r in by[z['parent_id']] if r['order']=='order1')
            checks+=near(z['paired_parent_coupling'],math.fsum(r['coupling_C_minus_B_v'] for r in by[z['parent_id']])/len(by[z['parent_id']]))
    return dict(status='pass',phase='P7_'+phase,cells=512,parents=256,scalar_checks=checks,scope='anonymous all128/previous32/added96/allorders/complete contrasts and bootstrap',CPU_only=True,GT_read=False)

if __name__=='__main__':print(json.dumps(run(*sys.argv[1:]),allow_nan=False))
