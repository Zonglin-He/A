"""Public v3 scalar and selection audit, without private tensors or labels."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_tastvg_paper48_public_v1 import read_rows,check_summary
from scripts.validate_tastvg_extended_selection_v3 import validate

def run(folder):
    b=Path(folder);read=lambda p:json.loads(p.read_text());design=read(b/'SEARCH_DESIGN.json');trials,sel=validate(b,design['screen'],design['anchor']);checks=cells=0
    tasks=sorted(set(b/t['result_dir'] for t in trials if t['state']=='COMPLETE'))+[d for d in (b/'confirmation').iterdir() if (d/'SCALARS.csv').exists()]
    for d in tasks:
        rows=read_rows(d/'SCALARS.csv');a=read(d/'AUDIT.json');s=read(d/'SUMMARY.json');score=read(d/'SCORE.json');req=read(d/'REQUEST.json');cfg=req['params'];n,total=(32,384) if req['split']=='search' else (16,192)
        assert len(rows)==a['state_links']==total and len({r['parent'] for r in rows})==n
        assert len({(r['parent'],r['order'],r['condition']) for r in rows})==total
        assert sum(r['updated'] for r in rows)==a['SGD_updates'] and sum(r['expert_scheduled'] for r in rows)==a['teacher_checks']
        c=a['compute'];assert c['spatial_provider_calls']==c['temporal_provider_calls']==a['teacher_checks']
        assert c['spatial_candidate_replays']==c['inner_steps']*(2*cfg['direction_count']+1)
        assert c['native_replays']==total+c['inner_steps']-a['teacher_checks']+c['spatial_candidate_replays']+2*c['backward_calls']
        assert a['SGD_updates']<=c['backward_calls']<=a['teacher_checks']*cfg['steps'];assert a['teacher_checks']<=c['inner_steps']<=a['teacher_checks']*cfg['steps']
        for r in rows:
            assert r['expert_scheduled']==(r['arrival']%4==0)
            if not r['expert_scheduled']:assert not r['updated'] and r['delta_m_tIoU']==0
            for k in ['m_tIoU','m_vIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:
                assert 0<=r['Frozen_'+k]<=1+1e-12 and 0<=r['Ours_'+k]<=1+1e-12;assert abs(r['Ours_'+k]-r['Frozen_'+k]-r['delta_'+k])<1e-12;checks+=2
        for group in ['clean','corruption']:
            for sub in ['all','nonexpert','expert']:checks+=check_summary([r for r in rows if (r['condition']=='clean')==(group=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],s[group][sub])
        assert score['objective']==s['corruption']['all']['metrics']['delta_m_vIoU']['mean'];cells+=len(rows)
    paired=read(b/'PAIRED_VS_ANCHOR.json');ar=read_rows(b/'search/screen_anchor/SCALARS.csv')
    def check_pair(a,z,expected):
        nonlocal checks
        for group in ['clean','corruption']:
            for sub in ['all','nonexpert','expert']:
                rr=[]
                for x,y in zip(a,z):
                    assert all(x[k]==y[k] for k in ['parent','order','condition','arrival','expert_scheduled','Frozen_m_vIoU'])
                    if (x['condition']=='clean')==(group=='clean') and (sub=='all' or x['expert_scheduled']==(sub=='expert')):rr.append(dict(parent=x['parent'],order=x['order'],delta_vs_anchor=y['Ours_m_vIoU']-x['Ours_m_vIoU']))
                checks+=check_summary(rr,expected[group][sub])
    for t in trials:
        if t['state']=='COMPLETE':check_pair(ar,read_rows(b/t['result_dir']/'SCALARS.csv'),paired[t['tag']])
        else:assert t['objective'] is None
    sr=b/'confirmation/selected'
    if (sr/'REUSE.json').exists():sr=b/read(sr/'REUSE.json')['source']
    if (sr/'SCALARS.csv').exists():check_pair(read_rows(b/'confirmation/anchor/SCALARS.csv'),read_rows(sr/'SCALARS.csv'),read(b/'CONFIRMATION_PAIRED.json'))
    return dict(status='pass',scheduled=len(trials),actual_arrivals=cells,scalar_checks=checks,selection_verified=True,scope='Anonymous outcomes/bootstrap, paired contrasts, work counts and locked selection; no GT or model execution')

if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2))
