"""Additional locked readout, never alters the running audit's protocol/pins."""
import collections,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_oracle_mixer_gap import D,ARMS,SEEDS,load
from vg_tta.desta3d_v3_oracle_io import read,write,sha,check_pins
from vg_tta.desta3d_v3_gap_decision import choose_branch,quadrant
P=D.parent/'gap_followups_v1'

def report():
    import numpy as np,torch
    torch.set_num_threads(4);start=time.monotonic();out=P/'decision_readback_v1';assert not out.exists()
    check_pins(read(P/'LOCK.json')['pins']);check_pins(read(D/'LOCK.json')['pins'])
    assert read(D/'COMPLETE.json')['seal_sha']==sha(D/'PREDICTIONS_SEAL.json')
    check_pins({str(D/p):h for p,h in read(D/'PREDICTIONS_SEAL.json')['files'].items()})
    rdir=D/'independent_readback_v1';r=read(rdir/'REPORT.json');assert read(rdir/'ROOT_SUMMARY_CROSSCHECK.json')['status']=='passed'
    assert read(rdir/'COMPLETE.json')['report_sha']==sha(rdir/'REPORT.json') and read(D/'ROOT_ALL_RAW_READBACK.json')['status']=='passed'
    basis=load(D/'BASIS.pt').numpy().astype(np.float64);cases=read(rdir/'GEOMETRY_CASES.json')
    alignment=[];quad={a:collections.Counter() for a in SEEDS};quadrant_cases=[]
    for i,g in enumerate(cases):
        assert shutil.disk_usage(ROOT).free>8*2**30+512*2**20
        raw=load(D/'episodes'/f'{i:04}'/'RAW_DIRECTIONS.pt');field=raw['deltas']['oracle'].numpy().reshape(-1,2560)
        coef=np.empty((len(field),256),np.float32)
        for startrow in range(0,len(field),128):coef[startrow:startrow+128]=(field[startrow:startrow+128].astype(np.float64)@basis).astype(np.float32)
        ep=out/'coefficients'/f'{i:04}';ep.mkdir(parents=True);torch.save(torch.from_numpy(coef.reshape(*raw['stock_shape'][:-1],256)),ep/'ORACLE_COEFFICIENTS.pt')
        oracle_norm=float(np.linalg.norm(coef.astype(np.float64)));cosines={}
        for a in SEEDS:
            learned=raw['coefficients'][a].numpy().astype(np.float64)*g['mixer'][a]['scale']
            learned=learned.reshape(-1,256);denom=oracle_norm*float(np.linalg.norm(learned))
            c=float(np.sum(learned*coef.astype(np.float64))/denom) if denom else None;cosines[a]=c
            fc=g['cosine_to_oracle'][a]
            if c is None:assert fc is None
            else:assert abs(c-fc)<2e-6
            od=r['arms']['oracle']['rows'][i]['metrics']['vIoU']-r['arms']['B1']['rows'][i]['metrics']['vIoU']
            ld=r['arms'][a]['rows'][i]['metrics']['vIoU']-r['arms']['B1']['rows'][i]['metrics']['vIoU']
            q=quadrant(od,ld);quad[a][q]+=1;quadrant_cases.append(dict(index=i,seed=a,quadrant=q,oracle_delta_v_pp=100*od,learned_delta_v_pp=100*ld))
        alignment.append(dict(index=i,cosine=cosines,oracle_coefficient_sha=sha(ep/'ORACLE_COEFFICIENTS.pt')))
    features=[];table={}
    for a in SEEDS:
        s=r['geometry_groups']['all']['stats'][a]
        positive={};opos={}
        for b in ('event','spatial'):
            eligible=[g for g in cases if g['gradient_norms'][b]>0]
            positive[b]=sum(g['descent_dot'][b][a]>0 for g in eligible)/len(eligible) if eligible else None
            opos[b]=sum(g['descent_dot'][b]['oracle']>0 for g in eligible)/len(eligible) if eligible else None
        f=dict(cos_mean=s['cosine']['mean'],cos_median=s['cosine']['median'],learned_descent=positive,oracle_descent=opos,
            saturated_fraction=sum(g['norm_over_cap'][a]>=.99 for g in cases)/447,v_mean=r['comparisons'][a]['vIoU']['mean_delta_pp'],
            oracle_minus_learned_lower=r['oracle_vs_learned'][a]['vIoU']['bootstrap_ci95_pp'][0]);features.append(f)
        table[a]=dict(oracle_delta_tsv_pp={m:r['comparisons']['oracle'][m]['mean_delta_pp'] for m in ['tIoU','sIoU','vIoU']},
            cosine=s['cosine'],learned_T_dot=s['T_descent'],learned_S_dot=s['S_descent'],
            oracle_T_dot=r['geometry_groups']['all']['stats']['oracle']['T_descent'],oracle_S_dot=r['geometry_groups']['all']['stats']['oracle']['S_descent'],
            norm_over_cap=s['norm_over_cap'],B1_good_oracle_delta_v=r['geometry_groups']['vIoU_B1_good']['stats']['oracle']['delta_v_pp'],
            B1_good_learned_delta_v=r['geometry_groups']['vIoU_B1_good']['stats'][a]['delta_v_pp'],
            B1_other_oracle_delta_v=r['geometry_groups']['vIoU_B1_other']['stats']['oracle']['delta_v_pp'],B1_other_learned_delta_v=r['geometry_groups']['vIoU_B1_other']['stats'][a]['delta_v_pp'],
            quadrants=dict(quad[a]),screen_features=f)
    oracle=dict(v_mean=r['comparisons']['oracle']['vIoU']['mean_delta_pp'],v_lower=r['comparisons']['oracle']['vIoU']['bootstrap_ci95_pp'][0],
        t_mean=r['comparisons']['oracle']['tIoU']['mean_delta_pp'],s_mean=r['comparisons']['oracle']['sIoU']['mean_delta_pp'])
    decision=choose_branch(oracle,features,rescue=None)
    write(out/'DECISION_TABLE.json',dict(status='completed_CPU_readback',table=table,decision=decision,oracle_full_practical_gate=r['gate']['oracle'],
        protocol_sha=sha(ROOT/'protocols/desta3d_v3_gap_followups_v1.md'),no_new_experiment_started=True,causal_limit='diagnostic screens only, not causal proof',CPU_seconds=time.monotonic()-start))
    write(out/'QUADRANT_CASES.json',quadrant_cases);write(out/'COEFFICIENT_ALIGNMENT.json',alignment)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'SEAL.json',dict(files=files))
    write(out/'COMPLETE.json',dict(seal_sha=sha(out/'SEAL.json'),queries=447,DECISION=decision['DECISION']))
    lines=['# Oracle–Mixer decision table','','Exposed447 source diagnosis; all final predictions sealed and independently scored.','',
        '|Readout|Seed1|Seed2|','|---|---:|---:|']
    for name in ['oracle_delta_tsv_pp','cosine','learned_T_dot','learned_S_dot','oracle_T_dot','oracle_S_dot','norm_over_cap','B1_good_oracle_delta_v','B1_good_learned_delta_v','B1_other_oracle_delta_v','B1_other_learned_delta_v','quadrants']:
        lines.append('|'+name+'|'+'|'.join(json.dumps(table[a][name],ensure_ascii=False) for a in SEEDS)+'|')
    lines+=['','DECISION = '+decision['DECISION'],'Recommended single next branch = '+str(decision['recommendation']),decision['reason'],'','No A/B/C GPU experiment was started. C1/C2/C3 are unmeasured.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
