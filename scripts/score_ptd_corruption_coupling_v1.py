"""P3 post-seal paired parent statistics; oracle training and evaluation clearly separated."""
import collections,json,sys,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.ptd_corruption_coupling_v1 import OUT,rows,path,checked,read,write,sha,verify,CONDS,ARMS,SEEDS,TAU
from scripts.ptd_dino_bridge_v1 import OLD,geometry,support
from scripts.score_ptd_spatial_adapter_ab_v1 import truth,evaluate

def stats(values):
    a=np.asarray(values,dtype=float);assert len(a) and np.isfinite(a).all()
    rng=np.random.default_rng(20260926);boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    sort=np.sort(a);k=int(.05*len(a));loo=(a.sum()-a)/(len(a)-1) if len(a)>1 else a
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),median=float(np.median(a)),
        trimmed5=float(sort[k:len(a)-k].mean()),minimum=float(a.min()),maximum=float(a.max()),
        positive=int((a>1e-12).sum()),negative=int((a< -1e-12).sum()),zero=int((abs(a)<=1e-12).sum()),
        severe_loss_gt5pp=int((a<-.05).sum()),leave_one_out=[float(loo.min()),float(loo.max())])

def metric(z,tokens,r,g):
    e=evaluate(z,tokens,r,g);e['s_intersection']=e['s'];e['s']=float(np.asarray(e['ious'])[g['valid']].mean()) if g['valid'].any() else None
    return e

def run():
    reg=verify();torch.set_num_threads(4);start=time.monotonic()
    bar=read(OUT/'REPLAY_DEVELOPMENT_BARRIER.json');assert len(bar['files'])==64*3*3*3
    for p,h in bar['files'].items():assert sha(p)==h
    # Every prediction endpoint and fixed design exists before final readout.
    files={str(p):sha(p) for folder in ['capture','design','oracle','fits','replay','time_fit'] for p in (OUT/folder).rglob('*.pt')}
    barrier=OUT/'FINAL_PREDICTION_BARRIER.json'
    if barrier.exists():assert read(barrier)['files']==files,'predictions changed after seal'
    else:write(barrier,dict(time=time.time(),files=files,oracle_training_GT=True,final_metric_readout_begins_after_this=True))
    labels=read(OLD/'LABELS_SCORER_ONLY.json');records=[];transitions=[];support_rows=[];seed_rows=[];steps=[]
    for r in rows('development'):
        g=truth(r,labels)
        for c in CONDS:
            data={s:checked(path(r,c,'capture',s)) for s in ['F','T']};z={s:data[s]['z'] for s in data}
            native={s:metric(z[s],z[s].get('base_tokens',[]),r,g) for s in z};out={**native};oracles={};designs={}
            fs=np.asarray(native['F']['ious']);good=g['valid']&(fs>=.5)
            losses={s:int((good&(np.asarray(native[s]['ious'])<.5)).sum()) for s in native}
            for arm,state in ARMS.items():
                m=[];o=checked(path(r,c,'oracle',arm));oracles[arm]=o
                ds=checked(path(r,c,'design',state if arm!='TS-stale' else 'TS-stale'));designs[arm]=ds
                for seed in SEEDS:
                    fit=checked(path(r,c,'fits',arm,seed));rp=checked(path(r,c,'replay',arm,seed))
                    result=metric(z[state],rp['tokens'],r,g);m.append(result)
                    seed_rows.append(dict(key=r['key'],domain=r['domain'],condition=c,arm=arm,seed=seed,v=result['v'],dv=result['v']-native['F']['v'],native_good_loss=int((good&(np.asarray(result['ious'])<.5)).sum())))
                    for h in fit['history']:
                        ev=metric(z[state],h['tokens'],r,g);steps.append(dict(key=r['key'],condition=c,arm=arm,seed=seed,step=h['step'],v=ev['v'],loss=h['loss']))
                out[arm]={k:float(np.mean([x[k] for x in m])) if all(x[k] is not None for x in m) else None for k in ['v','t','s','s_intersection']}
                out[arm]['ious']=np.mean([x['ious'] for x in m],axis=0).tolist()
                losses[arm]=float(np.mean([int((good&(np.asarray(x['ious'])<.5)).sum()) for x in m]))
                out['O_'+arm]=metric(z[state],o['oracle_output'],r,g)
            a={k:{q:v[q] for q in ['v','t','s','s_intersection']} for k,v in out.items()}
            rec=dict(key=r['key'],source=r['source'],domain=r['domain'],condition=c,arms=a,
                native_good_frames=int(good.sum()),native_good_losses=losses,
                format_F=bool(z['F']['format_ok']),format_T=bool(z['T']['format_ok']),
                I0=z['F'].get('interval'),IT=z['T'].get('interval'),
                high_F=int((data['F']['Dctrl']>=TAU).sum()),high_T=int((data['T']['Dctrl']>=TAU).sum()),
                nF=len(z['F'].get('positions',[])),nT=len(z['T'].get('positions',[])),
                stale_missing=len(designs['TS-stale'].get('dropped_positions',[])),
                oracle_F=a['O_S']['v']-a['F']['v'],oracle_T=a['O_TS']['v']-a['T']['v'],
                absorb_F=a['S']['v']-a['F']['v'],absorb_T=a['TS']['v']-a['T']['v'],
                interaction=a['TS']['v']-a['T']['v']-a['S']['v']+a['F']['v'],
                ious={arm:v['ious'] for arm,v in out.items()})
            records.append(rec)
            pos0=z['F'].get('positions',[]);post=z['T'].get('positions',[])
            support_rows.append(dict(key=r['key'],domain=r['domain'],condition=c,F=len(pos0),T=len(post),common=len(set(pos0)&set(post)),entered=sorted(set(post)-set(pos0)),left=sorted(set(pos0)-set(post)),legal_GT=int(g['valid'].sum())))
            ob={b['j']:b for b in oracles['TS']['blocks']}
            for pos in sorted(set(pos0)&set(post)):
                j,k=pos0.index(pos),post.index(pos);d0,dt=float(data['F']['Dctrl'][j]),float(data['T']['Dctrl'][k]);cat=('H' if d0>=TAU else 'L')+('H' if dt>=TAU else 'L')
                valid=bool(g['valid'][pos]);b=ob.get(k)
                transitions.append(dict(key=r['key'],domain=r['domain'],condition=c,position=pos,frame_id=r['input']['frame_ids'][pos],category=cat,D0=d0,DT=dt,valid=valid,
                    s_T_minus_F=out['T']['ious'][pos]-out['F']['ious'][pos] if valid else None,
                    s_TS_minus_T=out['TS']['ious'][pos]-out['T']['ious'][pos] if valid else None,
                    s_TS_minus_F=out['TS']['ious'][pos]-out['F']['ious'][pos] if valid else None,
                    oracle_credit=max(0.,float(b['credit'].max())) if b is not None else 0.,
                    absorption_credit=(out['TS']['ious'][pos]-out['T']['ious'][pos])/b['denominator'] if valid and b is not None else None))
    write(OUT/'SOURCE_RESULTS.json',records);write(OUT/'TRANSITIONS.json',transitions);write(OUT/'SUPPORT.json',support_rows)
    write(OUT/'SEED_RESULTS.json',seed_rows);write(OUT/'STEP_RESULTS.json',steps)
    by={(r['key'],r['condition']):r for r in records};schedule={x['key']:x['condition'] for x in reg['changing']};summary={};parent_tables=[]
    for domain in ['all','HC','Vid']:
        rr=[r for r in rows('development') if domain=='all' or r['domain']==domain]
        summary[domain]={}
        for condition in CONDS+['corr','changing']:
            table=[]
            for r in rr:
                cc=['noise_medium','defocus_extreme'] if condition=='corr' else [schedule[r['key']]] if condition=='changing' else [condition]
                vals=[by[(r['key'],c)] for c in cc]
                v=dict(key=r['key'],domain=r['domain'],condition=condition,arms={})
                for arm in ['F','T','S','TS','TS-stale','O_S','O_TS','O_TS-stale']:
                    v['arms'][arm]={metric:float(np.mean([x['arms'][arm][metric] for x in vals])) if all(x['arms'][arm][metric] is not None for x in vals) else None for metric in ['v','t','s','s_intersection']}
                for field in ['oracle_F','oracle_T','absorb_F','absorb_T','interaction','high_F','high_T','nF','nT','stale_missing','native_good_frames']:
                    v[field]=float(np.mean([x[field] for x in vals]))
                v['native_good_losses']={arm:float(np.mean([x['native_good_losses'][arm] for x in vals])) for arm in ['F','T','S','TS','TS-stale']}
                table.append(v)
            q={}
            for arm in ['F','T','S','TS','TS-stale']:
                q[arm]={}
                for metric_name in ['v','t','s']:
                    vv=[x for x in table if x['arms'][arm][metric_name] is not None and x['arms']['F'][metric_name] is not None]
                    q[arm][metric_name]=stats([x['arms'][arm][metric_name] for x in vv]) if vv else None
                    q[arm]['d'+metric_name]=stats([x['arms'][arm][metric_name]-x['arms']['F'][metric_name] for x in vv]) if vv else None
                q[arm]['native_good_losses']=sum(x['native_good_losses'][arm] for x in table)
            for name,a,b in [('TS_minus_T','TS','T'),('TS_minus_S','TS','S'),('TS_minus_stale','TS','TS-stale')]:q[name]=stats([x['arms'][a]['v']-x['arms'][b]['v'] for x in table])
            for field in ['oracle_F','oracle_T','absorb_F','absorb_T','interaction','high_F','high_T','nF','nT','stale_missing']:q[field]=stats([x[field] for x in table])
            q['delta_oracle']=stats([x['oracle_T']-x['oracle_F'] for x in table])
            q['absorption_ratio_F']=q['absorb_F']['mean']/q['oracle_F']['mean'] if q['oracle_F']['mean']>0 else None
            q['absorption_ratio_T']=q['absorb_T']['mean']/q['oracle_T']['mean'] if q['oracle_T']['mean']>0 else None
            drop=np.mean([by[(x['key'],'clean')]['arms']['F']['v']-x['arms']['F']['v'] for x in table]);q['drop_from_clean']=float(drop);q['recovery']=q['TS']['dv']['mean']/drop if drop>0 else None
            summary[domain][condition]=q
            if domain=='all':parent_tables.extend(table)
    primary=summary['all']['corr'];clean=summary['all']['clean']
    gateA=primary['TS']['dv']['ci95'][0]>0
    gateB=any(primary['TS_minus_'+arm]['mean']>0 and primary['TS']['dv']['severe_loss_gt5pp']<=primary[arm]['dv']['severe_loss_gt5pp'] for arm in ['T','S'])
    gateC=clean['TS']['dv']['mean']>-.005 and clean['TS']['dv']['severe_loss_gt5pp']<=min(clean['T']['dv']['severe_loss_gt5pp'],clean['S']['dv']['severe_loss_gt5pp'])
    write(OUT/'SUMMARY.json',dict(groups=summary,primary='source-wise mean noise_medium and defocus_extreme; then parent macro',spatial_support='all legal GT sampled positions, absent outputs zero',bootstrap=10000,exposed_development=True))
    write(OUT/'PARENT_TABLES.json',parent_tables)
    write(OUT/'DECISION.json',dict(gate_A=gateA,practical_gt1pp=primary['TS']['dv']['mean']>.01,gate_B=gateB,gate_C=gateC,all_gates=gateA and gateB and gateC,critic_started=False,temporal_interface_new=True,production_promoted=False))
    transition_summary={}
    for c in CONDS:
        transition_summary[c]={}
        for cat in ['LL','HL','LH','HH']:
            ts=[t for t in transitions if t['condition']==c and t['category']==cat];ks=sorted(set(t['key'] for t in ts));valid=[t for t in ts if t['valid']]
            q=dict(blocks=len(ts),sources=len(ks),legal_blocks=len(valid),legal_sources=len(set(t['key'] for t in valid)))
            for field in ['s_T_minus_F','s_TS_minus_T','s_TS_minus_F','oracle_credit','absorption_credit']:
                values=[np.mean([t[field] for t in valid if t['key']==k and t[field] is not None]) for k in ks if any(t['key']==k and t[field] is not None for t in valid)]
                q[field]=stats(values) if values else None
            transition_summary[c][cat]=q
    write(OUT/'TRANSITION_SUMMARY.json',transition_summary)
    categories=collections.defaultdict(list)
    for x in records:
        if x['condition']=='clean':continue
        a=x['arms'];tr=[t for t in transitions if t['key']==x['key'] and t['condition']==x['condition']];has=lambda cat:any(t['category']==cat for t in tr)
        criteria={'T_rescue':a['T']['v']-a['F']['v']>.02 and has('HL'),
            'T_induced_spatial':a['T']['t']>a['F']['t'] and ((a['T']['s'] is not None and a['T']['s']<a['F']['s']) or has('LH')),
            'S_rescue_after_T':a['TS']['v']-a['T']['v']>.02,
            'negative_interaction':a['T']['v']>a['F']['v'] and a['S']['v']>a['F']['v'] and a['TS']['v']<max(a['T']['v'],a['S']['v']),
            'oracle_rich_absorption_poor':x['oracle_T']>.02 and x['absorb_T']<=0,
            'candidate_poor':x['high_T']>0 and x['oracle_T']<=1e-12}
        for cat,yes in criteria.items():
            if yes:categories[cat].append(dict(key=x['key'],condition=x['condition'],domain=x['domain'],oracle_T=x['oracle_T'],absorb_T=x['absorb_T'],interaction=x['interaction'],dT=a['T']['v']-a['F']['v']))
    cases={}
    for cat in ['T_rescue','T_induced_spatial','S_rescue_after_T','negative_interaction','oracle_rich_absorption_poor','candidate_poor']:
        vv=sorted(categories[cat],key=lambda x:(-abs(x['interaction'])-x['oracle_T']-abs(x['dT']),x['key'],x['condition']));chosen=[]
        for v in vv:
            if v['key'] not in {x['key'] for x in chosen}:chosen.append(v)
            if len(chosen)==3:break
        cases[cat]=dict(available=len(vv),selected=chosen)
    write(OUT/'CASE_SELECTION.json',dict(categories=cases,selection_after_seal=True,oracle_rich_threshold=.02,candidate_poor_tolerance=1e-12))
    with open(OUT/'SOURCE_RESULTS.csv','w') as f:
        fields=['key','domain','condition']+[a+'_'+m for a in ['F','T','S','TS','TS-stale'] for m in ['v','t','s']]+['interaction','oracle_F','oracle_T','absorb_F','absorb_T','high_F','high_T']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in records:w.writerow({**{k:x[k] for k in fields if k in x},**{a+'_'+m:x['arms'][a][m] for a in ['F','T','S','TS','TS-stale'] for m in ['v','t','s']}})
    write(OUT/'SCORING_COMPLETE.json',dict(time=time.time(),sources=64,condition_rows=len(records),seed_rows=len(seed_rows),states=len(steps),seconds=time.monotonic()-start))
    print(json.dumps({'decision':read(OUT/'DECISION.json'),'primary':{k:primary[k] for k in ['TS_minus_T','TS_minus_S','TS_minus_stale','interaction']},'TS':primary['TS']['dv']},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
