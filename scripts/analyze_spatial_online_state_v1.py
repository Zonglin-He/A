"""Offline calibration and evaluation; never imported by the online adapter."""
import argparse, collections, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_spatial_online_state_v1 import OUT,verify,dest
from scripts.analyze_spatial_reference_absorption_v1 import score
METRICS=['sIoU','vIoU_corrected','tIoU','unobserved_sIoU']


def scored(x,gt):
    return {a:score(x[a],gt,x['frame_ids'],x['indices'],x['observed'],x['anchors'])
            for a in ['native','before','after']}


def calibration():
    p=verify();bar=read(OUT/'CALIBRATION_BARRIER.json')
    assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[]
    for f,h in bar['files'].items():
        assert sha(f)==h;x=load(f);assert x['position']<8
        gt=labels[x['key']]
        rows.append({k:x[k] for k in ['cohort','arm','multiplier','seed','position','key']} |
                    {'metrics':scored(x,gt)})
    chosen={};values={}
    for c in p['streams']:
        chosen[c]={};values[c]={}
        for a in p['arms']:
            values[c][a]={str(m):float(np.mean([r['metrics']['after']['vIoU_corrected'] for r in rows
                if r['cohort']==c and r['arm']==a and r['multiplier']==m])) for m in p['multipliers']}
            v=values[c][a];best=max(v.values())
            chosen[c][a]=next(m for m in p['multipliers'] if best-v[str(m)]<=1e-12)
    write(OUT/'CALIBRATION.json',dict(selected=chosen,values=values,rows=rows,
        calibration_sources_per_direction=8,evaluation_GT_used=False,barrier_sha256=sha(OUT/'CALIBRATION_BARRIER.json')))
    print('SELECTED',chosen,flush=True)


def stat(vals):
    a=np.asarray(vals,float);rng=np.random.default_rng(20260920)
    boots=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),median=float(np.median(a)),ci95_conditional=np.quantile(boots,[.025,.975]).tolist(),
        n=len(a),harm5=float(np.mean(a<-.05)),harm10=float(np.mean(a<-.1)))


def evaluation():
    p=verify();labels=read(p['labels']);chosen=read(OUT/'CALIBRATION.json')['selected'];rows=[];audits=collections.Counter()
    for phase in ['calibration','evaluation']:
        bar=read(OUT/(phase.upper()+'_BARRIER.json'))
        for f,h in bar['files'].items():
            assert sha(f)==h;x=load(f)
            if x['position']:
                prev=dest(x['cohort'],x['seed'],x['arm'],x['multiplier'],x['position']-1)
                assert sha(prev)==x['previous_file_sha256']
                from vg_tta.spatial_online_state_v1 import arrival
                from methods.decota_final_simplified_v1.tensors import state_hash
                source=load(dest(x['cohort'],x['seed'],x['arm'],x['multiplier'],0))['initial']
                assert state_hash(arrival(source,load(prev)['state'],x['arm']))==x['initial_hash']
                audits['state_chain']+=1
            audits.update(k for k,v in x['audit'].items() if v)
            assert x['selected_step']==min(range(len(x['losses'])),key=lambda i:x['losses'][i]) or x['failure']
            row={k:x[k] for k in ['cohort','arm','multiplier','seed','position','phase','source','key','selected_step',
                 'parameter_changed','skipped','backwards','seconds','failure']}
            row['metrics']=scored(x,labels[x['key']]);row['anchor_count']=len(x['anchors'])
            row['query_norm']=float(x['state']['spatial.query_residual'].norm())
            rows.append(row)
    summary={}
    for panel in ['original32','selected24','original24']:
        summary[panel]={}
        for c in p['streams']:
            ss={};selected_rows=[]
            for a in p['arms']:
                rr=[r for r in rows if r['cohort']==c and r['arm']==a and
                    (panel=='original32' or r['position']>=8) and
                    r['multiplier']==(chosen[c][a] if panel=='selected24' else 1.)]
                selected_rows+=rr
                sources=sorted({r['source'] for r in rr});assert len(rr)==3*len(sources)
                aa={stage:{m:stat([np.mean([r['metrics'][stage][m] for r in rr if r['source']==s])
                    for s in sources]) for m in METRICS} for stage in ['native','before','after']}
                aa['contrasts']={}
                for new,old in [('before','native'),('after','before'),('after','native')]:
                    aa['contrasts'][new+'-'+old]={m:stat([np.mean([r['metrics'][new][m]-r['metrics'][old][m]
                        for r in rr if r['source']==s]) for s in sources]) for m in METRICS}
                aa['orders']={str(seed):{stage:{m:float(np.mean([r['metrics'][stage][m] for r in rr if r['seed']==seed]))
                    for m in METRICS} for stage in ['native','before','after']} for seed in [20260920,20260921,20260922]}
                aa['position_bins']={str(b):{stage:{m:float(np.mean([r['metrics'][stage][m] for r in rr if r['position']//8==b]))
                    for m in METRICS} for stage in ['native','before','after']} for b in sorted({r['position']//8 for r in rr})}
                aa['trajectory_harm']={stage:{str(th):float(np.mean([r['metrics'][stage]['vIoU_corrected']-
                    r['metrics']['native']['vIoU_corrected'] < -th for r in rr])) for th in [.05,.1]}
                    for stage in ['before','after']}
                aa['changed']=sum(r['parameter_changed'] for r in rr);aa['skipped']=sum(r['skipped'] for r in rr)
                aa['failure_count']=sum(bool(r['failure']) for r in rr)
                ss[a]=aa
            for a in ['O-all','O-split']:
                rr=[r for r in selected_rows if r['arm']==a];er={(r['source'],r['seed']):r for r in selected_rows if r['arm']=='E'}
                ss[a]['after-E']={m:stat([np.mean([r['metrics']['after'][m]-er[(s,r['seed'])]['metrics']['after'][m]
                    for r in rr if r['source']==s]) for s in sorted({r['source'] for r in rr})]) for m in METRICS}
            summary[panel][c]=ss
    write(OUT/'ALL_SOURCE_RESULTS.json',rows);write(OUT/'SUMMARY.json',summary)
    write(OUT/'AUDIT.json',dict(status='passed',receipt_count=len(rows),checks=dict(audits),
        production_pins=len(p['production_pins']),GT_online=False,new_DINO_calls=0,
        conditional_CI_only=True,independent_sources=64,evaluation_sources=48,
        all_failures=sum(bool(r['failure']) for r in rows),backwards=sum(r['backwards'] for r in rows)))
    for panel,cc in summary.items():
        for c,ss in cc.items():
            print(panel,c,{a:{stage:round(v[stage]['vIoU_corrected']['mean']*100,3) for stage in ['native','before','after']}
                for a,v in ss.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['calibration','evaluation']);a=ap.parse_args()
    calibration() if a.phase=='calibration' else evaluation()
