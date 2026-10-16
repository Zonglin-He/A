"""Offline scores on sealed predictions, with local and unobserved denominators."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_spatial_reference_absorption_v1 import OUT,verify
from scripts.analyze_spatial10_components_v1 import checked_score,summary
from vg_tta.box_stability_diagnostics_v1 import overlap
METRICS=['sIoU','vIoU_corrected','tIoU','unobserved_sIoU','reference_GT_IoU','expert_IoU','L1_cx','L1_cy','L1_w','L1_h']


def mean(x):return float(np.mean(x)) if len(x) else None


def score(boxes,gt,ids,indices,observed,anchors):
    metric,q=checked_score(boxes,gt,ids,indices);valid=np.array(gt['valid'],bool);unseen=valid.copy();unseen[observed]=False
    pos=[a['position'] for a in anchors];target=np.array([a['box'] for a in anchors]).reshape(-1,4);pred=np.asarray(boxes)[pos]
    metric['unobserved_sIoU']=mean(np.array(q)[unseen]);metric['reference_GT_IoU']=mean([q[i] for i in pos if valid[i]])
    metric['expert_IoU']=mean(overlap(pred,target)) if pos else None
    for i,name in enumerate(['cx','cy','w','h']):metric['L1_'+name]=mean(np.abs(pred[:,i]-target[:,i]))
    return metric


def run(phase):
    p=verify();b=read(OUT/f'{phase.upper()}_BARRIER.json');assert not b['GT_online']
    for f,h in b['files'].items():assert sha(f)==h
    assert sha(p['labels'])==p['labels_sha256'];labels=read(p['labels']);rows=[]
    if phase=='probe':
        for f in b['files']:
            x=load(f);gt=labels[x['key']];states=[]
            for z in x['probe']['states']:
                states.append(dict(alpha=z['alpha'],details=z['details'],metrics=score(z['boxes'],gt,x['frame_ids'],x['indices'],x['actual_observations'],[x['anchor']])))
            rows.append(dict(key=x['key'],cohort=x['cohort'],source=x['source'],anchor=x['anchor'],states=states))
        write(OUT/'PROBE_RESULTS.json',rows)
        z=next(r for r in rows if r['key']=='vidstg_test:007868')
        print('Vid7868: alpha loss cx expertIoU localGT unobservedGT')
        for s in z['states']:print(s['alpha'],s['details']['loss'],s['details']['anchor_boxes'][0][0],s['metrics']['expert_IoU'],s['metrics']['reference_GT_IoU'],s['metrics']['unobserved_sIoU'])
        return
    allstats={};costs=[];cases=[]
    for f in b['files']:
        x=load(f);gt=labels[x['key']]
        for scope in ['K4','single']:
            if not x[scope]:continue
            anchors=x[scope]['O0']['anchors'];arms={'Native':score(x['native_boxes'],gt,x['frame_ids'],x['indices'],x['actual_observations'],anchors)}
            row=dict(key=x['key'],cohort=x['cohort'],group=x['source'],scope=scope,arms=arms,anchor_count=len(anchors))
            for arm,z in x[scope].items():
                arms[arm]=score(z['boxes'],gt,x['frame_ids'],x['indices'],x['actual_observations'],anchors)
                costs.append(dict(key=x['key'],cohort=x['cohort'],scope=scope,arm=arm,**{k:z[k] for k in
                    ['backwards','values_calls','offset_suffix_forwards','trial_forwards','fit_seconds','suffix_seconds','accepted_updates','selected_step']}))
                if scope=='single':
                    path=[dict(step=t['step'],loss=t['loss'],details=t['details'],
                        metrics=score(t['boxes'],gt,x['frame_ids'],x['indices'],x['actual_observations'],anchors)) for t in z['path']]
                    cases.append(dict(key=x['key'],cohort=x['cohort'],arm=arm,selected_step=z['selected_step'],path=path))
            rows.append(row)
    comparisons=[(a,'O0') for a in ['O1','O2','O3']]+[('O2','O1'),('O3','O2')]+[(a,'Native') for a in ['O0','O1','O2','O3']]
    for scope in ['K4','single']:
        allstats[scope]={}
        for c in p['configs']:
            rr=[r for r in rows if r['scope']==scope and r['cohort']==c]
            ss=summary(rr,['Native','O0','O1','O2','O3'],METRICS,comparisons)
            ss['new_harm_vs_native']={}
            for a in ['O1','O2','O3']:
                ss['new_harm_vs_native'][a]={str(thr):[r['key'] for r in rr if
                    r['arms'][a]['sIoU']-r['arms']['Native']['sIoU'] < -thr and
                    r['arms']['O0']['sIoU']-r['arms']['Native']['sIoU'] >= -thr] for thr in [.05,.1]}
            ss['local_better_unobserved_worse']={a:[r['key'] for r in rr if r['arms'][a]['expert_IoU'] is not None and
                r['arms'][a]['expert_IoU']-r['arms']['O0']['expert_IoU']>.001 and
                r['arms'][a]['unobserved_sIoU']-r['arms']['O0']['unobserved_sIoU']<-.001] for a in ['O1','O2','O3']}
            allstats[scope][c]=ss
            print(scope,c,{a:{k:round(ss['arms'][a][k]['mean']*100,3) for k in ['sIoU','vIoU_corrected','unobserved_sIoU','expert_IoU']} for a in ['O0','O1','O2','O3']})
    coststats={}
    for c in p['configs']:
        coststats[c]={}
        for a in ['O0','O1','O2','O3']:
            rr=[r for r in costs if r['scope']=='K4' and r['cohort']==c and r['arm']==a]
            coststats[c][a]={k:dict(mean=mean([r[k] for r in rr]),sum=sum(r[k] for r in rr)) for k in
                ['backwards','values_calls','offset_suffix_forwards','trial_forwards','fit_seconds','suffix_seconds','accepted_updates']}
    write(OUT/'ALL_SOURCE_RESULTS.json',rows);write(OUT/'SUMMARY.json',allstats);write(OUT/'COSTS.json',costs)
    write(OUT/'COST_SUMMARY.json',coststats);write(OUT/'CASE_TRAJECTORIES.json',cases)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['probe','main']);a=p.parse_args();run(a.phase)
