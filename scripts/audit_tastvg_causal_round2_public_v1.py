"""Rebuild Round2 public aggregates from anonymous scalar rows; no model access."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def audit(folder):
    read=lambda p:json.loads(p.read_text())
    index=read(folder/'ROWS_INDEX.json')['groups'];groups={}
    for name,entry in index.items():
        rows=[]
        for part in entry['files']:
            rr=read(folder/part['path']);assert len(rr)==part['rows'];rows+=rr
        assert len(rows)==entry['rows'];groups[name]=rows
    rows=groups['ORACLE_ROWS'];assert len(rows)==64 and len({r['key'] for r in rows})==64
    summary=read(folder/'ORACLE_SUMMARY.json');assoc=read(folder/'ASSOCIATIONS.json');pres=read(folder/'PRESERVATION.json');checks=0
    def equal(a,b):
        nonlocal checks
        assert abs(a-b)<1e-11,(a,b);checks+=1
    for cohort in ('hcstvg1_test','vidstg_test','all'):
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];s=summary[cohort];assert len(rr)==s['n']
        for metric in ('sIoU','tIoU','vIoU_corrected','tube_supported_sIoU'):
            equal(np.mean([r['baseline'][metric] for r in rr]),s['B0'][metric]['mean'])
            for arm,a in s['arms'].items():
                vals=np.array([r['arms'][arm]['metrics'][metric] for r in rr]);delta=vals-np.array([r['baseline'][metric] for r in rr])
                equal(vals.mean(),a['absolute'][metric]['mean']);equal(delta.mean(),a['delta'][metric]['mean'])
                assert int((delta<-.05).sum())==a['tails'][metric]['harm_gt5pp']
        for feature in ('VA','VAmean'):
            for name,y in [('baseline_spatial_error',[1-r['baseline']['sIoU'] for r in rr]),('spatial_oracle_gain',[r['arms']['OS']['metrics']['sIoU']-r['baseline']['sIoU'] for r in rr]),('joint_oracle_gain',[r['arms']['OST']['metrics']['vIoU_corrected']-r['baseline']['vIoU_corrected'] for r in rr])]:
                equal(float(spearmanr([r[feature] for r in rr],y).statistic),assoc[cohort][feature+'/'+name]['rho'])
        for bare,protected,metric in [('OS','OS_PT','sIoU'),('OT','OT_PS','tIoU')]:
            b=np.array([r['arms'][bare]['metrics'][metric]-r['baseline'][metric] for r in rr]);p=np.array([r['arms'][protected]['metrics'][metric]-r['baseline'][metric] for r in rr]);positive=b>1e-12
            assert int(positive.sum())==pres[cohort][protected]['bare_positive_queries']
            if positive.any():equal(p[positive].sum()/b[positive].sum(),pres[cohort][protected]['useful_gain_retention'])
    return dict(status='pass',queries=64,adaptations=384,aggregate_checks=checks,raw_tensors_read=False,scope='public scalar aggregation only; tensor and native-loss audits are separate')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();print(json.dumps(audit(args.folder),indent=2))
