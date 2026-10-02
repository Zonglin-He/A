"""Anonymous control, evidence-denominator and seal-chronology consistency."""
import sys,json,collections
from pathlib import Path
import numpy as np
def run(folder):
    checks=0
    def close(a,b):
        nonlocal checks
        assert abs(a-b)<1e-12,(a,b);checks+=1
    seal=json.loads((folder/'SEAL_RECEIPT.json').read_text())
    assert seal['global_seal_time']<min(seal['GT_exposure_times'].values())
    assert seal['attempts']==seal['receipts']==seal['call_cap']==62 and all(seal['uniform_parity'].values());checks+=5
    for ds in ['vidstg','hc2']:
        rows=json.loads((folder/ds/'ROWS.json').read_text());summary=json.loads((folder/ds/'SUMMARY.json').read_text())
        control=json.loads((folder/ds/'FINAL_INTERVAL_CONTROL.json').read_text())
        assert control['post_primary_supplement'] and control['primary_endpoint_unchanged'];checks+=2
        for a,r in zip(rows,control['rows']):
            assert (a['cell'],a['source_id'],a['condition'],a['order'])==(r['cell'],r['source_id'],r['condition'],r['order']);checks+=1
            for strategy in ['Uniform','Routed']:
                s=a[strategy+'_selected'];u=r['utilities'];close(r[strategy+'_v'],u[s]);close(r[strategy+'_gain'],u[s]-u[0]);close(r[strategy+'_regret'],max(u)-u[s])
                assert a[strategy+'_empty']==int(a[strategy+'_valid_frames']==0)
                assert a[strategy+'_no_event_valid']==int(a[strategy+'_valid_event_frames']==0);checks+=2
                close(a[strategy+'_event_hits']/5,a[strategy+'_event_precision'])
                strict=0;v=0.;rewards=a[strategy+'_rewards'] or [0.]*9
                for i in range(9):
                    for j in range(i+1,9):
                        b=int(u[i]-u[j]>1e-12)-int(u[i]-u[j]< -1e-12);es=int(rewards[i]-rewards[j]>1e-12)-int(rewards[i]-rewards[j]< -1e-12)
                        if b:strict+=1;v+=.5 if es==0 else float(es==b)
                if strict:close(v/strict,r[strategy+'_pairwise'])
                else:assert r[strategy+'_pairwise'] is None
            close(r['delta_v'],r['Routed_v']-r['Uniform_v'])
            if r['delta_pairwise'] is not None:close(r['delta_pairwise'],r['Routed_pairwise']-r['Uniform_pairwise'])
        for group,z in control['summary'].items():
            take=[r for r in control['rows'] if (r['condition']!='clean')==(group=='corruption')]
            for field,result in z.items():
                rr=[r for r in take if r[field] is not None];by=collections.defaultdict(list)
                for r in rr:by[r['source_id']].append(r[field])
                vector=np.asarray([np.mean(by[s]) for s in sorted(by)]);m=result['metrics'][field];close(vector.mean(),m['mean'])
                rng=np.random.default_rng(20261001);bootstrap=np.concatenate([vector[rng.integers(0,len(vector),(100,len(vector)))].mean(1) for _ in range(100)]);ci=np.quantile(bootstrap,[.025,.975])
                close(ci[0],m['ci95'][0]);close(ci[1],m['ci95'][1])
        for group,z in summary.items():
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption')]
            for strategy,counts in z['counts'].items():
                assert counts['empty_arrivals']==sum(r[strategy+'_empty'] for r in rr)
                assert counts['valid_frames']==sum(r[strategy+'_valid_frames'] for r in rr)
                assert counts['no_event_valid']==sum(r[strategy+'_no_event_valid'] for r in rr)
                assert counts['no_event_valid_nonempty']==sum(r[strategy+'_no_event_valid'] and not r[strategy+'_empty'] for r in rr);checks+=4
                for scope in ['fixed','joint']:
                    for threshold,c in counts[scope].items():
                        t=float(threshold);assert c['support_correct']==sum(max(r[scope+'_utilities'])>t for r in rr)
                        assert c['missed_correct']==sum(max(r[scope+'_utilities'])>t and r[strategy+'_'+scope+'_v']<=t for r in rr)
                        assert c['no_correct_support']==sum(max(r[scope+'_utilities'])<=t for r in rr)
                        assert c['native_correct_destroyed']==sum(r[scope+'_utilities'][0]>t and r[strategy+'_'+scope+'_v']<=t for r in rr);checks+=4
    r=dict(status='pass',checks=checks,post_primary_control_disclosed=True);print(json.dumps(r));return r
if __name__=='__main__':run(Path(sys.argv[1]))
