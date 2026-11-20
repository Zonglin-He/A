"""Recalculate public denominators, forecast contrasts and permutation checks from sealed outputs."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import gzip
import numpy as np
from paths_witness import HERE,WORK,OUT,LAST,OBS,PRED,name,read,save,digest
from evaluate_witness import check_seals
from predict_witness import ARMS,readout
from score_roles import csv_file

def run():
    sealed,observed=check_seals();labels=read(WORK/'labels.json')
    roster=read(WORK/'roster.json')
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    lookup={(x['case'],x['source'],x['arm']):x for x in rows}
    assert len(rows)==len(lookup)==len(roster)*len(ARMS)
    counts=Counter();structure=[];example=defaultdict(list);audited=0;err=0.
    for r in roster:
        uid,src=r['uid'],r['source']
        group=lookup[uid,src,'v1']['group'];target=r['frame']['target']
        path=name(r)
        with np.load(OBS/path) as z,np.load(PRED/path) as pred:
            rn,rf,boxes=pred['raw'],pred['ref'],pred['boxes']
            np.testing.assert_array_equal(rn[0],z['raw']);np.testing.assert_array_equal(boxes[0],z['raw'])
            np.testing.assert_array_equal(rf[0],z['ref'])
            np.testing.assert_array_equal(rn[5:],np.tile(rn[5],(4,1)))
            np.testing.assert_array_equal(rf[5:],np.tile(rf[5],(4,1)))
            if not z['active']:
                assert not pred['applied'].any()
            active=bool(pred['applied'][-1]);counts[group,'all']+=1;counts[group,'active']+=active
            counts[group,'same_choices_as_halfplane']+=int(np.array_equal(rn[5],rn[2]) and np.array_equal(rf[5],rf[2],equal_nan=True))
            counts[group,'different_vs_halfplane']+=int(not np.array_equal(rn[5],rn[2]))
            counts[group,'different_vs_wrong']+=int(not np.array_equal(rn[5],rn[4]))
            counts[group,'different_shape_from_unconditioned']+=int(np.max(abs(boxes[-1]-boxes[6]))>1e-8)
            counts[group,'immediate']+=int(bool(z['active']) and bool(z['immediate']))
            if active and counts[group,'permuted']<3:
                data=dict(z)
                for prefix in ('t','r'):
                    for key in ('box','project','p','m','count'):data[prefix+'_'+key]=data[prefix+'_'+key][::-1]
                data['a']=data['a'][::-1];data['ma']=data['ma'][::-1]
                rerun=readout(data)
                for key in ('boxes','ref','raw'):
                    np.testing.assert_allclose(rerun[key],pred[key],atol=1e-8,rtol=0,equal_nan=True)
                    err=max(err,float(np.nanmax(abs(rerun[key]-pred[key]))))
                np.testing.assert_allclose(rerun['evidence'],pred['evidence'],atol=1e-12,rtol=0)
                counts[group,'permuted']+=1;audited+=1
            for arm in ARMS:
                rr=lookup[uid,src,arm]
                if arm=='witness_chord':
                    a0=float(lookup[uid,src,'v1']['iou']);a1=float(rr['iou'])
                    kind='rescue' if a1>.5>=a0 else ('harm' if a0>.5>=a1 else None)
                    if kind:example[group,kind].append(dict(group=group,type=kind,case=uid,source=src,target=target,
                        v1_iou=a0,witness_iou=a1,raw_iou=float(rr['raw_iou']),all_pool=float(rr['all_pool'])))
    for g in sorted({x['group'] for x in rows}):
        s=dict(group=g)
        for k in ('all','active','immediate','same_choices_as_halfplane','different_vs_halfplane','different_vs_wrong','different_shape_from_unconditioned','permuted'):
            s[k]=counts[g,k]
        structure.append(s)
    csv_file(OUT/'structure.csv',structure)
    csv_file(OUT/'threshold_cases.csv',[x for stack in example.values() for x in stack])
    with (OUT/'paired.csv').open() as f:
        for item in csv.DictReader(f):
            g,arm,base=item['group'],item['arm'],item['control']
            if item['direction']!='macro':continue
            test=[r for r in rows if r['group']==g and r['arm']==arm]
            assert len(test)==int(item['n'])
            for th in (25,50):
                diffs=[int(float(r['iou'])>th/100)-int(float(lookup[r['case'],r['source'],base]['iou'])>th/100) for r in test]
                assert diffs.count(1)==int(item['rescue'+str(th)]) and diffs.count(-1)==int(item['harm'+str(th)])
    import verify_bound
    verify_bound.OUT=OUT
    verify_bound.run()
    assert digest(OUT/'bound_recheck.csv')==digest(LAST/'results/bound_recheck.csv')
    natural=[r for r in rows if r['group']=='natural' and r['arm']=='v1' and int(r['known_ends'])]
    assert len(natural)==42
    save(OUT/'verification.json',dict(all_records=len(roster),contrasts_recomputed=True,permuted_replays=audited,
        permutation_box_error=err,natural_known_records=len(natural),bound_reproduced_bytewise=True,
        current_method_sha=digest(HERE.parents[1]/'crossplatform/CURRENT_METHOD.json'),
        scoring_code={f:digest(HERE/f) for f in ('evaluate_witness.py','tables_witness.py','diagnose_witness.py','verify_witness.py')}))

if __name__=='__main__':run()
