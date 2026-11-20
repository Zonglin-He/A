"""Track the numerical weighted-median audit against the sealed first implementation."""
import csv
import gzip
import numpy as np
from paths_witness import WORK,PRED,OUT,name,read,save,digest
from predict_witness import ARMS
from score_roles import csv_file

def run():
    before=WORK/'before_tie_audit'
    oldseal=read(before/'predictions/SEALED.json');newseal=read(PRED/'SEALED.json')
    assert oldseal['obs']==newseal['obs'] and oldseal['protocol']==newseal['protocol']
    assert oldseal['files'].keys()==newseal['files'].keys()
    fields=['rows','raw_changes','ref_changes','shape_changes','threshold25_changes','threshold50_changes']
    counts={arm:dict.fromkeys(fields,0) for arm in ARMS}
    def values(path):
        with gzip.open(path,'rt') as f:
            return {(r['case'],r['source'],r['arm']):float(r['iou']) for r in csv.DictReader(f)}
    first=values(before/'results/all_rows.csv.gz');last=values(OUT/'all_rows.csv.gz')
    for r in read(WORK/'roster.json'):
        key=name(r)
        assert digest(before/'predictions'/key)==oldseal['files'][key] and digest(PRED/key)==newseal['files'][key]
        with np.load(before/'predictions'/key) as a,np.load(PRED/key) as b:
            for j,arm in enumerate(ARMS):
                stats=counts[arm];stats['rows']+=1
                stats['raw_changes']+=int(not np.array_equal(a['raw'][j],b['raw'][j],equal_nan=True))
                stats['ref_changes']+=int(not np.array_equal(a['ref'][j],b['ref'][j],equal_nan=True))
                stats['shape_changes']+=int(np.max(abs(a['boxes'][j]-b['boxes'][j]))>1.e-8)
                k=r['uid'],r['source'],arm
                for t in (.25,.5):
                    stats['threshold'+str(int(t*100))+'_changes']+=int((first[k]>t)!=(last[k]>t))
    assert not any(v['raw_changes'] or v['ref_changes'] for v in counts.values())
    csv_file(OUT/'numeric_replay.csv',[dict(arm=a,**v) for a,v in counts.items()])
    save(OUT/'numeric_receipt.json',dict(first_prediction_seal=digest(before/'predictions/SEALED.json'),
        corrected_prediction_seal=digest(PRED/'SEALED.json'),fixed='project original face onto both-sided numerical half-mass interval; canonical ordering within each bank',
        reran_every_case=len(oldseal['files']),unchanged_protocol_and_observations=True))

if __name__=='__main__':run()
