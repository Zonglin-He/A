"""Independent rank-metric re-score of round2 GREAT, after all prediction hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from audit_great import measure

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--targets',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    p=json.loads((a.run/'protocol.json').read_text());assert p['state']=='sealed'
    seal=json.loads((a.run/'SEALED.json').read_text())
    for name,digest in seal.items():assert sha(a.run/name)==digest
    first=json.loads((a.run/'SCORED.json').read_text());refs=json.loads(a.targets.read_text());assert len(refs)==p['count']
    targets={r['id']:(i,np.load(r['native_annotation_container'],allow_pickle=False)[:,3:].copy()) for i,r in enumerate(refs)}
    cache={};errors=[];out=[]
    for row in first['rows']:
        i,t=targets[row['id']];key=(i,row['objective'],row['step'])
        if key not in cache:
            with np.load(a.run/f'{i:04d}_{row["objective"]}_k{row["step"]}.npz',allow_pickle=False) as d:cache[key]={k:d[k].copy() for k in d.files if k!='native_views'}
        values=measure(cache[key][row['arm']],t)
        for k,v in values.items():
            if v is None:assert row[k] is None
            else:errors.append(abs(v-row[k]));assert abs(v-row[k])<1e-6
        out.append({k:row[k] for k in ['id','parent_id','objective','step','arm']}|values)
    result={'all_prediction_seals_verified_before_numeric_targets':True,'independent_AUC_rank_formula':True,'maximum_metric_error':max(errors),'n_parents':len(refs),'rows':out}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
