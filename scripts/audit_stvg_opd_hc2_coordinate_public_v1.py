"""Standalone NumPy-only reproduction of public coordinates, statistics and tails."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

FIELDS=['Frozen_v','Before_v','After_v','Frozen_t','After_t','Frozen_s','Before_s','After_s',
    'delta_total_v','delta_current_v','delta_inherited_v','delta_total_s',
    'correct_to_wrong_0.3','correct_to_wrong_0.5','wrong_to_correct_0.3','wrong_to_correct_0.5']
def read(path):return json.loads(path.read_text())
def run(base):
    base=Path(base);root=read(base/'ROOT_AUDIT.json');checks=0;frozen=None;effects={}
    for trial,audit in root['trials'].items():
        with gzip.open(base/'trials'/trial/'ROWS.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
        assert len(rows)==64 and len({(r['source_id'],r['order']) for r in rows})==64
        byparent={}
        for r in rows:byparent.setdefault(r['source_id'],[]).append(r)
        assert len(byparent)==32 and all(len(v)==2 for v in byparent.values())
        matrix=np.stack([np.array([[r[f] for f in FIELDS] for r in byparent[s]]).mean(axis=0) for s in sorted(byparent)])
        rng=np.random.default_rng(20261006);boots=np.concatenate([matrix[rng.integers(0,32,(100,32))].mean(axis=1) for _ in range(100)])
        ci=np.percentile(boots,[2.5,97.5],axis=0)
        for j,field in enumerate(FIELDS):
            m=audit['metrics'][field];assert abs(matrix[:,j].mean()-m['mean'])<3e-12
            assert np.max(np.abs(ci[:,j]-m['ci95']))<3e-12;checks+=3
            if field.startswith('delta_'):
                assert int((matrix[:,j]<-.05).sum())==m['harm_gt5pp_sources'] and int((matrix[:,j]<-.2).sum())==m['harm_gt20pp_sources'];checks+=2
        inputs={(r['query_ordinal'],r['source_id'],r['order'],r['arrival']):(r['Frozen_v'],r['Frozen_t'],r['Frozen_s']) for r in rows}
        if frozen is None:frozen=inputs
        else:assert inputs==frozen
        for r in rows:
            assert abs(r['delta_total_v']-r['delta_current_v']-r['delta_inherited_v'])<3e-15
            assert r['Frozen_t']==r['Before_t']==r['After_t'];checks+=2
        effects[trial]=audit['metrics']['delta_total_v']['mean']
    cfg=read(base/'DESIGN_LOCK.json')['start'];total=0
    for chain in root['coordinate_chain']:
        assert cfg==chain['incumbent'];valid=[]
        for c in chain['candidates']:
            total+=1
            assert all(c['config'][k]==cfg[k] for k in cfg if k!=chain['coordinate'])
            if c['status']!='complete':continue
            a=root['trials'][c['trial']]['metrics']['delta_total_v']
            key=(a['mean'],-a['harm_gt20pp_sources'],int(c['config']==cfg),-c['config']['steps'],-c['ordinal'])
            valid.append((key,c))
        winner=max(valid,key=lambda x:x[0])[1];assert winner['trial']==chain['trial'] and winner['config']==chain['selected'];cfg=winner['config'];checks+=1
    assert cfg==root['selected_config']==read(base/'SELECTED_CONFIG.json')['config'] and total==26
    out=dict(status='pass',public_scalar_checks=checks,complete_configurations=len(root['trials']),
        numerical_invalid_configurations=len(root['invalid_trials']),all_32_parent_sources=True,
        logical_candidate_evaluations=26,paired_source_bootstrap=10000,selected_config=cfg,
        verification_scope='anonymous scalar rows/greedy ranking/bootstrap only; no private model or GT replay')
    print(json.dumps(out));return out

if __name__=='__main__':run(sys.argv[1])
