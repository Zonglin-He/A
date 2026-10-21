"""Summarize already public saved scores, with no labels or prediction access."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
    d=ROOT/'results/score_audit';p=json.loads((d/'cross_anchor.json').read_text())
    rows=p['queries_anonymous'];a=np.asarray([q['ranking']['CrossAnchor']['pair_accuracy']-q['ranking']['Projection']['pair_accuracy']for q in rows if q['ranking']['CrossAnchor']['pair_accuracy'] is not None and q['ranking']['Projection']['pair_accuracy'] is not None])
    rng=np.random.default_rng(20261021);draw=rng.integers(0,len(a),(10000,len(a)))
    out=dict(metric='macro_candidate_pair_accuracy_CrossAnchor_minus_Projection',parents=len(a),mean_pp=float(100*a.mean()),CI95_pp=(100*np.quantile(a[draw].mean(1),[.025,.975])).tolist(),source_public_sha256=hashlib.sha256((d/'cross_anchor.json').read_bytes()).hexdigest(),no_new_predictions_or_GT_reads=True)
    (d/'ranking_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
