"""Algebra-only audit; production selector is read and hashed, never changed."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 import numpy as np
 p=ROOT/'scripts/stvg_referent_abc_tokens002.py';before=hashlib.sha256(p.read_bytes()).hexdigest();rng=np.random.default_rng(20261017);x=np.vstack([rng.random((10000,2)),[[1,0],[.2,.2],[0,0],[1,1],[1/3,2/3]]]);a,b=x.T;old=.75*np.minimum(a,b)+.25*np.abs(a-b);agreement=.75*(a+b)/2-.125*np.abs(a-b);robust=np.minimum(a/3+2*b/3,2*a/3+b/3);assert np.max(np.abs(old-agreement))<1e-15 and np.max(np.abs(old/.75-robust))<1e-15
 # Dividing the relevance score alone does not preserve a relevance-diversity MMR.
 s=np.array([.30,.15]);similarity=np.array([.18,0.]);assert np.argmax(.5*s-.5*similarity)!=np.argmax(.5*(s/.75)-.5*similarity)
 assert np.allclose((.5*s-.5*similarity)/.75,.5*(s/.75)-(2/3)*similarity)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==before
 d=dict(status='pass',count=len(x),maximum_equivalence_absolute_error=float(np.max(np.abs(old-agreement))),maximum_robust_equivalence_absolute_error=float(np.max(np.abs(old/.75-robust))),same_average_disagreement_coefficient=-.125,original_scale_retained=True,MMR_uncompensated_rescaling_not_equivalent=True,selector_sha256_before=before,selector_sha256_after=before,new_models=0,new_GT_reads=0,GT_read=False)
 out=ROOT/'artifacts/stvg_b_score_algebra_v1';out.mkdir(exist_ok=False);(out/'CPU_CONTRACT_COMPLETION.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
