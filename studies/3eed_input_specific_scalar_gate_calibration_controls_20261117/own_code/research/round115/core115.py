"""Test input-specific scalar gate adaptation against constant and shuffled edits."""
from pathlib import Path
import sys
R115=Path(__file__).resolve().parents[2];H115=R115/'research/round115';B115=R115/'crossplatform/round115'
sys.path.insert(0,str(R115/'research/round114'));from core114 import *
REF115=['original_source','prepared_frozen','native_real_fusedABC','direct_three_geometric','neither_correct_A','neither_correct_fusedABC','anchor_frozen_C','anchor_noLate_fusedABC','native_gate_only_A','native_gate_only_C','native_gate_only_fusedABC','native_bridge_no_gate_fusedABC','sourcekey_frozen_fusedABC','gateI0_frozen_fusedABC']
NEW115=[k+'_'+v for k in ['fixed_source_mean_gate','shuffled_gate'] for v in V114];ARMS115=REF115+NEW115;PRIMARY115='native_gate_only_fusedABC'

def controls115():
 """Only saved current-A gradients, no task labels or heldout hidden states."""
 out={};summary=[]
 for src in ['waymo','drone','quad']:
  sc=jread(B113/(src+'_THETA_SCHEMA_PRIVATE.json'));j=sc['names'].index('gate_I');off=sum(sc['sizes'][:j]);assert sc['sizes'][j]==1;rows=[]
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   for r in jread(R115/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    name=src+'_to_'+tgt;sid=r['id'];meta=jread(B113/name/'records_PRIVATE'/(sid+'.json'))
    if meta['protected_alias_fallback']:continue
    with np.load(B113/name/'fits_PRIVATE'/(sid+'.npz')) as fs:gi=float(fs['gradient_anchor'][off])
    # Exact same float32 scalar clip1 step as R114 raw gate-only proposal.
    g=torch.tensor(gi,dtype=torch.float32);v=float(-.5*torch.clamp(1/(g.abs()+1e-6),max=1)*g);rows.append((name,sid,v))
  n=len(rows);seed=int(hashlib.sha256(('R115_source_gate_shuffle|'+src).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed);perm=rng.permutation(n)
  while np.any(perm==np.arange(n)):perm=rng.permutation(n)
  mean=float(np.mean([x[2] for x in rows]));summary.append({'source':src,'n':n,'mean_delta_gate':mean,'source_specific_mean_uses_A_only_development_no_GT':True,'shuffled_fixedpoints':0})
  for i,(name,sid,v) in enumerate(rows):out[(name,sid)]={'dynamic':v,'fixed_source_mean_gate':mean,'shuffled_gate':rows[int(perm[i])][2]}
 return out,summary
