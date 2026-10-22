import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from research.joint_event_v3.run import select_local
pairs=np.stack(np.triu_indices(9,1),-1);base=int(np.flatnonzero((pairs==[2,6]).all(1))[0]);s=np.zeros(len(pairs));good=int(np.flatnonzero((pairs==[1,7]).all(1))[0]);far=int(np.flatnonzero((pairs==[0,8]).all(1))[0]);s[good]=1;s[far]=10
raw,pick,diag=select_local(pairs,np.zeros(len(pairs),int),base,s,s[:,None],[s]*18,[dict(total_gain=1)]*2)
assert raw==pick==good and diag['local_candidates']==9
assert abs(pairs[pick]-pairs[base]).max()==1
bad=s.copy();bad[good]=-1
_,pick,_=select_local(pairs,np.zeros(len(pairs),int),base,s,s[:,None],[s]*17+[bad],[dict(total_gain=1)]*2)
assert pick==base
print('PASS: paired nine-neighbor correction, far candidate excluded, any semantic view disagreement returns native')
