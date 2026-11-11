"""R59 targets/rho unchanged, only clip(gamma * outer confidence,0,1)."""
import sys,torch,numpy as np
from common68 import ROOT
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import cpu_head,readouts,score

def parts(head,hs,z,g):return readouts(head,hs,z,g)[2]
def fusion(part,z,g,gamma=1.):
 out={}
 for v in 'ABC':
  c=np.minimum(1.,gamma*part['confidence'][v])[:,None];out[v]=(1-c)*part['raw'][v]+c*part['targets'][v]
 return score(out,z,g)
