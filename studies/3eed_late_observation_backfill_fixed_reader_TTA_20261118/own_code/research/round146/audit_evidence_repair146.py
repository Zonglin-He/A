"""No GT: compare all generated arrays across metadata-only repair."""
from common146 import *
import numpy as np

def main():
 a=B/'evidence_attempt001_PRIVATE';b=B/'evidence_PRIVATE';n=0
 for p in sorted(a.rglob('*.npz')):
  x=np.load(p);y=np.load(b/p.relative_to(a));assert x.files==y.files
  for k in x.files:assert np.array_equal(x[k],y[k]),(p,k)
  n+=1
 assert n==1798;dump(H/'EVIDENCE_REPAIR_AUDIT.json',{'status':'PASS','n':n,'all_arrays_exact':True,'failure':'numpy int64 counter not JSON serializable afterall_evidencearrays_written_beforeseal','repair':'Pythonint counter serialization only','before_any_R146_gradient':True,'GT_read':False});print(n,flush=True)
if __name__=='__main__':main()
