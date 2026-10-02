"""Model-free analytic aggregation and adversarial scalar-tamper checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_tastvg_selected_rollout_public_v1 import summary,equal,ranks
from scripts.score_tastvg_best_quick_v1 import source_summary
def run():
 # Unequal multiplicities and orders must not silently become cell means.
 rows=[]
 for source,orders in [(0,{'o1':[1.,1.],'o2':[0.]}),(1,{'o1':[-1.],'o2':[3.,3.,3.]})]:
  for order,vals in orders.items():
   for i,value in enumerate(vals):rows.append(dict(source_id=source,order=order,condition='blur' if i%2 else 'drop',delta_x=value))
 a=summary(rows,['delta_x']);b=source_summary(rows,['delta_x']);assert equal(a,b)>0
 assert a['metrics']['delta_x']['mean']==.75 and a['metrics']['delta_x']['query_macro']!=.75
 # Independent tie grouping and intentional tampering must be detected.
 np.testing.assert_array_equal(ranks([1.,1.,.2]),[.5,.5,2.])
 assert equal({'x':None,'y':[True,2.]},{'x':None,'y':[True,2.]})==3
 caught=False
 try:equal(a,{**a,'sources':3})
 except AssertionError:caught=True
 assert caught
 print('PASS: analytic source/order/condition weighting, independent bootstrap parity, rank ties, null/bool checks, deliberate tamper detection')
if __name__=='__main__':run()
