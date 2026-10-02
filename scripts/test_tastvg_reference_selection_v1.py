"""Analytic contracts for duplicates, unequal evidence, empty feedback and ties."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_reference_selection_v1 import student_frames,pairwise,decision
def run():
    c=[dict(physical_interval=[0,10])];r=student_frames(c,list(range(10)))
    assert r['raw_quantiles']==[0,2,4,6,8] and r['actual_unique']==5
    r=student_frames([dict(physical_interval=[4,5])],list(range(10)))
    assert r['raw_unique']==1 and r['actual_unique']==5 and 4 in r['positions'] and len(r['filled'])==4
    assert r==student_frames([dict(physical_interval=[4,5])],list(range(10)))
    u=list(range(9));r=pairwise(u,u);assert r['pairwise_accuracy']==r['decisive_coverage']==1 and r['strict_GT_pairs']==36
    r=pairwise(None,u);assert r['pairwise_accuracy']==.5 and r['decisive_coverage']==0
    r=pairwise(u,[1]*9);assert r['GT_ties']==36 and r['pairwise_accuracy'] is None
    d=decision([np.zeros((5,4))]*9,np.zeros((5,4)),np.zeros(5,bool));assert not d['available'] and d['selected']==0
    print('PASS6 analytic contracts; no model or GT')
if __name__=='__main__':run()
