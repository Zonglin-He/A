"""Small analytic cases, not model-dependent implementation snapshots."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_aligned_token_binding_v1 import phrases,roi,score,independent_score
class Cases(unittest.TestCase):
 def test_explicit_noun_action(self):
  rows=[('man','man','NOUN',5,'nsubj'),('in','in','ADP',4,'case'),('blue','blue','ADJ',4,'amod'),('shirt','shirt','NOUN',1,'nmod'),('opens','open','VERB',0,'root'),('door','door','NOUN',5,'obj')]
  words=[dict(id=i+1,text=z[0],lemma=z[1],upos=z[2],head=z[3],deprel=z[4]) for i,z in enumerate(rows)];p=phrases(words);self.assertEqual(p['object'],['man','blue shirt']);self.assertEqual(p['event'],['opens','door'])
 def test_roi_no_fill(self):
  self.assertFalse(roi([2.,2.,.1,.1],2,2).any());self.assertEqual(roi([.25,.25,.5,.5],2,2).tolist(),[True,False,False,False])
 def test_max_direction_and_independent(self):
  p=np.array([[[1.,0],[0,1],[0,1],[0,1]],[[0.,1],[1,0],[1,0],[1,0]]]);t=np.tile([[.25,.25,.5,.5],[.25,.25,.5,.5]],(9,1,1));q=np.array([[1.,0]])
  x=score(p,q,q,t,[0,0]);self.assertEqual(x['binding_scores'],[1.]*9);self.assertEqual(x['unique_ROI_signatures'],1);self.assertEqual(independent_score(p,q,q,t,[0,0]),[(1.,1.)]*9)
 def test_empty_event_and_outside(self):
  p=np.ones((2,4,2));q=np.ones((1,2));t=np.tile([[.5,.5,1.,1.],[.5,.5,1.,1.]],(9,1,1));self.assertFalse(score(p,q,[],t,[0,0])['all9_available']);self.assertEqual(score(p,q,q,t,[0,1])['binding_scores'],[None]*9)
 def test_negative_and_scale(self):
  p=np.array([[[0.,1]]*4,[[1.,0]]*4]);q=np.array([[1.,0]]);t=np.tile([[.5,.5,1.,1.],[.5,.5,1.,1.]],(9,1,1));self.assertEqual(score(p*4,q*3,q*3,t,[0,0])['binding_scores'],[-1.]*9)
if __name__=='__main__':unittest.main()
