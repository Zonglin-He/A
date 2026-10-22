import unittest
import numpy as np
from research.native_event_composition_v1.core import from_words,envelope

class CompositionTests(unittest.TestCase):
    def test_envelope_matches_enumeration(self):
        rng=np.random.default_rng(4);q=np.triu(rng.random((6,6)),1);r=np.triu(rng.random((6,6)),1);q/=q.sum();r/=r.sum();expected=np.zeros_like(q)
        for a,b in zip(*np.triu_indices(6,1)):
            for c,d in zip(*np.triu_indices(6,1)):expected[min(a,c),max(b,d)]+=q[a,b]*r[c,d]
        np.testing.assert_allclose(envelope(q,r),expected,atol=1e-15,rtol=1e-13)
    def test_simple_and_retains_subject(self):
        vals=[('the','DET',2,'det'),('man','NOUN',3,'nsubj'),('walks','VERB',0,'root'),('and','CCONJ',5,'cc'),('sits','VERB',3,'conj'),('down','ADP',5,'compound')]
        w=[dict(id=i+1,text=x,lemma=x,upos=u,head=h,rel=d)for i,(x,u,h,d)in enumerate(vals)];d=from_words(w,'the man walks and sits down.');self.assertTrue(d['supported']);self.assertEqual(d['first'],'the man walks.');self.assertEqual(d['last'],'the man sits down.')
    def test_no_and_does_not_invent_event(self):
        w=[dict(id=1,text='walks',lemma='walk',upos='VERB',head=0,rel='root')];self.assertFalse(from_words(w,'walks')['supported'])

if __name__=='__main__':unittest.main()
