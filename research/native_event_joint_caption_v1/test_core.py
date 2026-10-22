import unittest
from collections import Counter
from research.native_event_joint_caption_v1.core import joint_variants
def toy():
 data=[('the',2,'DET','det'),('man',3,'NOUN','nsubj'),('closes',0,'VERB','root'),('door',3,'NOUN','obj'),('and',6,'CCONJ','cc'),('walks',3,'VERB','conj'),('outside',6,'ADV','advmod'),('and',9,'CCONJ','cc'),('turns',3,'VERB','conj'),('back',9,'ADV','advmod')]
 return [dict(id=i+1,head=h,upos=u,rel=rel,text=t,lemma=t)for i,(t,h,u,rel)in enumerate(data)]
class Tests(unittest.TestCase):
 def test_all_middle_and_arguments_kept(self):
  x=joint_variants(toy(),'original');self.assertTrue(x['supported']);self.assertEqual(len(x['parent']['clauses']),3)
  self.assertEqual(x['canonical'],'the man closes door and the man walks outside and the man turns back.')
  self.assertTrue(all(v in x['canonical']for v in ['closes door','walks outside','turns back']))
 def test_same_word_bag_binding_change(self):
  x=joint_variants(toy(),'original');self.assertEqual(Counter(x['canonical'].split()),Counter(x['predicate_binding_control'].split()))
  self.assertNotEqual(x['canonical'],x['predicate_binding_control']);self.assertIn('walks door',x['predicate_binding_control'])
 def test_shared_subject_only_once(self):
  x=joint_variants(toy(),'original');self.assertEqual(x['implicit_shared_subject'],'the man closes door and walks outside and turns back.')
  self.assertEqual(x['nominal_subject_word_IDs'],[1,2])
 def test_unsupported_is_original_alias(self):
  ws=toy();ws[4]['lemma']='while';x=joint_variants(ws,'original');self.assertFalse(x['supported']);self.assertEqual(x['parent']['clauses'],['original'])
if __name__=='__main__':unittest.main()
