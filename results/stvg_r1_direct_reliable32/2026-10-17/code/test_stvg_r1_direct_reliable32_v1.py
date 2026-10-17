"""Source-independent synthetic wire contracts, no model, pixels or GT."""
import unittest
from scripts.stvg_r1_direct_reliable32_v1 import parse,prompt
from scripts.audit_stvg_r1_direct_interface_v1 import literal_box
class Contracts(unittest.TestCase):
    def test_pixel_mapping(self):self.assertEqual(parse('<answer>[10,20,60,80]</answer>',100,100)['box'],[.35,.5,.5,.6])
    def test_portrait_pixels(self):self.assertEqual(parse('<answer>[20,20,120,80]</answer>',200,100)['box'],[.35,.5,.5,.6])
    def test_duplicate_or_multiple_never_extract_first(self):self.assertIsNone(parse('<answer>[0,0,10,10]</answer><answer>[1,1,9,9]</answer>',20,20)['box'])
    def test_no_guess_clamp_or_reorder(self):
        for a in ['[0,0,1000,1000]','[5,0,2,10]','[0,0,10,0]','[-1,0,5,5]']:self.assertIsNone(parse('<answer>'+a+'</answer>',100,50)['box'])
    def test_no_boolean_nan(self):
        for a in ['[true,0,10,10]','[NaN,0,10,10]']:self.assertIsNone(parse('<answer>'+a+'</answer>',20,20)['box'])
    def test_truncation(self):self.assertEqual(parse('<answer>[0,0,10,10]</answer>',20,20,True)['status'],'truncated')
    def test_null_separate(self):self.assertEqual(parse('<answer>null</answer>',20,20)['status'],'abstained')
    def test_invalid_json_has_no_supervision(self):
        d=literal_box('{"bbox_2d":[0,0,10,10], "bbox_2d}',20,20)
        self.assertEqual(d['category'],'multiple_or_dangling_duplicate_key_ambiguous');self.assertTrue(d['diagnostic_never_admitted_as_reference'])
    def test_one_domain_independent_prompt(self):
        a=dict(physical_timestamp=1.25,query_text='quoted data',dataset='hc2');b=dict(a,dataset='vidstg');self.assertEqual(prompt(a,100,50),prompt(b,100,50))
if __name__=='__main__':unittest.main()
