import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_controlled_query_math_v1 import choose_donor,pair_kind,query_form,native_subject
from scripts.tastvg_information_atlas_math_v1 import moments,paired_difference


def row(i,caption,ref,action,subject='man',media='v',segment=(0,100)):
    return dict(key=str(i),caption=caption,form=query_form(caption),target_id=ref,video_sha256=media,segment=segment,
        signature=dict(subject=subject,subject_signature=(subject,),root_action=action,actions=(action,),lexical_verbs=(action,)))


class Checks(unittest.TestCase):
    def test_same_video_priority_and_identity(self):
        a=row(0,'A man walks.',1,'walk');b=row(1,'A man runs.',1,'run');c=row(2,'A man jumps.',2,'jump',media='other')
        d=choose_donor(a,[c,b],'event','test');self.assertEqual(d['donor']['key'],'1');self.assertEqual(d['tier'],0)
        self.assertIsNone(pair_kind(a,row(3,'A woman runs.',2,'run',subject='woman'),'event'))

    def test_subject_exact_action_and_wrong_action(self):
        a=row(0,'A man walks.',1,'walk');b=row(1,'A woman walks.',2,'walk',subject='woman')
        self.assertEqual(pair_kind(a,b,'subject')[0],0)
        self.assertIsNone(pair_kind(a,row(2,'A woman runs.',2,'run',subject='woman'),'subject'))

    def test_no_generic_fill_and_caption_form(self):
        a=row(0,'A man walks.',1,'walk');b=row(1,'Who runs?',2,'run')
        self.assertIsNone(choose_donor(a,[b],'event','test'))
        self.assertIsNone(choose_donor(a,[a],'subject','test'))

    def test_disjoint_requires_all_verbs(self):
        a=row(0,'A man walks.',1,'walk');b=row(1,'A man runs while walking.',1,'run')
        b['signature']['lexical_verbs']=('run','walk')
        self.assertIsNone(pair_kind(a,b,'event'))

    def test_same_source_is_not_same_media(self):
        a=row(0,'A man walks.',1,'walk');b=row(1,'A man runs.',1,'run',media='cut2')
        a['source']=b['source']='youtube';self.assertGreater(pair_kind(a,b,'event')[0],1)

    def test_specificity_cancels_true_on_common_cohort(self):
        rows=[]
        for i in range(3):
            y=[0,.5,1];rows.append(dict(source_index=i,order='o1',condition='blur',metrics=dict(event=moments(y,[.3,.4,.7]),subject=moments(y,[.1,.5,.9]))))
        s=paired_difference(rows,'subject','event','r2',1000)
        self.assertGreater(s['mean'],0);self.assertGreater(s['ci95'][0],0)

    def test_unknown_what_is_not_inferred_identity(self):
        words=[dict(lemma='what',deprel='nsubj',upos='PRON',local_id=1,local_head=2),dict(lemma='walk',deprel='root',upos='VERB',local_id=2,local_head=0)]
        self.assertEqual(native_subject(words),'')

    def test_shared_light_verb_is_only_weak_control(self):
        a=row(0,'A girl takes steps.',1,'take',subject='girl');b=row(1,'A man takes a cup.',2,'take',media='other')
        a['signature']['identity_signature']=[('verb','take'),('obj','step')]
        b['signature']['identity_signature']=[('verb','take'),('obj','cup')]
        self.assertEqual(pair_kind(a,b,'subject')[0],3)
        b['signature']['identity_signature']=[('verb','take'),('obj','step')]
        self.assertEqual(pair_kind(a,b,'subject')[0],2)

    def test_voice_change_is_not_strong_identity_control(self):
        a=row(0,'A man pulls.',1,'pull');b=row(1,'A woman is pulled.',2,'pull',subject='woman',media='other')
        a['signature']['identity_signature']=[('verb','pull')]
        b['signature']['identity_signature']=[('verb','pull'),('voice','passive')]
        self.assertEqual(pair_kind(a,b,'subject')[0],3)

if __name__=='__main__':unittest.main()
