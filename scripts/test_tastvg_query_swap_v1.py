"""Meaningful intervention, geometry-control and aggregation checks on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scripts.tastvg_query_swap_math_v1 import donor_map, caption_norm
from scripts.tastvg_information_atlas_math_v1 import (
    frame_labels, candidate_labels, moments, paired_difference, bootstrap_summary)
from scripts.tastvg_latent_quality_math_v1 import interval_features, geometry_features


class Checks(unittest.TestCase):
    def test_donor(self):
        rows = [dict(source=str(i), input=dict(video_sha256=str(i), caption=f'person {i}')) for i in range(8)]
        a, offset = donor_map(rows, 'test'); b, _ = donor_map(rows, 'test')
        self.assertEqual(a, b); self.assertEqual(len(set(a.values())), 8)
        self.assertTrue(all(i != d for i, d in a.items())); self.assertGreater(offset, 0)
        self.assertEqual(caption_norm(' A  PERSON\n'), 'a person')
        rows[0]['input']['caption'] = rows[1]['input']['caption']
        a, _ = donor_map(rows, 'test')
        self.assertTrue(all(caption_norm(rows[i]['input']['caption']) != caption_norm(rows[d]['input']['caption']) for i,d in a.items()))

    def test_impossible_donor(self):
        with self.assertRaises(ValueError):
            donor_map([dict(source=str(i),input=dict(video_sha256=str(i),caption='same')) for i in range(4)], 'test')

    def test_fixed_support(self):
        ids = [0,2,4,6]; pairs=[(0,1),(1,3)]; h=np.arange(4*256).reshape(4,256)
        a, ctx = interval_features(h, ids, pairs, 2)
        b, ctx2 = interval_features(h+1, ids, pairs, 2)
        self.assertFalse(np.array_equal(a,b));self.assertEqual(ctx,ctx2)
        np.testing.assert_array_equal(geometry_features(ids,pairs),geometry_features(ids,pairs))
        labels=candidate_labels([[0,3],[2,7]],[2,6],0)
        np.testing.assert_allclose(labels['precision'],[1/3,4/5])
        np.testing.assert_allclose(labels['recall'],[1/4,1])
        np.testing.assert_array_equal(frame_labels(ids,[2,6])['event'],[0,1,1,0])

    def test_paired_source(self):
        rows=[]
        for s in range(3):
            for c in ['blur_5','freeze_5']:
                y=np.array([0.,.5,1.]);p=y+.05*s
                rows.append(dict(source_index=s,order='order0',condition=c,
                    metrics={'true':moments(y,p),'swap':moments(y,p+.2)}))
        d=paired_difference(rows,'true','swap','r2',1000)
        self.assertGreater(d['mean'],0);self.assertGreater(d['ci95'][0],0)
        z=paired_difference(rows,'true','true','r2',1000)
        self.assertEqual(z['mean'],0);self.assertEqual(z['ci95'],[0.,0.])

    def test_event_undefined_and_r2(self):
        self.assertIsNone(moments([1,1],[.5,.6],True)['auc'])
        self.assertIsNone(moments([.2,.2],[.3,.3])['within_r2'])
        v=moments([0,1],[.1,.9],True,[-2,2])
        self.assertEqual(v['auc'],1.);self.assertGreater(v['logloss'],0)

if __name__ == '__main__':unittest.main()
