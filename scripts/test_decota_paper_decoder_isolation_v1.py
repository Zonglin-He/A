"""CPU regression checks for same-process HC/Vid decoder contamination."""
import sys
from pathlib import Path
from collections import OrderedDict
from unittest import TestCase, main
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_decota_paper_main_v1 import frames_for
from vg_tta import exact_frame_decode_audit_v2 as vid
from vg_tta import tastvg_paper48_hc2_decode_v1 as hc


class DecoderIsolation(TestCase):
    def check_order(self, order):
        calls = []

        def vid_decode(query):
            calls.append('vid')
            return np.zeros((2, 1, 1, 3), np.uint8), [0, 1]

        def hc_decode(query):
            calls.append('hc')
            return np.ones((2, 1, 1, 3), np.uint8), [0, 1]

        row = {'input': {'video_path': 'synthetic_shared_video',
                        'video_sha256': 'synthetic', 'frame_ids': [0, 1],
                        'width': 1, 'height': 1}, 'frame_ids': [0, 1]}
        cache = OrderedDict()
        with patch.object(vid, 'decode', vid_decode), patch.object(hc, 'decode', hc_decode):
            for repeat in (False, True):
                for dataset in order:
                    frames, ids, reused = frames_for(dataset, row, cache)
                    self.assertEqual(reused, repeat)
                    self.assertEqual(ids, [0, 1])
                    self.assertTrue(np.all(frames == (1 if dataset == 'hc2' else 0)))
                    self.assertIs(vid.decode, vid_decode)
            self.assertEqual(len(cache), 2)
        self.assertEqual(calls, ['hc' if x == 'hc2' else 'vid' for x in order])

    def test_hc_then_vid_same_metadata(self):
        self.check_order(('hc2', 'vidstg'))

    def test_vid_then_hc_same_metadata(self):
        self.check_order(('vidstg', 'hc2'))


if __name__ == '__main__':
    main()
