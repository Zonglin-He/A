"""CPU integration contracts for the actual installed native image/video processors."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_qwen35_geometry_run_v1 as run
from scripts import stvg_qwen35_geometry_adapters_v1 as adapters

class NativeContracts(unittest.TestCase):
    def test_native_processors_preserve_all_four_frames(self):
        import numpy as np,torch
        from transformers import Qwen2VLImageProcessor,Qwen3VLVideoProcessor
        from transformers.video_utils import VideoMetadata
        from PIL import Image
        ip=Qwen2VLImageProcessor(patch_size=16,merge_size=2,temporal_patch_size=2);vp=Qwen3VLVideoProcessor()
        for H,W in [(160,320),(320,160)]:
            image=Image.fromarray(np.zeros((H,W,3),np.uint8));b=ip(images=[image],return_tensors='pt');grid=b['image_grid_thw'][0].tolist()
            self.assertEqual(grid,[1,H//16,W//16]);self.assertTrue(torch.isfinite(b['pixel_values']).all())
        frames=torch.stack([torch.full((3,448,448),i*50,dtype=torch.uint8) for i in range(4)])
        meta=VideoMetadata(total_num_frames=4,fps=.5,frames_indices=[0,1,2,3],duration=8.)
        b=vp(videos=[frames],do_sample_frames=False,video_metadata=[meta],return_metadata=True,return_tensors='pt')
        self.assertEqual(b['video_grid_thw'][0,0].item(),2)
        self.assertEqual(b['video_metadata'][0].frames_indices,[0,1,2,3]);self.assertEqual(b['video_metadata'][0].fps,.5)
        self.assertTrue(torch.isfinite(b['pixel_values_videos']).all())
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_partial_prediction_set_can_open_GT_stage(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(adapters,'BASE',Path(temp)),patch.object(adapters,'execution_gate',return_value={}):
            # A missing OPD or propagation root stops before any GT helper is imported/called.
            with self.assertRaises(FileNotFoundError):adapters.require_prediction_seals_before_score()

    def test_runtime_pin_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'BASE',Path(temp)),patch.object(run,'ROOT',Path(temp)):
            p=Path(temp);(p/'bound.py').write_text('fixed');digest=run.sha(p/'bound.py')
            (p/'EXECUTION_RUNTIME001.json').write_text(json.dumps(dict(pins={'bound.py':digest})))
            run.verify_run();(p/'bound.py').write_text('changed')
            with self.assertRaises(AssertionError):run.verify_run()

    def test_science_deadline_not_reset_by_reentry(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'BASE',Path(temp)):
            p=Path(temp);(p/'EXECUTION_RUNTIME001.json').write_text(json.dumps(dict(pins={})))
            (p/'ACTIVE_EXECUTION.json').write_text(json.dumps(dict(deadline_epoch=1)))
            with self.assertRaisesRegex(AssertionError,'wall budget'):run.verify_run()

if __name__=='__main__':unittest.main()
