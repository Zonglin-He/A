"""Verify the restored subject output on all directions and the source holdouts."""

import numpy as np
import torch

from evaluate import PRIVATE, observation
from features import FEATURES
from geometry_readout_check import ARMS, OUT
from retained_geometry import RetainedGeometryGrounder, retained_subject
from scope import HERE, PLATFORMS, evaluation_rows, file_sha, protocol, read, save
from train import decisions, many_iou, metrics, read_features, take


@torch.inference_mode()
def run():
    rows = evaluation_rows()
    replay = read(OUT/'SEALED.json')
    prediction_pin = read(PRIVATE/'SEALED.json')
    assert replay['original_prediction_seal'] == file_sha(PRIVATE/'SEALED.json')
    ns, exact, max_error = 0, 0, 0.
    source_result = {}
    arms = protocol()['comparisons']
    for source in PLATFORMS:
        model = RetainedGeometryGrounder(source)
        for target in PLATFORMS:
            if source == target:
                continue
            key = f'{source}_to_{target}.npz'
            f, p = OUT/key, PRIVATE/key
            assert file_sha(f) == replay['files'][key]['sha256']
            assert file_sha(p) == prediction_pin['files'][key]['sha256']
            with np.load(f,allow_pickle=False) as saved, np.load(p,allow_pickle=False) as base:
                for j, ordinal in enumerate(saved['ordinals'][:2]):
                    prediction = model.predict(**observation(rows[int(ordinal)]))
                    expected = saved['raw'][j,ARMS.index('source_selected')]
                    actual = np.array(prediction['subject_box'])
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                    exact += int(np.array_equal(actual,expected))
                    max_error = max(max_error,float(np.abs(actual-expected).max()))
                    arm = arms.index('source_selected')
                    if base['reference_known'][j,arm]:
                        np.testing.assert_array_equal(prediction['reference_box'],base['referents'][j,arm])
                    else:
                        assert prediction['reference_box'] is None
                    ns += 1
        holdout = read_features(FEATURES,source,'calibration')
        index = torch.nonzero(holdout['origin'] < 2,as_tuple=True)[0]
        ious = []
        for chunk in index.split(8):
            batch = take(holdout,chunk)
            out = model.head(batch)
            prediction,_,_,ids = decisions(out,batch)
            method = model.source_readout
            boxes=torch.stack((batch['v1_box'],prediction[method],retained_subject(out,batch,ids[method])),1)
            for box,target_box in zip(boxes.cpu().numpy(),batch['gt'][:,0].cpu().numpy()):
                ious.append(many_iou(box,target_box))
        ious = np.stack(ious)
        source_result[source] = dict(original_selection=method,
            **{name:metrics(values) for name,values in zip(('v1','earlier','retain_original'),ious.T)})
        del model,holdout
        torch.cuda.empty_cache()
    assert ns == 12
    save(HERE/'results/retained_geometry_runtime.json',dict(
        target_runtime_checks=ns,bit_exact_subject_boxes=exact,max_subject_abs_error=max_error,
        reference_outputs_unchanged=ns,source_holdouts=source_result,
        sealed_replay=file_sha(OUT/'SEALED.json'),implementation=file_sha(HERE/'retained_geometry.py'),
        target_labels_used_by_runtime_check=False,FINAL_access=False))


if __name__=='__main__':
    run()
