"""Single-grounder RGB/text/point pseudo regression, no teacher and no GT API.
Experimental step sizes/budgets must be supplied explicitly by a registered run.
"""
import numpy as np
import torch
from camera_projection import project_model_points
from multimodal_pseudo import generate_pseudo, DEFAULT_CONFIG
from regression_adapter import capture_source, first_noun_mask, adapt, PseudoTarget


def source_fallback(cache, reason):
    source = cache.boxes_raw6.detach().clone()
    return {'query_index': cache.query_index,
            'source': source, 'direct_G': source.clone(),
            'parameter_SGD': source.clone(), 'exact_output_space_SGD': source.clone(),
            'trace': [], 'all_weights_zero': True, 'skip_reason': reason,
            'restored_selected_parameters': True, 'yaw': 0.}


def run_episode(*, model, points_model, rgb, actual_model_text, camera,
                step_config, g_config=DEFAULT_CONFIG,
                modes=('rgb_text', 'pc_only', 'shuffled_color')):
    """One P/text source forward followed by matched G and parameter controls.

    P is exactly the model's Nx3 input, not a box-cropped cloud. Camera z-buffer
    visibility is relative to these observed points, not an oracle visibility map.
    Caller supplies the exact model text, including sentinel if its protocol uses
    one. No GT/category/positive_map/target box is accepted by this interface.
    """
    assert isinstance(actual_model_text, str) and actual_model_text
    assert points_model.ndim == 3 and points_model.shape[0] == 1 and points_model.shape[-1] == 3
    assert isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    assert modes and len(set(modes)) == len(modes) and set(modes) <= {'rgb_text', 'pc_only', 'shuffled_color'}
    step_config.validate()
    tokenized = model.tokenizer.batch_encode_plus(
        [actual_model_text], padding='longest', return_tensors='pt', return_offsets_mapping=True)
    token_mask = first_noun_mask(actual_model_text, tokenized['offset_mapping'][0])
    cache = capture_source(model, {'point_clouds': points_model, 'text': [actual_model_text]}, token_mask)
    q = cache.query_index
    source_np = cache.boxes_raw6[0].cpu().numpy()
    scores = cache.text_scores[0].cpu().numpy()
    if not np.isfinite(source_np).all():
        raise FloatingPointError('Nonfinite source predictions; retain as failed inference')
    result = {'source_cache': cache, 'query_index': q, 'modes': {},
              'interpretation': 'Functional output only; task quality requires separately sealed evaluation.'}
    if np.any(source_np[q, 3:] <= 0):
        result['modes'] = {mode: {'adaptation': source_fallback(cache, 'nonpositive_source_size'),
                                  'pseudo': None} for mode in modes}
        return result
    points = points_model[0].detach().cpu().numpy()
    try:
        projected = project_model_points(points, camera, (rgb.shape[1], rgb.shape[0]))
    except ValueError as exc:
        result['modes'] = {mode: {'adaptation': source_fallback(cache, 'camera_contract_rejected: '+str(exc)),
                                  'pseudo': None} for mode in modes}
        return result
    result['camera_diagnostics'] = projected['diagnostics']
    for mode in modes:
        pseudo = generate_pseudo(points=points, rgb=rgb, text=actual_model_text,
                                 source_boxes=source_np, text_scores=scores, query_index=q,
                                 uv=projected['uv'], depth=projected['depth'], mode=mode, config=g_config)
        target = PseudoTarget(q, torch.as_tensor(pseudo['target6']),
                              torch.as_tensor(pseudo['reliability6']))
        updated = adapt(model, cache, target, step_config)
        result['modes'][mode] = {'pseudo': pseudo, 'adaptation': updated}
    return result
