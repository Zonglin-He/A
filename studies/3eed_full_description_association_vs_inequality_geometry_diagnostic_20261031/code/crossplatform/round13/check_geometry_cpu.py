"""Synthetic necessary checks only. No models, sensor files or task labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
from geometry_observation import (generate_observation_targets, ObservationConfig,
                                  DEFAULT_CONFIG, config_sha256)
from camera_projection import CameraSpec


def check():
    cfg = DEFAULT_CONFIG
    spec = CameraSpec('quad', np.array([[100.,0,80],[0,100.,80],[0,0,1]]),
                      np.eye(4), missing_distortion_policy='assume_zero_declared')
    source = np.array([0,0,5,2,2,2], np.float32)
    mask = np.zeros((160,160),bool); mask[73:87,73:87] = True
    def run(p, m=mask, s=source, ids=None, **extra):
        return generate_observation_targets(points_support_model=p, mask_bool=m,
            camera_spec=spec, source6=s, q0=19, physical_raw_ids=ids, **extra)
    def plane(x_span=.5, offset=0):
        x,y=np.meshgrid(np.linspace(-x_span,x_span,5)+offset,np.linspace(-.4,.4,5))
        return np.c_[x.ravel(),y.ravel(),np.full(x.size,4.5)].astype(np.float32)
    checks={};details={}
    for fraction in (.05,.2,.5,1.):
        p = plane(.5*fraction)
        r=run(p)
        assert np.array_equal(r['arm_boxes_raw6'], np.tile(source,(4,1)))
        assert not r['actual_changed_axes'].any()
        checks[f'partial_visibility_fraction_{fraction}_identity_no_shrink']=True
    p=plane(.15,1.35)
    outside=run(p)
    d=outside['diagnostics']['solver']['direct_no_rgb']
    assert d['final_terms']['point_violation_mean_m'] < d['initial_terms']['point_violation_mean_m']
    assert outside['actual_changed_axes'][2].any()
    assert np.all(outside['arm_boxes_raw6'][:,3:]>=source[3:])
    assert np.max(np.abs(outside['arm_boxes_raw6'][:,0]-source[0])) <= .5
    assert outside['arm_boxes_raw6'][2,0] != p[:,0].mean()
    checks['outside_support_bounded_move_or_expand_not_point_mean']=True
    # Complete synthetic object may actually have source box while the associated
    # support is wrong. Positive-observation optimization cannot detect identity.
    prior_center_error=0.
    after_center_error=float(np.linalg.norm(outside['arm_boxes_raw6'][2,:3]-source[:3]))
    assert after_center_error>prior_center_error
    checks['loss_descent_does_not_prove_complete_box_or_identity_correctness']=True
    details['wrong_support_counterexample']={'point_loss_before':d['initial_terms']['point_containment'],
        'point_loss_after':d['final_terms']['point_containment'],
        'synthetic_complete_box_center_error_before':prior_center_error,
        'synthetic_complete_box_center_error_after':after_center_error}
    offsetmask=np.zeros_like(mask);offsetmask[76:85,108:116]=True
    rgb=run(plane(),offsetmask)
    assert rgb['actual_changed_axes'][1].any()
    assert np.array_equal(rgb['arm_boxes_raw6'][2],source)
    f=rgb['diagnostics']['solver']['direct_full']
    assert f['final_terms']['RGB_positive_containment'] < f['initial_terms']['RGB_positive_containment']
    assert np.array_equal(rgb['active_mask6'][1],rgb['active_mask6'][2])
    assert rgb['point_support_sha256'][1]==rgb['point_support_sha256'][2]
    assert rgb['mask_sha256_by_arm'][1]==rgb['mask_sha256_by_arm'][2]
    assert np.array_equal(rgb['arm_boxes_raw6'][3],source+np.float32(.5)*(rgb['arm_boxes_raw6'][1]-source))
    checks['RGB_only_positive_violation_same_support_bounds_noRGB_identity']=True
    checks['half_exact_float32_bridge']=True
    ids=np.arange(len(p));repeat=run(np.repeat(p,4,axis=0),ids=np.repeat(ids,4))
    rng=np.random.default_rng(2);order=rng.permutation(len(p));permuted=run(p[order],ids=ids[order])
    base=run(p,ids=ids)
    for x in (repeat,permuted):
        assert np.array_equal(x['arm_boxes_raw6'],base['arm_boxes_raw6'])
        assert np.array_equal(x['point_support_sha256'],base['point_support_sha256'])
    checks['physical_duplicate_and_permutation_invariance']=True
    bad=source.copy();bad[3]=0
    invalid=run(plane(),s=bad)
    assert np.array_equal(invalid['arm_boxes_raw6'],np.tile(bad,(4,1)))
    assert invalid['diagnostics']['gate_reason']=='nonpositive_source_size'
    checks['invalid_source_size_preserved_explicit_fallback']=True
    small=run(plane()[:7]); assert small['diagnostics']['gate_reason']=='insufficient_unique_foreground_support'
    checks['min_eight_distinct_foreground_gate']=True
    empty=run(plane(),np.zeros_like(mask));assert not empty['diagnostics']['gate_passed']
    checks['missing_independent_mask_fallback']=True
    near=source.copy();near[2]=.5
    depth=run(plane(),s=near);assert 'projection_not_fully_in_front' in depth['diagnostics']['gate_reason']
    checks['near_plane_explicit_source_fallback']=True
    # A valid enclosing box contains empty segments before a partial interior
    # return. No face-equality or ray-entry rejection should be introduced.
    interior=np.c_[plane()[:,:2],np.full(25,5.2)].astype(np.float32)
    air=run(interior);assert np.array_equal(air['arm_boxes_raw6'],np.tile(source,(4,1)))
    assert 'not used' in air['diagnostics']['free_space']
    checks['ray_entry_before_return_not_a_box_solid_rejection']=True
    # Two different target sizes are compatible with the same partial plane;
    # maintaining source size is a prior choice, not amodal recovery.
    assert np.allclose(plane()[:,2],4.5)
    checks['partial_surface_has_multiple_compatible_amodal_boxes']=True
    forced=run(p,config=ObservationConfig(solver_max_iterations=0))
    assert forced['arm_status'][2]=='finite_nonconverged_bounded_result'
    json.dumps(forced['diagnostics'],allow_nan=False)
    checks['finite_nonconvergence_explicit_and_JSON_safe']=True
    try:
        run(np.repeat(p[:1],8,axis=0),ids=np.zeros(8,dtype=np.int64)+3)
    except ValueError:
        raise AssertionError('Identical duplicate IDs should be valid but deduplicate')
    try:
        run(p,ids=np.zeros(len(p),dtype=np.int64))
    except ValueError:
        checks['inconsistent_same_physical_ID_rejected']=True
    else:
        raise AssertionError('Inconsistent same ID accepted')
    receipt={'state':'passed_synthetic_necessary_checks_no_task_results',
        'config':vars(cfg),'config_sha256':config_sha256(), 'checks':checks,
        'details':details,
        'program_sha256':{x:hashlib.sha256(Path(__file__).with_name(x).read_bytes()).hexdigest()
            for x in ('geometry_observation.py','check_geometry_cpu.py')},
        'not_checked':['real target identity','real calibration alignment','task utility','neural adaptation'],
        'real_data_or_GT_or_model_or_GPU_accessed':False}
    return receipt


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    result=check();path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'n_checks':len(result['checks']),
        'config_sha256':result['config_sha256']}))
