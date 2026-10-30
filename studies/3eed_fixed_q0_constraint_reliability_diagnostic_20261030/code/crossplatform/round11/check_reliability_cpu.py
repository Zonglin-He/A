"""R11 necessary CPU synthetic checks; no real media/points/GT/models."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(D.parent/'round10'),str(D.parent/'round7')]
import reliability as rel
from visibility_geometry import generate_geometry_targets
from weighted_geometry import generate_weighted_geometry_targets
from camera_projection import CameraSpec

def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def main():
    K=np.array([[50.,0,64],[0,50.,64],[0,0,1.]]);E=np.eye(4);E[:3,:3]=np.diag([1.,-1.,-1.]);E[:3,3]=[0,0,10]
    cam=CameraSpec('quad',K,E,np.zeros(5));source=np.array([0,0,5,2,1,2.])
    P=np.array([(x,y,6.2) for x in np.linspace(-1,1,11) for y in np.linspace(-.5,.5,11)],np.float32);ids=np.arange(len(P))
    mask=np.zeros((128,128),bool);mask[56:73,49:80]=True
    kw=dict(points_fit=P,raw_indices=ids,source_box6=source,q0=7,camera_spec=cam,rgb_mask2d=mask,segmentation_valid=True)
    base=generate_geometry_targets(**kw);one=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(rel.FACES,1.),RGB_weight=1.)
    zero=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(rel.FACES,0.),RGB_weight=0.)
    for key in ('arm_boxes_raw6','pre_constraint_geometry6','acceptance_mask6','point_support_sha256'):
        assert np.array_equal(base[key],one[key]),key
    assert np.array_equal(zero['arm_boxes_raw6'],np.tile(source,(4,1)))
    assert np.array_equal(base['acceptance_mask6'],zero['acceptance_mask6'])
    half=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(rel.FACES,.5),RGB_weight=.5)
    assert 0<abs(half['arm_boxes_raw6'][2,2]-source[2])<abs(base['arm_boxes_raw6'][2,2]-source[2])
    # Physical points and coincident XYZ cannot leak between A/B or depend on array order.
    p,raw,part,diag=rel.physical_partitions(P,ids,'fixture');pp,rr,pa,dd=rel.physical_partitions(np.tile(P,(3,1)),np.tile(ids,3),'fixture')
    assert np.array_equal(p,pp) and np.array_equal(part,pa) and diag['partition_rawid_sha256']==dd['partition_rawid_sha256']
    order=np.random.default_rng(4).permutation(len(P));q,r,b,d=rel.physical_partitions(P[order],ids[order],'fixture');assert np.array_equal(q,p) and np.array_equal(b,part)
    # Face consistency ignores source residual: equal but jointly wrong medians remain high.
    f={'count':8,'defined':True,'median':3.,'MAD_m':0.,'tangent_span_m':np.array([1.,1.]),'normal':np.array([0.,0.,1.]),'mean_planarity':1.}
    stable=rel.face_consistency(f,f,f,2,source[3:]);assert stable['score']==1
    shifted=dict(f,median=3.15);unstable=rel.face_consistency(f,f,shifted,2,source[3:]);assert unstable['score']<.01
    sparse=dict(f,count=2);assert not rel.face_consistency(f,f,sparse,2,source[3:])['defined']
    # Fragmentation/depth layering alone is not an incompatible-object assertion.
    groups=np.r_[np.array([[0,0,0],[.01,0,0],[.02,0,0]]),np.array([[.2,0,0],[.21,0,0],[.22,0,0]])]
    cp=rel.component_ambiguity(groups,np.arange(6.),[1,1,1]);assert cp['component_count']==2 and not cp['conflict_observed'] and cp['ambiguity_factor']==1
    far=groups.copy();far[3:,0]+=3;cf=rel.component_ambiguity(far,np.arange(6.),[1,1,1]);assert cf['conflict_observed'] and cf['ambiguity_factor']==0
    singletons=np.arange(20.)[:,None]*np.array([[1,1,1.]])
    cs=rel.component_ambiguity(singletons,np.arange(20.),[1,1,1]);assert cs['substantial_component_count']==0 and not cs['conflict_observed']
    # Consistent masks alone may give score1 despite not proving identity.
    ms=rel.mask_stability(mask,mask,3);assert ms['score']==1 and rel.mask_stability(mask,np.zeros_like(mask),3)['score']==0
    # Integration runs artificial RGB only. Source6.2 plane has no within-q0 seed:
    # undefined evidence must survive as explicit NA and zero observational weight.
    image=np.zeros((128,128,3),np.uint8);image[56:73,49:80]=[220,60,30]
    result=rel.compute_reliability(sample_id='fixture',points=P,raw_indices=ids,rgb=image,source_box6=source,camera_spec=cam,base_mask2d=mask,base_result=base)
    assert 0<=result['RGB_weight']<=1 and all(0<=v<=1 for v in result['face_weights'].values())
    assert result['association_status'] in ('association_supported','association_conflicted','association_undefined')
    assert result['case_reliability'] is None and result['RGB_weight']==0
    json.dumps(clean(result),allow_nan=False)
    # Actual-mask path inside source with a known appearance patch, no data files.
    inside=P.copy();inside[:,2]=5.9;kw2={**kw,'points_fit':inside};base2=generate_geometry_targets(**kw2)
    ri=rel.compute_reliability(sample_id='fixture2',points=inside,raw_indices=ids,rgb=image,source_box6=source,camera_spec=cam,base_mask2d=mask,base_result=base2)
    json.dumps(clean(ri),allow_nan=False)
    assert ri['features']['partition_segmentation_calls']==2
    # Ranking NA-last, exact platform K and deterministic control selection.
    units=[{'id':str(i),'platform':'p','eligible':True,'scores':{'reliability':None if i==0 else float(i),'point_count':i}} for i in range(5)]
    selection=rel.select_exact_half(units,'reliability');assert len(selection['selected_ids'])==3 and '0' not in selection['selected_ids']
    assert rel.select_exact_half(units,'hash_random')==rel.select_exact_half(units[::-1],'hash_random')
    # Correspondence precedes fusion; raw useful-looking weights cannot leak.
    import copy
    template={'face_weights':dict.fromkeys(rel.FACES,.8),'RGB_weight':.7,'RGB_weight_defined':True,'axis_reliability6':np.array([.8,.2,0,0,0,0]),'axis_reliability_defined6':np.array([True,False,False,False,False,False]),'association_status':'association_supported','features':{'faces':{key:{'defined':True} for key in rel.FACES}},'control_scores':{},'undefined_reasons':[]}
    ac=np.array([True,True,False,False,False,False]);changed=np.array([True,False,False,False,False,False])
    supported=rel.apply_correspondence_gate(copy.deepcopy(template),ac,changed)
    assert supported['case_reliability']==.8  # accepted-but-unchanged undefined axis is excluded
    conflict=copy.deepcopy(template);conflict['association_status']='association_conflicted';conflict=rel.apply_correspondence_gate(conflict,ac,changed)
    assert conflict['raw_RGB_weight']==.7 and conflict['RGB_weight']==0 and not any(conflict['face_weights'].values()) and conflict['case_reliability']==0
    assert conflict['axis_reliability_defined6'][changed].all() and not conflict['undefined_reasons']
    undefined=copy.deepcopy(template);undefined['association_status']='association_undefined';undefined=rel.apply_correspondence_gate(undefined,ac,changed)
    assert undefined['case_reliability'] is None and not undefined['axis_reliability_defined6'].any() and undefined['RGB_weight']==0
    assert undefined['raw_face_weights']==template['face_weights']
    # RGB_weight0 is an absent term, not a hidden geometry veto or loss.
    rgb0=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(rel.FACES,1.),RGB_weight=0.)
    assert np.array_equal(rgb0['arm_boxes_raw6'][1],base['arm_boxes_raw6'][2])
    receipt={'state':'PASS_CPU_synthetic_only_not_reliability_quality_result','correspondence_gate_preserves_raw_but_zeros_unsupported_effective_weights':True,'case_min_only_actual_float32_changed_axes':True,'defined_zero_distinct_from_NA':True,'RGBzero_no_hidden_geometry_veto':True,'weighted_r1_bitexact_R10_boxes_preconstraint_acceptance_support':True,'all_observation_r0_exact_source_with_same_acceptance':True,'uniform_half_weights_attenuate_without_weight_renormalization':True,'physical_spatial_partition_dedup_and_permutation_invariant':True,'stable_jointly_wrong_plane_counterexample_retained':True,'sparse_fragmentation_not_confident_conflict':True,'incompatible_large_components_can_reduce_RGB_weight':True,'undefined_evidence_NA_last_zero_weights':True,'exact_samecoverage_per_platform_hash_ties':True,'no_real_model_GPU_GT_media_or_points_used':True,'config':asdict(rel.DEFAULT_CONFIG),'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('reliability.py','weighted_geometry.py','check_reliability_cpu.py')}}
    (D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
