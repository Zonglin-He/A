"""Pure diagnostic fixture for original R23 geometry/relationship fidelity.

This fixture is NOT a final-safe feedback input. The caller may supply old-mask
geometry/old-reference weights only under its separate diagnostic authorization.
No cache/media/model access occurs on import. Supplied descriptions are inert.
"""
import ast
from pathlib import Path
import numpy as np
from evidence51_fidelity import raw_relations

ROOT=Path(__file__).resolve().parents[2]


def _original_functions():
    namespace={'np':np}
    for rel,names in (('crossplatform/round20/detect.py',{'rect_iou'}),
                      ('crossplatform/round23/rank23.py',{'sigmoid','relation_value'})):
        module=ast.parse((ROOT/rel).read_text())
        nodes=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes}!=names:raise ValueError('Original function extraction mismatch')
        # Only pinned, manually reviewed repository code is compiled; no data,
        # captions, graph strings or model output are ever evaluated as code.
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),rel,'exec'),namespace)
    return namespace


def compare_legacy_fixture(graph,mask_objects,old_reference_evidence,image_size_wh,expected_raw_pr=None):
    """Compare literal old function with new typed interface, no model/GT.

    With camera-image global relations and left/right reference predicates,
    and the same frame/self-overlap semantics, raw P/R should match exactly.
    Other frames/predicates are returned NOT_COMPARABLE rather than changing
    the new UNKNOWN semantics to force a positive fixture result.
    """
    W,H=image_size_wh
    frame_ok=all(r.get('frame')=='camera_image' for r in graph.get('global_relations',[]))
    predicates_ok=all(r.get('predicate') in ('LEFT_OF','RIGHT_OF') for r in graph.get('reference_relations',[])
                      if r.get('frame')=='camera_image')
    if not frame_ok or not predicates_ok:
        return {'state':'NOT_COMPARABLE_UNKNOWN_FRAME_OR_PREDICATE',
                'all_global_camera_image':frame_ok,'reference_predicates_left_right':predicates_ok}
    old=_original_functions()
    objects=[]
    for i,obj in enumerate(mask_objects):
        visible=bool(obj['mask_geometry_defined'])
        objects.append({'visible':visible,'border':bool(obj['touches_image_border']),
            'xy':obj['mask_centroid_norm2'] if visible else None,
            'rect':np.asarray(obj['detector_box_xyxy'],float),
            'domain':'diagnostic_legacy_mask','instance_index':i})
    refs={}
    for key,items in old_reference_evidence['references'].items():
        refs[key]=[]
        for a in items:
            box=np.asarray(a['box_PRIVATE'])/np.array([W,H,W,H])
            weight=float(old['sigmoid'](a['full']['log_odds_yes_no'])*old['sigmoid'](a['visual_evidence'])*a['detector_score'])
            refs[key].append({'xy_norm2':[float((box[0]+box[2])/2),float((box[1]+box[3])/2)],
                'width_norm':float(box[2]-box[0]),'rect':a['box_PRIVATE'],
                'probability':weight,'instance_domain':'diagnostic_reference_proposal',
                'uncertainty_mass_explicit':False})
    baseline=np.zeros((len(mask_objects),2),float)
    for i,obj in enumerate(mask_objects):
        if obj['mask_geometry_defined']:
            baseline[i]=old['relation_value'](graph,obj,old_reference_evidence,W,H)
    current,known=raw_relations(graph,objects,refs,(W,H))
    difference=np.abs(current-baseline)
    reference_byteexact=(current.shape==baseline.shape and current.dtype==baseline.dtype and current.tobytes()==baseline.tobytes())
    saved_bridge=None
    if expected_raw_pr is not None:
        saved=np.asarray(expected_raw_pr)
        saved_bridge=bool(saved.shape==baseline.shape and saved.dtype==baseline.dtype and saved.tobytes()==baseline.tobytes())
    return {'state':'PASS' if reference_byteexact and saved_bridge is not False else 'FAILED_EXACT_RELATION_FIDELITY',
        'n_objects':len(mask_objects),'all_global_camera_image':frame_ok,
        'reference_predicates_left_right':predicates_ok,
        'literal_original_vs_new_byteexact':reference_byteexact,
        'literal_original_vs_saved_raw_PR_byteexact':saved_bridge,
        'max_abs_difference':float(difference.max()) if difference.size else 0.,
        'known_cues':known.tolist(),'baseline':baseline,'current':current}


def synthetic_fixture():
    """No sample/caption/cache access; root may execute under its CPU guard."""
    graph={'global_relations':[{'frame':'camera_image','predicate':'GLOBAL_LEFT'}],
           'reference_relations':[{'subject':'R0','frame':'camera_image','predicate':'RIGHT_OF'}]}
    objects=[{'mask_geometry_defined':True,'touches_image_border':False,
              'mask_centroid_norm2':[.25,.5],'detector_box_xyxy':np.array([10,20,30,40],np.float32)},
             {'mask_geometry_defined':True,'touches_image_border':True,
              'mask_centroid_norm2':[.25,.5],'detector_box_xyxy':np.array([0,20,20,40],np.float32)}]
    evidence={'references':{'R0':[{'box_PRIVATE':np.array([60,20,80,40],np.float32),
            'full':{'log_odds_yes_no':1.25},'visual_evidence':.75,'detector_score':.7}]}}
    result=compare_legacy_fixture(graph,objects,evidence,(100,80))
    if result['state']!='PASS':raise AssertionError(result)
    return result
