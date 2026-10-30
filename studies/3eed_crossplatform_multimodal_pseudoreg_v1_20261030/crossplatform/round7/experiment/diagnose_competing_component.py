"""Bounded CPU-only unlabeled component-overlap diagnostic, not a new method.
Reads only sealed predictions/trace, allowlisted inputs and camera whitelist.
Never imports torch/model/scorer or target GT; no threshold/method change.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from runner_helpers import sha,read,validate_manifest,sample_points,array_sha,to_json
HERE=Path(__file__).resolve().parent;R7=HERE.parent;sys.path.insert(0,str(R7))
SALT='R7_same_component_diagnostic_v1'
PURPOSE='R7_unlabelled_same_component_diagnostic'
MAX_PER_PLATFORM=8
OVERLAP=.8


def selection(trace):
    selected=[];eligible={}
    for platform in ('drone','quad'):
        rows=[r for r in trace if r['platform']==platform and r['modes']['rgb_text']['reason']=='same_evidence_competing_candidate']
        eligible[platform]=len(rows)
        key=lambda r:hashlib.sha256((SALT+'|'+platform+'|'+r['id']).encode()).hexdigest()
        selected.extend(sorted(rows,key=key)[:MAX_PER_PLATFORM])
    return selected,eligible


def rowset(points):return {tuple(float(v) for v in row) for row in points}


def overlap_metrics(component,alt,global_components,min_support):
    source_set=rowset(component);alt_set=rowset(alt);shared=source_set&alt_set
    independent=[]
    for comp in global_components:
        cset=rowset(comp)
        if cset&source_set:continue
        n=len(cset&alt_set)
        if n:independent.append(n)
    a=len(shared)/len(source_set) if source_set else 0.
    b=len(shared)/len(alt_set) if alt_set else 0.
    supported=[n for n in independent if n>=min_support]
    high=a>=OVERLAP and b>=OVERLAP
    return {'source_component_points':len(source_set),'alt_supported_unique_points':len(alt_set),
            'shared_sourcecomponent_alt_points':len(shared),
            'source_component_covered_by_alt_fraction':a,
            'alt_support_from_source_component_fraction':b,
            'extra_alt_points_outside_source_local_component':len(alt_set-source_set),
            'extra_alt_points_in_global_components_disconnected_from_source':sum(independent),
            'extra_independent_global_components_with_any_alt_points':len(independent),
            'extra_independent_global_components_meeting_min_support':len(supported),
            'extra_alt_points_in_supported_independent_components':sum(supported),
            'high_overlap_both_ge_0p8':high,
            'high_overlap_and_no_extra_supported_independent_component':high and not supported}



def global_components_exact(points,radius):
    """Same Euclidean graph as G, pruning already visited bucket occupants.
    Pure engineering acceleration for full visible color graph; G is unchanged.
    """
    import itertools
    p=np.unique(points,axis=0)
    if not len(p):return []
    cells=np.floor(p/radius).astype(np.int64);buckets={}
    for i,c in enumerate(cells):buckets.setdefault(tuple(c),set()).add(i)
    remaining=set(range(len(p)));out=[]
    while remaining:
        first=min(remaining);remaining.remove(first);buckets[tuple(cells[first])].remove(first)
        queue=[first];members=[]
        while queue:
            i=queue.pop();members.append(i)
            for delta in itertools.product((-1,0,1),repeat=3):
                bucket=buckets.get(tuple(cells[i]+delta))
                if not bucket:continue
                ids=np.array(sorted(bucket),dtype=np.int64)
                hits=ids[np.linalg.norm(p[ids]-p[i],axis=1)<=radius]
                for j in hits:
                    j=int(j);bucket.remove(j);remaining.remove(j);queue.append(j)
        out.append(p[sorted(members)])
    return sorted(out,key=lambda x:(-len(x),tuple(x[0])))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['purpose']==PURPOSE and plan['CPU_execution_authorized'] is True
    assert plan['max_per_platform']==MAX_PER_PLATFORM and plan['selection_salt']==SALT and plan['high_overlap_fraction']==OVERLAP
    for rel,digest in plan['diagnostic_program_sha256'].items():
        p=R7/rel;assert p.resolve().is_relative_to(R7.resolve()) and sha(p)==digest
    seal_path=Path(plan['full_prediction_seal_path']);assert sha(seal_path)==plan['full_prediction_seal_sha256'];seal=read(seal_path)
    assert seal['phase']=='full' and seal['n_samples']==5912 and seal['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
    outdir=seal_path.parent;trace_path=outdir/'trace.json';assert sha(trace_path)==seal['files']['trace.json']
    manifest_path=Path(plan['manifest_path']);assert sha(manifest_path)==seal['manifest_sha256']==plan['manifest_sha256']
    manifest={r['id']:r for r in validate_manifest(read(manifest_path))};trace=read(trace_path);assert len(trace)==5912
    for rel,digest in seal['program_sha256'].items():assert sha(R7/rel)==digest
    sourceplan=read(outdir/'registered_plan.json');assert sha(outdir/'registered_plan.json')==seal['plan_sha256']
    selected,eligible=selection(trace);assert len(selected)<=16
    assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    # Selection is committed before reading any selected P/RGB/camera/cache.
    selectedfile=a.output/'SELECTED_PRIVATE.json'
    selectedfile.write_text(json.dumps({'selection_salt':SALT,'max_per_platform':8,'eligible_by_platform':eligible,'selected':[{'id':x['id'],'platform':x['platform']} for x in selected],'no_GT_or_IoU_selection':True,'full_prediction_seal_sha256':sha(seal_path)},indent=2)+'\n')
    from PIL import Image
    from camera_projection import read_camera_allowlist,CameraSpec,project_model_points
    from multimodal_pseudo import (DEFAULT_CONFIG,target_color,visible_points,hsv_support,in_box,components,aabb_iou,generate_pseudo)
    started=time.perf_counter();cases=[]
    for old in selected:
        row=manifest[old['id']];cp=outdir/(row['id']+'.npz');assert sha(cp)==seal['files'][cp.name]
        with np.load(cp,allow_pickle=False) as saved:
            boxes=saved['source_boxes_raw6'].astype(np.float64);scores=saved['source_text_scores256'].astype(np.float64);q=int(saved['fixed_query_index']);point_hash=saved['point_sample_sha256'].tobytes().hex()
        pc,ix,_=sample_points(row);assert array_sha(pc)==point_hash==old['point_sample_sha256'];assert array_sha(ix)==old['sample_indices_sha256']
        assert sha(row['pointpath'])==old['point_file_sha256'] and sha(row['imagepath'])==old['image_file_sha256'] and sha(row['metapath'])==old['camera_metadata_file_sha256']
        with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB'),np.uint8).copy()
        camera=CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(row['metapath']),waymo_view_suffix=0,missing_distortion_policy=sourceplan['missing_distortion_policy'])
        projection=project_model_points(pc,camera,(rgb.shape[1],rgb.shape[0]))
        text=row['caption']+sourceplan['sentinel'];color,reason=target_color(text);assert color is not None
        reproduced=generate_pseudo(points=pc,rgb=rgb,text=text,source_boxes=boxes,text_scores=scores,query_index=q,uv=projection['uv'],depth=projection['depth'],mode='rgb_text')
        diag=reproduced['diagnostics'];old_diag=old['modes']['rgb_text']['pseudo_diagnostics']
        assert diag==old_diag and diag['reason']=='same_evidence_competing_candidate'
        altq=int(diag['ambiguous_other_query']);cfg=DEFAULT_CONFIG
        visible,pixels=visible_points(pc.astype(np.float64),rgb,projection['uv'],projection['depth'])
        supported=visible[hsv_support(pixels,color,cfg)]
        source_crop=supported[in_box(supported,boxes[q],cfg.crop_margin_m)]
        source_components=components(source_crop,cfg.component_radius_m);assert len(source_components[0])==diag['component_sizes'][0]
        source_component=source_components[0]
        alt=np.unique(supported[in_box(supported,boxes[altq],cfg.crop_margin_m)],axis=0)
        # Global RGB-color graph distinguishes points merely outside the source
        # crop from a genuinely disconnected *geometric* observation component.
        global_components=global_components_exact(supported,cfg.component_radius_m)
        metrics=overlap_metrics(source_component,alt,global_components,cfg.min_component_points)
        cases.append({'id':row['id'],'platform':row['platform'],'scene':row['scene'],'fixed_source_query':q,'recorded_alt_query':altq,'source_vs_alt_AABB_IoU_no_GT':aabb_iou(boxes[q],boxes[altq]),'source_text_score':float(scores[q]),'alt_text_score':float(scores[altq]),'reproduced_G_diagnostics_bitexact':True,'metrics':metrics,'point_sample_sha256':point_hash})
    def aggregate(rows):
        return {'n':len(rows),'high_overlap_both_ge_0p8':sum(r['metrics']['high_overlap_both_ge_0p8'] for r in rows),'high_overlap_and_no_extra_supported_independent_component':sum(r['metrics']['high_overlap_and_no_extra_supported_independent_component'] for r in rows),'source_coverage_fraction_mean':float(np.mean([r['metrics']['source_component_covered_by_alt_fraction'] for r in rows])) if rows else None,'alt_from_source_component_fraction_mean':float(np.mean([r['metrics']['alt_support_from_source_component_fraction'] for r in rows])) if rows else None,'extra_independent_supported_component_count_total':sum(r['metrics']['extra_independent_global_components_meeting_min_support'] for r in rows)}
    public={'state':'bounded_unlabelled_descriptive_diagnostic_no_method_change','purpose':PURPOSE,'full_prediction_seal_sha256':sha(seal_path),'source_checkpoint_sha256':seal['source_checkpoint_sha256'],'plan_sha256':sha(a.plan),'selected_n':len(cases),'eligible_counts_by_platform':eligible,'selection':'hash(salt|platform|id), up to8 per platform, reason-only; no GT/IoU selection','overlap_threshold_descriptive_only':OVERLAP,'independent_component_definition':'Global visible same-color graph at frozen G radius; no source-component point overlap, at least frozen min_component_points among alt support. Geometric observation disconnection, not object identity.','overall':aggregate(cases),'platforms':{p:aggregate([r for r in cases if r['platform']==p]) for p in ('drone','quad')},'no_GT_instance_identity_or_task_improvement_claim':True,'no_G_threshold_plan_or_adoption_change':True,'seconds':time.perf_counter()-started}
    (a.output/'PERCASE_PRIVATE.json').write_text(json.dumps(to_json(cases),indent=2)+'\n');(a.output/'PUBLIC_AGGREGATES.json').write_text(json.dumps(to_json(public),indent=2)+'\n')
    (a.output/'SEALED.json').write_text(json.dumps({'purpose':PURPOSE,'plan_sha256':sha(a.plan),'full_prediction_seal_sha256':sha(seal_path),'files':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}},indent=2)+'\n');print('COMPONENT_DIAGNOSTIC_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':main()
