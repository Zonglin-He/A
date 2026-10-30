"""R11 author-side unlabelled mechanism reduction. Release-gated, no GT/media/P.
Reads bound seals/trace/controls/bridges and explicit NPZ selected-array keys only.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math
import numpy as np
ARMS=('source','unconditional_full','no_rgb_coordinate','weighted_constraints','uniform_weighted_constraints','reliable_50','hash_random_50','source_agreement_50','low_plane_residual_50','point_count_50','stability_only_50')
CONTROLS=('reliability','hash_random','source_agreement','low_plane_residual','point_count','stability_only')
KEYS=('sample_id','q_source','arm_names','arm_query_index','arm_boxes_raw6','atomic_active6','base_acceptance_mask6','coordinate_reliability6','coordinate_defined6','RGB_reliability','RGB_defined','case_reliability','case_defined','control_score_names','control_scores','control_defined','gate_selection','association_status','face_keys','face_weights')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def describe(values):
    x=np.asarray([v for v in values if v is not None],float);assert np.isfinite(x).all()
    return {'n_defined':len(x),'zero_count':int((x==0).sum()),'min':float(x.min()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'mean':float(x.mean()) if len(x) else None,'max':float(x.max()) if len(x) else None}
def score_distribution(values):
    vals=[float(v) for v in values if v is not None];cc=Counter(vals)
    return {'n':len(values),'NA':len(values)-len(vals),'defined':len(vals),'zero_count':sum(v==0 for v in vals),'distinct_exact_values':len(cc),'tied_value_count':sum(n>1 for n in cc.values()),'largest_exact_tie':max(cc.values(),default=0),'defined_scores':describe(vals),'ties_are_exact_float_values_not_posthoc_tolerance':True}
def solver_summary(diag):
    sol=diag.get('solver',{}).get('full_rgb')
    if sol is None:return None
    return {k:sol[k] for k in ('success','status','iterations','numerical_invalid')}
def one(trace,z):
    assert tuple(z['arm_names'].tolist())==ARMS and tuple(z['control_score_names'].tolist())==CONTROLS
    b=z['arm_boxes_raw6'].astype(float);assert b.shape==(11,6) and np.isfinite(b).all()
    changed=np.any(b!=b[0],axis=1);eligible=bool(changed[1]);assert eligible==trace['base_actual_change']
    active=np.asarray(z['atomic_active6'],bool);assert np.array_equal(active,b[1]!=b[0])
    assert not (active & ~np.asarray(z['base_acceptance_mask6'],bool)).any()
    assert np.all(z['arm_query_index']==int(z['q_source'])) and str(z['sample_id'])==trace['id']
    rel=trace['reliability'];status=rel['association_status'];assert str(z['association_status'])==status
    assert np.array_equal(z['gate_selection'],np.array([trace['gates'][c] for c in CONTROLS]))
    scores={c:(float(z['control_scores'][i]) if bool(z['control_defined'][i]) else None) for i,c in enumerate(CONTROLS)}
    for c in CONTROLS:assert scores[c]==rel['control_scores'].get(c)
    case=float(z['case_reliability']) if bool(z['case_defined']) else None;assert case==rel['case_reliability']
    effaxis=np.asarray(z['coordinate_reliability6'],float);effdef=np.asarray(z['coordinate_defined6'],bool)
    assert np.array_equal(effaxis,np.asarray(rel['axis_reliability6'])) and np.array_equal(effdef,np.asarray(rel['axis_reliability_defined6']))
    assert float(z['RGB_reliability'])==rel['RGB_weight'] and bool(z['RGB_defined'])==rel['RGB_weight_defined']
    facekeys=z['face_keys'].tolist();assert np.array_equal(z['face_weights'],[rel['face_weights'][k] for k in facekeys])
    if eligible and status!='association_supported':assert not z['face_weights'].any() and float(z['RGB_reliability'])==0 and not changed[3] and not changed[4]
    if case is not None and status=='association_supported':assert case==min(effaxis[active]) and effdef[active].all()
    features=rel.get('features',{});part=rel.get('partition_hashes',{});faces=features.get('faces',{});activefaces=rel['active_face_keys']
    frows=[]
    for key in activefaces:
        f=faces[key];raw=rel.get('raw_face_weights',{}).get(key)
        rawdefined=bool(rel.get('raw_face_weight_defined',{}).get(key,f.get('defined',False)))
        defined=True if status=='association_conflicted' else status=='association_supported' and bool(f.get('defined',False))
        frows.append({'face_key':key,'raw_weight':raw,'raw_defined':rawdefined,'effective_weight':float(rel['face_weights'][key]),'effective_defined':defined,'undefined_reason':f.get('reason'),'partition_counts':f.get('counts_base_A_B'),'point_count_control':f.get('physical_support_count_control'),'final_plane_residual_m':f.get('final_plane_residual_m'),'offset_disagreement_m':f.get('offset_disagreement_m'),'normal_alignment':f.get('unsigned_normal_alignment'),'coverage_ratio':f.get('min_tangent_coverage_ratio')})
    rawaxis=np.asarray(rel.get('raw_axis_reliability6',np.zeros(6)),float);rawdef=np.asarray(rel.get('raw_axis_reliability_defined6',np.zeros(6)),bool)
    if eligible:
        assert sum(part['partition_unique_XYZ_counts'])==trace['geometry_unique_XYZ']
        assert part['disjoint_physical_observations'] and part['same_XYZ_same_partition'] and part['normal_graphs_must_be_rebuilt_per_partition']
    comp=features.get('component_ambiguity',{});ms=features.get('mask_stability',{})
    gate=features.get('correspondence_gate',{})
    return {'id':trace['id'],'platform':trace['platform'],'baseC':eligible,'association_status':status,'segmentation_reason':trace['RGB'].get('reason','unknown'),'segmentation_available':trace['RGB'].get('state')=='point_seeded_grabcut_component','global_gate_reason':trace['base_geometry'].get('diagnostics',{}).get('global_gate_reason',trace['base_geometry'].get('diagnostics',{}).get('reason','unknown')),
      'changed_arms':changed.tolist(),'center_increment_l2_m':np.linalg.norm(b[:,:3]-b[0,:3],axis=1).tolist(),'size_increment_l2_m':np.linalg.norm(b[:,3:]-b[0,3:],axis=1).tolist(),'weighted_uniform_float32_identical':bool(np.array_equal(b[3],b[4])),
      'active_axis6':active.tolist(),'effective_axis_defined6':effdef.tolist(),'raw_axis_defined6':rawdef.tolist(),'effective_axis_weights6':effaxis.tolist(),'raw_axis_weights6':rawaxis.tolist(),
      'case_reliability':case,'raw_case_reliability':rel.get('raw_case_reliability'),'scores':scores,'selections':{c:bool(z['gate_selection'][i]) for i,c in enumerate(CONTROLS)},
      'face_constraints':frows,'RGB_effective':float(rel['RGB_weight']),'RGB_effective_defined':bool(rel['RGB_weight_defined']),'RGB_raw':rel.get('raw_RGB_weight'),'RGB_raw_defined':bool(rel.get('raw_RGB_weight_defined',False)),
      'raw_positive_effective_zero_faces':sum(f['raw_weight'] is not None and f['raw_weight']>0 and f['effective_weight']==0 for f in frows),'raw_positive_RGB_effective_zero':rel.get('raw_RGB_weight',0)>0 and rel['RGB_weight']==0,
      'undefined_reasons':rel.get('undefined_reasons',[]),'correspondence_gate':gate,'partition_binding':part,'partition_segmentations':features.get('partition_segmentation',[]),
      'mask_stability':ms,'component_ambiguity':comp,'crossheldout_foreground_support':features.get('crossheldout_foreground_support'),
      'solver':{name:solver_summary(diag) for name,diag in [('unconditional_full',trace['base_geometry'].get('diagnostics',{})),('weighted_constraints',trace['weighted_solver']),('uniform_weighted_constraints',trace['uniform_solver'])]},
      'uniform_means':trace['uniform_means'],'atomic_scope':'coordinate application, not leave-one-face-out causal intervention'}
def aggregate(rows):
    C=[r for r in rows if r['baseC']];faces=[f for r in C for f in r['face_constraints']]
    out={'n':len(rows),'baseC':len(C),'segmentation_available':sum(r['segmentation_available'] for r in rows),'segmentation_reasons':dict(Counter(r['segmentation_reason'] for r in rows)),'base_gate_reasons':dict(Counter(r['global_gate_reason'] for r in rows)),
      'association_all':dict(Counter(r['association_status'] for r in rows)),'association_in_C':dict(Counter(r['association_status'] for r in C)),
      'case_effective_in_C':score_distribution([r['case_reliability'] for r in C]),'case_raw_in_C':score_distribution([r['raw_case_reliability'] for r in C]),
      'undefined_reasons_in_C':dict(Counter(x for r in C for x in r['undefined_reasons'])),
      'actual_coordinate_axes6_in_C':np.sum([r['active_axis6'] for r in C],axis=0).tolist() if C else [0]*6,
      'effective_defined_actual_axes':sum(sum(a and d for a,d in zip(r['active_axis6'],r['effective_axis_defined6'])) for r in C),
      'raw_defined_actual_axes':sum(sum(a and d for a,d in zip(r['active_axis6'],r['raw_axis_defined6'])) for r in C),
      'raw_positive_face_effective_zero_count':sum(r['raw_positive_effective_zero_faces'] for r in C),'raw_positive_RGB_effective_zero_cases':sum(r['raw_positive_RGB_effective_zero'] for r in C),
      'face_active_count':len(faces),'face_raw_defined':sum(f['raw_defined'] for f in faces),'face_effective_defined':sum(f['effective_defined'] for f in faces),
      'face_raw_defined_weights':describe([f['raw_weight'] for f in faces if f['raw_defined']]),'face_effective_defined_weights':describe([f['effective_weight'] for f in faces if f['effective_defined']]),
      'active_face_reason_counts':dict(Counter(f['undefined_reason'] for f in faces)),
      'RGB_raw_defined_in_C':sum(r['RGB_raw_defined'] for r in C),'RGB_effective_defined_in_C':sum(r['RGB_effective_defined'] for r in C),
      'RGB_raw_defined_weights':describe([r['RGB_raw'] for r in C if r['RGB_raw_defined']]),'RGB_effective_defined_weights':describe([r['RGB_effective'] for r in C if r['RGB_effective_defined']]),
      'weighted_uniform_identical_in_C':sum(r['weighted_uniform_float32_identical'] for r in C),
      'partition_bound_cases':sum(bool(r['partition_binding']) for r in C),'partition_A_B_unique_XYZ_counts':[describe([r['partition_binding']['partition_unique_XYZ_counts'][h] for r in C]) for h in (0,1)],
      'partition_segmentation_reasons':dict(Counter(p['reason'] for r in C for p in r['partition_segmentations'])),
      'mask_stability':describe([r['mask_stability'].get('score') for r in C]),'mask_IoU':describe([r['mask_stability'].get('IoU') for r in C]),'crossheldout_support':describe([r['crossheldout_foreground_support'] for r in C]),
      'component_conflict_cases':sum(bool(r['component_ambiguity'].get('conflict_observed',False)) for r in C),'zero_substantial_component_cases':sum(r['component_ambiguity'].get('substantial_component_count')==0 for r in C),
      'fragmentation_purity_diagnostic_only':describe([r['component_ambiguity'].get('fragmentation_purity') for r in C]),'ambiguity_factor':describe([r['component_ambiguity'].get('ambiguity_factor') for r in C])}
    out['per_axis_actual_applications']={str(j):{'n_active':sum(r['active_axis6'][j] for r in C),'raw_defined':sum(r['active_axis6'][j] and r['raw_axis_defined6'][j] for r in C),'effective_defined':sum(r['active_axis6'][j] and r['effective_axis_defined6'][j] for r in C),'raw_defined_weights':describe([r['raw_axis_weights6'][j] for r in C if r['active_axis6'][j] and r['raw_axis_defined6'][j]]),'effective_defined_weights':describe([r['effective_axis_weights6'][j] for r in C if r['active_axis6'][j] and r['effective_axis_defined6'][j]])} for j in range(6)}
    out['active_face_partition_counts_base_A_B']=[describe([f['partition_counts'][h] for f in faces if f['partition_counts'] is not None]) for h in range(3)]
    out['arms']={}
    for i,name in enumerate(ARMS):out['arms'][name]={'actual_nonzero_samples':sum(r['changed_arms'][i] for r in rows),'actual_nonzero_in_C':sum(r['changed_arms'][i] for r in C),'center_increment_l2_m_all':describe([r['center_increment_l2_m'][i] for r in rows]),'size_increment_l2_m_all':describe([r['size_increment_l2_m'][i] for r in rows])}
    out['controls']={}
    for c in CONTROLS:
        chosen=[r for r in C if r['selections'][c]]
        out['controls'][c]={'distribution_in_C':score_distribution([r['scores'][c] for r in C]),'selected':len(chosen),'selected_score_NA':sum(r['scores'][c] is None for r in chosen) if c!='hash_random' else None,'selected_association':dict(Counter(r['association_status'] for r in chosen)),'selected_case_NA':sum(r['case_reliability'] is None for r in chosen),'selected_case_defined_zero':sum(r['case_reliability']==0 for r in chosen)}
    out['solvers']={}
    for name in ('unconditional_full','weighted_constraints','uniform_weighted_constraints'):
        sol=[r['solver'][name] for r in rows if r['solver'][name] is not None]
        out['solvers'][name]={'objects':len(sol),'statuses':dict(Counter(s['status'] for s in sol)),'success':sum(s['success'] for s in sol),'numerical_invalid':sum(s['numerical_invalid'] for s in sol),'finite_nonconverged':sum(s['status']=='finite_nonconverged_bounded_result' for s in sol),'iterations':describe([s['iterations'] for s in sol])}
    return out

def main(a):
    assert sha(a.release)==a.release_sha256;release=read(a.release)
    assert release['mechanism_reduction_permitted'] is True and release['GT_read_permitted'] is False and release['reducer_sha256']==sha(Path(__file__))
    gd=Path(release['geometry_dir']);sp=Path(release['source_dir'])/'SEALED.json'
    assert sha(gd/'SEALED.json')==release['geometry_seal_sha256'] and sha(sp)==release['source_seal_sha256']
    seal=read(gd/'SEALED.json');ss=read(sp);root=Path(__file__).resolve().parents[3]
    assert seal['source_seal_sha256']==release['source_seal_sha256'] and seal['source_checkpoint_sha256']==ss['source_checkpoint_sha256']
    for n,h in seal['program_sha256'].items():assert sha(root/n)==h,n
    for n,h in seal['files'].items():assert Path(n).name==n and sha(gd/n)==h,n
    traces=read(gd/'trace.json');controls=read(gd/'COHORT_CONTROL_LOCK.json');bridges=read(gd/'ACTUAL_WEIGHT_BRIDGES.json')
    assert len(traces)==seal['n_samples'];rows=[]
    for tr in traces:
        with np.load(gd/(tr['id']+'.npz'),allow_pickle=False) as z:fields={k:z[k] for k in KEYS}
        rows.append(one(tr,fields))
    assert len({r['id'] for r in rows})==len(rows)
    C=[r for r in rows if r['baseC']]
    assert len(bridges)==len(C)==seal['actual_allone_allzero_bridge_cases']
    assert {b['id'] for b in bridges}=={r['id'] for r in C}
    assert all(b['allone_base_bitexact'] and b['allzero_source_bitexact'] for b in bridges)
    denom={}
    for platform in sorted({r['platform'] for r in rows}):
        sub=[r for r in C if r['platform']==platform];fv=[f['effective_weight'] for r in sub for f in r['face_constraints'] if f['effective_defined']];iv=[r['RGB_effective'] for r in sub if r['RGB_effective_defined']]
        denom[platform]={}
        for key,values in [('face',fv),('RGB',iv)]:
            v={'n_defined_active':len(values),'mean':math.fsum(values)/len(values) if values else None,'defined_zero_count':sum(x==0 for x in values)}
            assert v==controls['uniform_means_perplatform'][platform][key]==seal['uniform_means_perplatform'][platform][key];denom[platform][key]=v
        for c in CONTROLS:
            selected={r['id'] for r in sub if r['selections'][c]};registered=controls['selection'][c]
            assert selected==set(registered['selected_ids']) & {r['id'] for r in sub}
            assert len(selected)==(len(sub)+1)//2==registered['by_platform'][platform]['K']
    public={'state':'author_side_no_GT_R11_mechanism_description_not_independent_quality_audit','binding':{'geometry_seal_sha256':release['geometry_seal_sha256'],'source_seal_sha256':release['source_seal_sha256'],'source_checkpoint_sha256':seal['source_checkpoint_sha256'],'program_sha256':seal['program_sha256'],'reducer_sha256':sha(Path(__file__)),'release_sha256':sha(a.release)},'all':aggregate(rows),'by_platform':{p:aggregate([r for r in rows if r['platform']==p]) for p in sorted({r['platform'] for r in rows})},'uniform_means_and_exact_denominators_recomputed':denom,'actual_weight_bridges':{'n':len(bridges),'allone_base_bitexact_pass':sum(b['allone_base_bitexact'] for b in bridges),'allzero_source_bitexact_pass':sum(b['allzero_source_bitexact'] for b in bridges)},'limits':['No target GT, scores, images, RGB masks, raw points, observation/feature caches or index sidecars decoded.','Physical A/B disjointness is a bound implementation contract and trace hash/count check; reducer does not reopen P to independently prove it.','Shared RGB/source q0/shape prior can preserve the same wrong evidence in both partitions; association_supported is geometric compatibility only.','Effective scores are primary; raw scores are pre-gate diagnostics. Defined conflict0 remains different from association-undefined NA.','Uniform means include defined conflict0, exclude inactive and undefined; active-face and RGB denominators are separate.','Exact-half ranking can retain NA to meet K; it is not the weighted-constraint failclosed rule.','Changed coordinates, stable evidence and converged loss do not establish task improvement.']}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'SUMMARY.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n');(a.output/'CASES_PRIVATE.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    lines=['# R11 无标签机制汇总','', '实现作者侧描述，不是独立质量审计；未读取GT、分数或原始媒体。','']
    for name,v in [('all',public['all']),*public['by_platform'].items()]:lines.extend([f"{name}: N={v['n']}, baseC={v['baseC']}, association(C)={v['association_in_C']}, weighted changed={v['arms']['weighted_constraints']['actual_nonzero_samples']}, uniform changed={v['arms']['uniform_weighted_constraints']['actual_nonzero_samples']}; case NA={v['case_effective_in_C']['NA']}, exact distinct={v['case_effective_in_C']['distinct_exact_values']}.",''])
    lines.extend(['全部覆盖、权重、分母、排序、门控与solver计数见SUMMARY.json。保留零与未定义结果；不据此判断定位正确。','',*['- '+x for x in public['limits']]])
    (a.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n');files={n:sha(a.output/n) for n in ('SUMMARY.json','CASES_PRIVATE.json','SUMMARY.md')}
    (a.output/'SEALED.json').write_text(json.dumps({'state':'sealed_R11_author_no_GT_mechanism_reduction','files':files,'release_sha256':sha(a.release),'reducer_sha256':sha(Path(__file__))},indent=2)+'\n')
    print(json.dumps({'n':len(rows),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
