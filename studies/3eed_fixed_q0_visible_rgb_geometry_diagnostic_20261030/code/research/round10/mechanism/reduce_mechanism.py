"""Author-side R10 unlabelled mechanism reduction, executed only by bound release.
Reads seals, trace and an explicit prediction-array allowlist. Never opens source
caches, observations, RGB masks, annotations, scoring sidecars or task scores.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')
ARRAY_KEYS=('sample_id','q_source','arm_names','arm_query_index','arm_boxes_raw6','pre_constraint_geometry6','acceptance_mask6','actual_changed_axes','point_support_sha256','arm_status')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def describe(x):
    x=np.asarray([v for v in x if v is not None],float)
    assert np.isfinite(x).all()
    return {'n':len(x),'min':float(x.min()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'mean':float(x.mean()) if len(x) else None,'max':float(x.max()) if len(x) else None,'sum':float(x.sum())}
def one(trace,z):
    assert tuple(z['arm_names'].tolist())==ARMS
    q=int(z['q_source']);assert np.all(z['arm_query_index']==q)
    b=np.asarray(z['arm_boxes_raw6']);m=np.asarray(z['acceptance_mask6'],bool)
    assert b.shape==m.shape==(4,6) and np.isfinite(b).all()
    changed=b!=b[0];assert np.array_equal(changed,z['actual_changed_axes'])
    assert not changed[~m].any()
    assert np.array_equal(m[1],m[2]) and np.array_equal(m[1],m[3])
    assert z['point_support_sha256'][1]==z['point_support_sha256'][2]
    geom=trace['geometry'];d=geom.get('diagnostics',{});rgb=trace['RGB'];s=d.get('solver',{})
    faces=d.get('source_face_evidence_RGB',{})
    both=[all(faces.get(f'{j}:{v}',{}).get('accepted_plane',False) for v in (-1,1)) for j in range(3)]
    support=d.get('RGB_support_count');all_support=d.get('geometric_support_count')
    terms={}
    for name in ('full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation'):
        v=s.get(name)
        terms[name]=None if v is None else {k:v[k] for k in ('success','status','numerical_invalid','iterations','initial_terms','final_terms')}
    delta=b.astype(float)-b[0].astype(float)
    return {'id':trace['id'],'platform':trace['platform'],
      'segmentation_valid':rgb.get('state')=='point_seeded_grabcut_component',
      'segmentation_reason':rgb.get('reason','unknown'),
      'global_gate_reason':d.get('global_gate_reason',d.get('reason','unknown')),
      'model_input_rows':trace['model_input_rows'],'unique_raw_indices':trace['unique_raw_indices'],
      'unique_XYZ':trace['geometry_unique_XYZ_rows'],
      'visible_unique_count':d.get('visible_unique_count'),'geometry_support_count':all_support,
      'RGB_support_count':support,'RGB_over_geometric_support':support/all_support if all_support else None,
      'accepted_axis6':m[1].tolist(),'any_accepted':bool(m[1].any()),
      'two_qualified_opposing_plane_axes':both,'strict_size_pair_axes':m[1,3:].tolist(),
      'size_pair_is_source_conditioned_hypothesis_not_object_completeness':True,
      'arm_actual_changed_axes':changed.tolist(),'arm_any_changed':changed.any(1).tolist(),
      'arm_center_increment_l2_m':np.linalg.norm(delta[:,:3],axis=1).tolist(),
      'arm_size_increment_l2_m':np.linalg.norm(delta[:,3:],axis=1).tolist(),
      'full_noRGB_coordinates_identical_float32':bool(np.array_equal(b[1],b[2])),
      'full_noRGB_maxabs_coordinate_difference_m':float(np.max(np.abs(b[1].astype(float)-b[2].astype(float)))),
      'ablation_full_maxabs_coordinate_difference_m':float(np.max(np.abs(b[3].astype(float)-b[1].astype(float)))),
      'full_noRGB_shared_preconstraint_exact':bool(np.array_equal(z['pre_constraint_geometry6'][1],z['pre_constraint_geometry6'][2])),
      'RGB_boundary_sample_count':d.get('RGB_boundary_sample_count'),
      'RGB_boundary_origin':'segmentation mask erosion boundary; deterministic row-major subsample of actual FG pixels; not amodal silhouette',
      'solver':terms,'arm_status':z['arm_status'].tolist()}
def aggregate(rows):
    out={'n':len(rows),'segmentation_valid':sum(r['segmentation_valid'] for r in rows),
      'segmentation_reasons':dict(Counter(r['segmentation_reason'] for r in rows)),
      'global_gate_reasons':dict(Counter(r['global_gate_reason'] for r in rows)),
      'any_accepted':sum(r['any_accepted'] for r in rows),
      'accepted_axis_counts6':np.sum([r['accepted_axis6'] for r in rows],axis=0).tolist() if rows else [0]*6,
      'two_qualified_opposing_plane_axis_counts':np.sum([r['two_qualified_opposing_plane_axes'] for r in rows],axis=0).tolist() if rows else [0]*3,
      'strict_size_pair_axis_counts':np.sum([r['strict_size_pair_axes'] for r in rows],axis=0).tolist() if rows else [0]*3,
      'any_strict_size_pair':sum(any(r['strict_size_pair_axes']) for r in rows),
      'full_noRGB_identical_float32':sum(r['full_noRGB_coordinates_identical_float32'] for r in rows),
      'full_noRGB_different_float32':sum(not r['full_noRGB_coordinates_identical_float32'] for r in rows),
      'full_noRGB_shared_preconstraint_exact_all':all(r['full_noRGB_shared_preconstraint_exact'] for r in rows)}
    for k in ('model_input_rows','unique_raw_indices','unique_XYZ','visible_unique_count','geometry_support_count','RGB_support_count','RGB_over_geometric_support','RGB_boundary_sample_count','full_noRGB_maxabs_coordinate_difference_m','ablation_full_maxabs_coordinate_difference_m'):
        out[k]=describe([r[k] for r in rows])
    out['arms']={}
    for a,name in enumerate(ARMS):
        ar={'actually_changed_samples':sum(r['arm_any_changed'][a] for r in rows),'changed_axis_counts6':np.sum([r['arm_actual_changed_axes'][a] for r in rows],axis=0).tolist() if rows else [0]*6,
            'center_increment_l2_m':describe([r['arm_center_increment_l2_m'][a] for r in rows]),'size_increment_l2_m':describe([r['arm_size_increment_l2_m'][a] for r in rows]),'statuses':dict(Counter(r['arm_status'][a] for r in rows))}
        if a:
            ss=[r['solver'][name] for r in rows if r['solver'][name] is not None]
            ar['solver_executed']=len(ss);ar['solver_success']=sum(x['success'] for x in ss)
            ar['numerical_invalid']=sum(x['numerical_invalid'] for x in ss)
            ar['finite_nonconverged']=sum(x['status']=='finite_nonconverged_bounded_result' for x in ss)
            ar['iterations']=describe([x['iterations'] for x in ss])
            ar['terms']={term:{stage:describe([x[stage+'_terms'][term] for x in ss]) for stage in ('initial','final')} for term in ('prior','observed_surface','RGB_one_sided_containment','total')}
            ar['RGB_term_positive_at_source']=sum(x['initial_terms']['RGB_one_sided_containment']>0 for x in ss)
            ar['RGB_term_positive_after']=sum(x['final_terms']['RGB_one_sided_containment']>0 for x in ss)
            ar['RGB_term_recorded_as_zero_by_design_when_disabled']=name=='no_rgb_coordinate'
        out['arms'][name]=ar
    return out

def main(a):
    assert sha(a.release)==a.release_sha256
    release=read(a.release);assert release['mechanism_reduction_permitted'] is True and release['GT_read_permitted'] is False
    assert sha(Path(__file__))==release['reducer_sha256']
    gd=Path(release['geometry_dir']);sp=Path(release['source_dir'])/'SEALED.json'
    assert sha(gd/'SEALED.json')==release['geometry_seal_sha256'] and sha(sp)==release['source_seal_sha256']
    seal=read(gd/'SEALED.json');ss=read(sp)
    assert seal['source_seal_sha256']==release['source_seal_sha256'] and seal['source_checkpoint_sha256']==ss['source_checkpoint_sha256']
    root=Path(__file__).resolve().parents[3]
    for name,digest in seal['program_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in seal['files'].items():
        assert Path(name).name==name and sha(gd/name)==digest,name
    traces=read(gd/'trace.json');assert len(traces)==seal['n_samples']
    rows=[]
    for t in traces:
        name=t['id']+'.npz';assert name in seal['files']
        with np.load(gd/name,allow_pickle=False) as z:
            # In particular RGB_mask2d_private and source pool coordinates are not opened.
            fields={k:z[k] for k in ARRAY_KEYS}
        assert str(fields['sample_id'])==t['id'];rows.append(one(t,fields))
    assert len({r['id'] for r in rows})==len(rows)
    a.output.mkdir(parents=True,exist_ok=False)
    public={'state':'author_side_no_GT_mechanism_description_not_independent_quality_audit','binding':{'geometry_seal_sha256':release['geometry_seal_sha256'],'source_seal_sha256':release['source_seal_sha256'],'source_checkpoint_sha256':seal['source_checkpoint_sha256'],'program_sha256':seal['program_sha256'],'reducer_sha256':sha(Path(__file__)),'release_sha256':sha(a.release)},'all':aggregate(rows),'by_platform':{p:aggregate([r for r in rows if r['platform']==p]) for p in sorted({r['platform'] for r in rows})},'limitations':['No target labels, scores, annotation files, raw images, RGB masks, observations or source feature caches read.','Accepted opposing planes are measured support under source-conditioned object hypothesis, not verified instance completeness.','Changed coordinates and decreasing objective are not task correctness.','Whole-observation direct-G stage; no held-out improvement or parameter adaptation claimed.','No-RGB-coordinate shares RGB-derived support and acceptance; pointmask ablation also inherits this RGB gate.','No-RGB-coordinate solver records RGB term as zero because disabled, not because its box contains all foreground.']}
    (a.output/'SUMMARY.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    (a.output/'CASES_PRIVATE.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    lines=['# R10 无标签机制汇总','',f"样本 {len(rows)}；本文件是实现作者侧机制描述，不是独立质量审计。未读取 GT 或任务分数。",'']
    for p,v in [('all',public['all']),*public['by_platform'].items()]:
        lines.extend([f"{p}：分割可用 {v['segmentation_valid']}/{v['n']}；接受任意坐标 {v['any_accepted']}；具备严格双侧尺寸资格 {v['any_strict_size_pair']}；full/offcoord 框 float32 不同 {v['full_noRGB_different_float32']}。",''])
    lines.extend(['支持点、接受轴、实际变化、求解器和各损失项的全部计数见 SUMMARY.json。未接受的轴保持原框；未观测尺寸保持先验并不证明先验正确。','',*['- '+x for x in public['limitations']]])
    (a.output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    files={n:sha(a.output/n) for n in ('SUMMARY.json','CASES_PRIVATE.json','SUMMARY.md')}
    (a.output/'SEALED.json').write_text(json.dumps({'state':'sealed_author_no_GT_mechanism_reduction','files':files,'release_sha256':sha(a.release),'reducer_sha256':sha(Path(__file__))},indent=2)+'\n')
    print(json.dumps({'n':len(rows),'seal_sha256':sha(a.output/'SEALED.json')}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--release-sha256',required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
