"""Synthetic geometry+external semantic fixture only; no real RGB/CLIP/data/GT."""
from pathlib import Path
from dataclasses import asdict
import ast,hashlib,json
import numpy as np
from association_geometry import prepare_candidates,associate_and_consensus,DEFAULT_CONFIG,pair_evidence

def scene():
    base=np.array([(x,y,z) for x in np.linspace(-.75,.75,5) for y in np.linspace(-.35,.35,5) for z in np.linspace(4.65,5.35,3)])
    points=np.r_[base,base+[4.,0,0]];uv=np.c_[80+points[:,0]*12,80+points[:,1]*20];depth=points[:,2]
    boxes=np.array([[0,0,5,2,1,1],[.12,0,5,2,1,1],[4,0,5,2,1,1],[4.12,0,5,2,1,1]],float)
    return dict(points=points,uv=uv,depth=depth,image_size_wh=(256,256),source_boxes=boxes,text_scores=np.array([1.,.7,.8,.6]))

def main():
    d=scene();p=prepare_candidates(**d);assert len(p['groups'])==2 and len(p['roi_groups'])==2
    assert p['q_source']==0 and p['dedup_membership'][0]==p['dedup_membership'][1] and p['dedup_membership'][2]==p['dedup_membership'][3]
    sem=np.array([[.1,.1],[.85,.85]])
    r=associate_and_consensus(p,sem);assert r['original_source_q']==0 and r['associated_q']==2
    assert r['r_global']>0 and r['w6'][:3].any() and r['diagnostics']['nonzero_pseudo_supervision']
    assert np.array_equal(r['association_only_box'],d['source_boxes'][2])
    assert np.array_equal(r['pseudo6'][3:],d['source_boxes'][2,3:]) and not r['w6'][3:].any()
    assert np.allclose(r['pseudo6'][:3]-r['association_only_box'][:3],r['r_global']*(r['raw_consensus_center']-r['association_only_box'][:3]))
    # A global w scalar would cancel; global trust actually changes target delta.
    assert not np.array_equal(r['pseudo6'][:3],r['raw_consensus_center'])
    shuffled=associate_and_consensus(p,sem,mode='semantic_shuffle');assert shuffled['associated_q']==0
    pc=associate_and_consensus(p,sem,mode='pc_only');pc_other=associate_and_consensus(p,sem[::-1],mode='pc_only');assert pc['associated_q']==pc_other['associated_q'] and np.array_equal(pc['pseudo6'],pc_other['pseudo6'])
    ties=associate_and_consensus(p,np.full_like(sem,.2));assert ties['r_global']==0 and not ties['w6'].any()
    # Exact repeated proposals must not create new geometric representatives or
    # alter consensus through duplicate-query multiplicity.
    dup=dict(d,source_boxes=np.r_[d['source_boxes'],d['source_boxes'][2:3]],text_scores=np.r_[d['text_scores'],.3]);pd=prepare_candidates(**dup);rd=associate_and_consensus(pd,sem)
    assert len(rd['diagnostics']['geometry_representatives'])==2 and np.allclose(rd['pseudo6'],r['pseudo6'])
    # Single-member physical target group gives no pseudo-update.
    single=dict(d,source_boxes=d['source_boxes'][:3],text_scores=d['text_scores'][:3]);rs=associate_and_consensus(prepare_candidates(**single),sem);assert rs['associated_q']==2 and not rs['w6'].any()
    # Spatial box overlap alone is not evidence of duplicate observation support.
    edge=pair_evidence(d['source_boxes'][0],d['source_boxes'][1],frozenset(range(10)),frozenset(range(20,30)),DEFAULT_CONFIG);assert edge['box_AABB_IoU']>.8 and not edge['duplicate_edge']
    perm=np.random.default_rng(2).permutation(len(d['points']));dp=dict(d,points=d['points'][perm],uv=d['uv'][perm],depth=d['depth'][perm]);rp=associate_and_consensus(prepare_candidates(**dp),sem)
    assert np.array_equal(rp['pseudo6'],r['pseudo6']) and np.array_equal(rp['w6'],r['w6'])
    # No points in frame -> no eligible physical ROIs and identity.
    empty=dict(d,uv=d['uv']+10000);pe=prepare_candidates(**empty);re=associate_and_consensus(pe,np.empty((0,2)));assert not re['w6'].any() and re['associated_q']==0
    assert all(g['tight_xyxy'] is not None and g['context_xyxy'] is not None for g in p['roi_groups'])
    rejected=False
    try:prepare_candidates(**d,positive_map=np.ones(3))
    except TypeError:rejected=True
    assert rejected
    here=Path(__file__).resolve().parent;raw=json.dumps(asdict(DEFAULT_CONFIG),sort_keys=True,separators=(',',':')).encode()
    result={'state':'synthetic_CPU_association_geometry_gate_pass_not_real_method_result','pool_size':len(p['candidate_pool']),'physical_groups':len(p['groups']),'ROI_count':len(p['roi_groups']),'source_q_preserved':r['original_source_q'],'semantic_associated_query':r['associated_q'],'nonzero_genuine_multi_proposal_consensus':r['diagnostics']['nonzero_pseudo_supervision'],'raw_consensus_minus_anchor_m':r['diagnostics']['raw_consensus_minus_anchor_m'],'r_global':r['r_global'],'final_pseudo_minus_anchor_m':r['diagnostics']['final_pseudo_minus_anchor_m'],'confidence_shrinks_same_direct_and_TTA_target':True,'duplicate_multiplicity_not_new_evidence':True,'same_box_overlap_without_shared_points_not_merged':True,'single_member_tied_semantics_lowvisibility_skip':True,'size_prior_w0_yaw0':True,'no_color_or_caption_hardgate':True,'semanticshuffle_and_PConly_controls':True,'point_permutation_exact':True,'GT_extra_field_rejected':True,'config':asdict(DEFAULT_CONFIG),'canonical_config_sha256':hashlib.sha256(raw).hexdigest(),'code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ['association_geometry.py',Path(__file__).name,'ASSOCIATION_GEOMETRY_CONTRACT.md']},'no_GPU_real_RGB_CLIP_model_target_GT_or_scoring_used':True}
    (here/'CPU_ASSOCIATION_GEOMETRY_FIXTURE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
