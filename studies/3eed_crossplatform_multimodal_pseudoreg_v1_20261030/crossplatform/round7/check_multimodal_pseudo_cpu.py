"""Synthetic point/RGB/projection fixtures only; no real data/model/GT/GPU."""
import hashlib,json
from pathlib import Path
import numpy as np
from multimodal_pseudo import generate_pseudo,DEFAULT_CONFIG,target_color

def cloud(delta=.1):
    yz=np.array([(y,z) for y in np.linspace(-.5,.5,6) for z in np.linspace(-.5,.5,6)])
    faces=np.concatenate([np.c_[np.full(len(yz),x+delta),yz[:,0],yz[:,1]+5] for x in (-1,1)])
    bridge=np.c_[np.linspace(-.8,.8,9)+delta,np.zeros(9),np.full(9,5.)]
    return np.r_[faces,bridge]
def inputs(p):
    n=len(p);im=np.zeros((32,32,3),np.uint8);uv=np.c_[np.arange(n)%32,np.arange(n)//32];im[uv[:,1],uv[:,0]]=[255,0,0]
    return dict(points=p,rgb=im,text='the red car beside the blue car. not mentioned',source_boxes=np.array([[0,0,5,2,1,1.]]),text_scores=np.array([1.]),query_index=0,uv=uv.astype(float),depth=np.full(n,5.))
def run(d,**kw):return generate_pseudo(**d,**kw)
def check_identity(r,b):assert np.array_equal(r['target6'],b) and not r['reliability6'].any()
def main():
    d=inputs(cloud());r=run(d)
    assert r['reliability6'][0]==1 and abs(r['target6'][0]-.1)<1e-12
    assert np.array_equal(r['target6'][3:],d['source_boxes'][0,3:]) and not r['reliability6'][3:].any()
    assert r['query_index']==0 and r['yaw']==0
    # Projection fixture intentionally assigns unique image pixels; it tests G's
    # visibility algebra, not a real camera calibration or photorealistic image.
    pp=inputs(cloud()[cloud()[:,0]>.9]);check_identity(run(pp),pp['source_boxes'][0])
    bad=inputs(cloud());n=len(bad['points']);bad['points']=np.r_[bad['points'],np.tile([100.,100.,1.],(n,1))];bad['uv']=np.r_[bad['uv'],bad['uv']];bad['depth']=np.r_[bad['depth'],np.ones(n)]
    check_identity(run(bad),bad['source_boxes'][0])
    # Same-color evidence at another nonduplicate candidate is ambiguous.
    c=cloud();bg=inputs(np.r_[c,c+[4.,0,0]]);bg['source_boxes']=np.r_[bg['source_boxes'],[[4,0,5,2,1,1]]];bg['text_scores']=np.array([1.,.1]);br=run(bg)
    check_identity(br,bg['source_boxes'][0]);assert br['diagnostics']['reason']=='same_evidence_competing_candidate'
    # A similarly large disconnected same-color background patch in the padded
    # crop is an evidence ambiguity, even without an alternate proposal box.
    clutter=np.array([(x,y,z) for x in np.linspace(-.08,.08,6) for y in np.linspace(.67,.73,6) for z in np.linspace(4.27,4.33,3)])
    cluttered=inputs(np.r_[cloud(),clutter]);cr=run(cluttered)
    check_identity(cr,cluttered['source_boxes'][0]);assert cr['diagnostics']['reason']=='ambiguous_components'
    outside=dict(d,uv=d['uv']+10000);check_identity(run(outside),d['source_boxes'][0])
    behind=dict(d,depth=-d['depth']);check_identity(run(behind),d['source_boxes'][0])
    wrong=dict(d,text='the blue car');check_identity(run(wrong),d['source_boxes'][0])
    assert target_color('the car beside the red bus')[0] is None
    assert target_color('the red car beside the blue car')[0]=='red'
    for caption in ('the not red car','the red and blue car','the purple car','the red car without color','no red car'):
        check_identity(run(dict(d,text=caption)),d['source_boxes'][0])
    check_identity(run(d,mode='shuffled_color'),d['source_boxes'][0])
    pc=run(d,mode='pc_only');assert pc['reliability6'][0]==1
    # Change only image evidence: RGB G stops, PC-only is invariant.
    green=dict(d,rgb=d['rgb'].copy());green['rgb'][d['uv'][:,1].astype(int),d['uv'][:,0].astype(int)]=[0,255,0]
    check_identity(run(green),d['source_boxes'][0]);assert np.array_equal(run(green,mode='pc_only')['target6'],pc['target6'])
    perm=np.random.default_rng(3).permutation(len(d['points']));per=dict(d,points=d['points'][perm],uv=d['uv'][perm],depth=d['depth'][perm]);pr=run(per)
    assert np.array_equal(pr['target6'],r['target6']) and np.array_equal(pr['reliability6'],r['reliability6'])
    assert pr['diagnostics']==r['diagnostics']
    over=run(inputs(cloud(.21)));assert over['reliability6'][0]==0 and over['target6'][0]==0
    false_gt_rejected=False
    try:generate_pseudo(**d,positive_map=np.ones(3))
    except TypeError:false_gt_rejected=True
    assert false_gt_rejected
    b=dict(d,source_boxes=np.array([[0,0,5,-2,1,1.]]));check_identity(run(b),b['source_boxes'][0])
    out={'state':'synthetic_CPU_G_contract_pass_no_real_data_or_task_result','valid_two_sided_x_translation':float(r['target6'][0]),'size_prior_unchanged_w_size_zero':True,'partial_visible_skip':True,'allcloud_pixel_zbuffer_occluder_skip':True,'samecolor_nondeduplicate_competitor_skip':True,'disconnected_samecolor_background_ambiguity_skip':True,'external_projection_FOV_and_positive_depth_required':True,'wrongcolor_relationobject_negation_multicolor_unsupported_checks':True,'fixed_color_cycle_wrongcolor_control_skip':True,'PConly_same_geometry_and_camera_color_invariant':True,'point_permutation_exact_outputs_and_diagnostics':True,'excess_delta_axis_rejected':True,'GT_extra_field_TypeError':True,'negative_source_size_identity_skip':True,'synthetic_projection_scope':'Unique image pixels, algebra fixture only, not actual camera calibration or photo rendering','config':DEFAULT_CONFIG.__dict__,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('multimodal_pseudo.py'),Path(__file__).with_name('G_CONTRACT.md')]},'GPU_model_GT_or_real_RGB_point_files_used':False}
    Path(__file__).with_name('CPU_G_FIXTURE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
