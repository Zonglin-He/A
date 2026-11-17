"""Native camera support and resolution check; no grounding model/task labels."""
from common150 import *
import numpy as np,collections
from PIL import Image
sys.path.insert(0,str(R/'research/round105'));import geometry105 as G
from common150 import R,H,B,rd,sha,dump,check,utc

def proj_B(e,z,wh):
 ca,cb=G.camera(e['A_sensor']),G.camera(e['pair_PRIVATE']['B']);rr=rd(R/e['registration_path']);assert sha(R/e['registration_path'])==e['registration_sha256'];rawA=G.transform_points(z['xyz'][0].astype(float),G.inverse_se3(G.raw_to_model_transform(ca)));rawB=G.transform_points(rawA,G.inverse_se3(np.asarray(rr['forward']['T_raw_B_to_A'])));modB=G.raw_to_model_points(rawB,e['pair_PRIVATE']['B']['platform'],int(e['pair_PRIVATE']['B'].get('suffix') or 0));return G.project_model_points(modB,cb,wh)
def main():
 check();rows=[];region=[];n=0;seal=rd(R/'crossplatform/round145/image/SEALED.json')['files']
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  if not e['has_B']:continue
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;z=np.load(R/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));ev=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));path=R/'crossplatform/round145/image'/name/'banks_PRIVATE'/(sid+'.npz');assert sha(path)==seal[str(path.relative_to(R/'crossplatform/round145/image'))];bb=np.load(path);p=np.load(R/'crossplatform/round149/method'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(p['arm_names'],p['scores']));sensor=e['pair_PRIVATE']['B'];assert sensor['eligible_for_extra_media'] and not sensor['protected_image_content_alias'];wh=Image.open(sensor['imagepath']).size;pr=proj_B(e,z,wh);vb=bb['norm_Bimage_visible'];va=bb['norm_Aimage_visible'];assert (pr['in_image'][vb]).all();gb=G.graph_kernel(pr,wh,vb)
  for k in ['p2i','i2p','visible']:assert np.array_equal(gb[k],bb['norm_Bimage_'+k]),k
  for k in ['p2i_weight','i2p_weight']:assert abs(gb[k]-bb['norm_Bimage_'+k]).max(initial=0)<1e-5
  xyz,inv=np.unique(z['xyz'][0],axis=0,return_inverse=True);visA=np.zeros(len(xyz),bool);visB=visA.copy();np.logical_or.at(visA,inv,va);np.logical_or.at(visB,inv,vb);gs=[np.flatnonzero(ev['assignment']==g) for g in range(int(ev['assignment'].max()+1))];boxes=z['boxes'];inside=(abs(xyz[None]-boxes[:,None,:3])<=np.maximum(boxes[:,None,3:6],1e-6)/2).all(-1);support=np.array([inside[g].any(0) for g in gs]);nA=(support&visA).sum(1);nB=(support&visB).sum(1);C,S=ev['C'],ev['S'];tgroup=int(np.flatnonzero(C)[ev['target'].argmax()]);q0=int(ps['norm_joint_zero'].argmax());g0=int(ev['assignment'][q0]);order=sorted(np.flatnonzero(S),key=lambda g:(-float(ps['norm_joint_zero'][gs[g]].max()),int(g)));sel=[g for g in order if nB[g]>0][:3];rec=dict(source=src,target=tgt,id_PRIVATE=sid,S_groups=int(S.sum()),C_groups=int(C.sum()),RGB_A_qualified_S=int((S&(nA>=4)).sum()),RGB_B_qualified_S=int((S&(nB>=4)).sum()),RGB_either_qualified_S=int((S&((nA>=4)|(nB>=4))).sum()),RGB_neither_qualified_S=int((S&(nA<4)&(nB<4)).sum()),teacher_B_unique_visible_support=int(nB[tgroup]),teacher_A_unique_visible_support=int(nA[tgroup]),teacher_B_no_visible_support=bool(nB[tgroup]==0),teacher_B_below4=bool(nB[tgroup]<4),zero_group_B_unique_visible_support=int(nB[g0]),selected_regions=len(sel));rows.append(rec)
  for rank,g in enumerate(sel):
   ids=np.flatnonzero(support[g][inv]&vb);uv=gb['uv_grid'][ids];span=np.ptp(uv,axis=0);cells=set(gb['cell'][ids].tolist());other=[]
   for h in sel:
    if h==g:continue
    jac=float((support[g]&support[h]).sum()/max((support[g]|support[h]).sum(),1))
    if jac<=.1:
     jj=np.flatnonzero(support[h][inv]&vb);cc=set(gb['cell'][jj].tolist());other.append(len(cells&cc)/max(len(cells),1))
   region.append(dict(source=src,target=tgt,id_PRIVATE=sid,rank=rank+1,unique_visible_support=int(nB[g]),width_cells=float(span[0]),height_cells=float(span[1]),short_cells=float(span.min()),distinct_center_cells=len(cells),max_shared_center_cell_fraction_disjoint_support=float(max(other,default=0)),has_disjoint_competitor=bool(other)))
  n+=1
  if n%64==0:print('RGB_METADATA150',n,flush=True)
 out=[]
 for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
  rr=[r for r in rows if (r['source'],r['target'])==(src,tgt)];rg=[r for r in region if (r['source'],r['target'])==(src,tgt)];out.append(dict(source=src,target=tgt,n=len(rr),teacher_B_zero_visible=sum(r['teacher_B_no_visible_support'] for r in rr),teacher_B_below4=sum(r['teacher_B_below4'] for r in rr),teacher_A_ge4_B_lt4=sum(r['teacher_A_unique_visible_support']>=4 and r['teacher_B_below4'] for r in rr),mean_S_RGB_B_qualified_fraction=float(np.mean([r['RGB_B_qualified_S']/r['S_groups'] for r in rr])),mean_S_RGB_either_qualified_fraction=float(np.mean([r['RGB_either_qualified_S']/r['S_groups'] for r in rr])),selected_regions=len(rg),median_short_cells=float(np.median([r['short_cells'] for r in rg])),fraction_short_lt3=float(np.mean([r['short_cells']<3 for r in rg])),fraction_short_lt1=float(np.mean([r['short_cells']<1 for r in rg])),fraction_with_shared_center_cell=float(np.mean([r['max_shared_center_cell_fraction_disjoint_support']>0 for r in rg])),mean_shared_center_cell_fraction=float(np.mean([r['max_shared_center_cell_fraction_disjoint_support'] for r in rg]))))
 dump(H/'METADATA_ROWS_PRIVATE.json',rows);dump(H/'REGION_METADATA_ROWS_PRIVATE.json',region);dump(H/'METADATA_SUMMARY.json',dict(status='COMPLETE_NO_MODELS_NO_TASK_GT',n=n,summary=out,graph_B_reproduces_pinned_R145=True,only_original_1024seed_support_not_instance_identity=True,feature_grid=[48,32],no_new_crop_pixels_or_model_inference=True));print('RGB_METADATA150_DONE',n,out,flush=True)
if __name__=='__main__':main()
