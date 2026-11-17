"""Model-free full-cohort projection/grid audit and pixel provenance lock."""
from geometry152 import *
from audit_geometry152 import independent_specs,independent_merge
from common152 import R,H,B,rd,sha,dump,check,utc
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');records=[];mx=0.;active=0
 for i,e in enumerate(reg):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];r=dict(source=src,target=tgt,id_PRIVATE=sid,has_B=e['has_B'],arms={})
  if e['has_B']:
   z=load(R/'crossplatform/round131/formal'/(src+'_to_'+tgt)/'prefix_PRIVATE'/(sid+'.npz'));ss=specs(e,z);ind=independent_specs(e,z,False)
   for arm,gg in ss.items():
    g=merged(gg);a=independent_merge(ind[arm])
    for k,v in a.items():
     if k.endswith('weight'):
      err=float(abs(v-g[k]).max(initial=0));mx=max(mx,err);assert err<1e-5
     else:assert np.array_equal(v,g[k]),(arm,k)
    r['arms'][arm]=dict(graph_hashes={k:ah(g[k]) for k in ['p2i','p2i_weight','i2p','i2p_weight','visible']},pixel_hashes=[ah(q['image']) for q in gg],visible_points=[int(q['visible'].sum()) for q in gg],image_shape=list(gg[0]['image'].shape),image_tokens=len(g['i2p']))
   r['new_detail_changes_pixels']=[not np.array_equal(a['image'],b['image']) for a,b in zip(ss['full_high'],ss['full_low'])];active+=1
  records.append(r)
  if (i+1)%128==0:print('PREP152',i+1,active,flush=True)
 dump(H/'GEOMETRY_ROWS_PRIVATE.json',records);dump(H/'PREPARATION_AUDIT.json',dict(status='PASS',n=len(records),B=active,max_graph_weight_error=mx,high_image_size=[768,512],per_bank_tokens=6144,total_AB_tokens=12288,independent_camera_grid_reconstruction=True,GT_read=False,models_run=False));print('PREP152_COMPLETE',len(records),active,mx,flush=True)
if __name__=='__main__':main()
