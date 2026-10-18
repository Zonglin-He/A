"""Independent CPU checks of actual region-conditioned predictions and metrics."""
from pathlib import Path
import os,sys,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_object_condition32_v1 import D,P,PRE,PREFIX,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import torch,numpy as np
 torch.set_num_threads(4)
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 p=verify();result=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in result['rows']};checks=collections.Counter();err=0.;norms={}
 for ds,file in [('hc2','TASTVG_VidSTG.pth'),('vidstg','TASTVG_HCSTVG2.pth')]:
  z=torch.load(ROOT/'checkpoints'/file,map_location='cpu',weights_only=False,mmap=True)['model_ema'];norm=torch.nn.LayerNorm(256)
  norm.load_state_dict({k:z['ground_encoder.encoder.norm.'+k]for k in ['weight','bias']});norms[ds]=norm.eval().requires_grad_(False)
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));c=load(PREFIX/(a+'.pt'))
  assert np.array_equal(o['outputs']['E_MBR'],np.asarray(e['E']))and torch.equal(o['values']['E_MBR']['native_boxes'],s['Source']);checks['zero_Source_E_exact']+=1
  assert np.array_equal(o['outputs']['Object_temporal'],o['outputs']['E_MBR']);checks['temporal_only_spatial_fixed']+=1
  assert np.array_equal(o['outputs']['DINO_region_spatial'],o['outputs']['Spatial_then_temporal']);checks['matched_spatial_joint']+=1
  selected=[];targets=[]
  from vg_tta.stvg_shared_text_query32_v1 import input_pool
  for f,k in zip(input_pool(s['expert']),e['estimate']['selected']):
   if k>=0:selected.append(f['position']);targets.append(f['boxes'][k])
  assert selected==o['positions']
  assert np.allclose(np.asarray(targets).reshape(-1,4),o['targets'],atol=3e-8,rtol=1e-7);checks['same_E_selected_DINO_references']+=1
  for arm,v in o['values'].items():
   assert v['gates']==o['values']['E_MBR']['gates'];checks['native_frame_routing_exact']+=1
   if arm in ['E_MBR','E_region_spatial','DINO_region_spatial']:
    assert all(torch.equal(x,y)for x,y in zip(v['logits'],s['logits']));checks['fixed_spatial_arm_time_logits']+=1
   for tr in v['traces']:
    vi=tr['view'];view=c['views'][vi];H=norms[ds](view['prefix'].float()).detach();h,w=view['info']['fea_map_size'];l=h*w
    for kind in ['spatial','temporal']:
     t=tr[kind]
     if not t or t['noop']:continue
     Hpart=H[:l]if kind=='spatial'else H[-l:];feature=Hpart.permute(1,0,2)[t['positions']]
     coef=t['coefficients'];raw=(feature*coef[...,None]).sum((0,1));expected=raw*(t['native_query'].norm()/raw.norm());de=float((expected-t['conditioned_query']).abs().max());err=max(err,de)
     assert de<1e-5,(a,arm,kind,de);checks['CPU_reconstructed_conditioned_query']+=1
     assert abs(float(coef.sum()-t['native_mass']))<2e-6 and abs(float(t['conditioned_norm']-t['native_norm']))<2e-5;checks['coefficient_mass_query_norm']+=1
     region=E=None
     if kind=='spatial':region=o['outputs']['E_MBR'][o['positions']]if arm=='E_region_spatial'else o['targets'].numpy();local=[j for j,i in enumerate(o['positions'])if i%2==vi];region=np.asarray(region)[local]
     else:region=o['outputs']['E_MBR'if arm=='Object_temporal'else'DINO_region_spatial'][vi::2];assert t['positions']==list(range(H.shape[1]));checks['full_video_motion_support']+=1
     # Independent cell/box intersection using explicit two-dimensional loops.
     valid=(~view['info']['encoded_mask'][:,:l]if kind=='spatial'else~view['info']['encoded_mask'][:,-l:])[t['positions']]
     for j,b in enumerate(region):
      vv=valid[j].reshape(h,w).numpy();vh=vv.any(1).sum();vw=vv.any(0).sum();cx,cy,bw,bh=b;left,right=(cx-bw/2)*vw,(cx+bw/2)*vw;top,bottom=(cy-bh/2)*vh,(cy+bh/2)*vh
      cover=np.array([[max(0,min(x+1,right)-max(x,left))*max(0,min(y+1,bottom)-max(y,top))if vv[y,x]else 0 for x in range(w)]for y in range(h)])
      assert np.max(abs(cover.flatten()-t['coverage'][j].numpy()))<2e-5;checks['independent_actual_fractional_ROI']+=1
   if arm in o['distributions']:
    g=probabilities(c['records'],v['logits']);M=independent_cdf(g,o['frame_ids']);assert np.array_equal(M,o['distributions'][arm])
    assert final_mbr(o['frame_ids'],M)['interval']==o['intervals'][arm];checks['full_physical_MBR_recomputed']+=1
    # Verify returned MBR and deterministic spaced alternatives via brute pairwise tIoU.
    ids=np.array(o['frame_ids']);ia,ib=np.nonzero(M>0);ps=ids[ia];pe=ids[ib]+1;mass=M[ia,ib];iv=o['intervals'][arm]
    def utility(x,y):return float((mass*np.maximum(0,np.minimum(y,pe)-np.maximum(x,ps))/(np.maximum(y,pe)-np.minimum(x,ps))).sum())
    best=utility(*iv)
    for j in np.linspace(0,len(ia)-1,min(16,len(ia)),dtype=int):assert utility(int(ps[j]),int(pe[j]))<=best+1e-12;checks['brute_tIoU_utility_controls']+=1
  if not selected:assert all(np.array_equal(b,o['outputs']['E_MBR'])and o['intervals'][arm]==o['intervals']['E_MBR']for arm,b in o['outputs'].items());checks['no_reference_all_arm_noop']+=1
  row=plans[ds][q];gg=gt[ds][0][q];span=gt[ds][1][q];fid=np.array(sorted(gg));truth=np.array([gg[int(i)]for i in fid]);ids=np.array(o['frame_ids']);planned=np.array(o['planned_positions'])
  for arm,b in o['outputs'].items():
   iv=o['intervals'][arm];xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
   if ds=='hc2':xy=np.maximum(xy,0)
   pred=np.column_stack([np.interp(fid,ids,xy[:,j])for j in range(4)]);inter=np.maximum(0,np.minimum(pred[:,2:],truth[:,2:])-np.maximum(pred[:,:2],truth[:,:2])).prod(1);union=np.maximum(0,pred[:,2:]-pred[:,:2]).prod(1)+np.maximum(0,truth[:,2:]-truth[:,:2]).prod(1)-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);iou[(fid<ids[0])|(fid>ids[-1])]=0
   den=max(max(iv[1],span[1])-min(iv[0],span[0]),1);inside=(fid>=max(iv[0],span[0]))&(fid<min(iv[1],span[1]));tests=dict(v=float(iou[inside].sum()/den),t=max(0,min(iv[1],span[1])-max(iv[0],span[0]))/den,s=float(iou.mean()))
   actual=np.isin(fid,ids);observed=np.isin(fid,ids[planned]);dist=abs(fid[:,None]-ids[planned][None]).min(1)/max(1,ids[-1]-ids[0])
   for key,mask in [('observed_actual_s',observed),('unobserved_actual_s',actual&~observed),('near_actual_s',actual&~observed&(dist<=.125)),('far_actual_s',actual&~observed&(dist>.125))]:tests[key]=float(iou[mask].mean())if mask.any()else None
   for k,v in tests.items():
    got=rows[a]['values'][arm][k];assert got is None if v is None else abs(got-v)<2e-10,(a,arm,k,v,got);checks['independent_official_and_actual_metric_scalar']+=1
 verify(full=True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_CPU_GPU_query_difference=err,all32_before_GT=True,source_prefix_hashes_verified=True,pretrained_weights_unchanged=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),err,flush=True)

if __name__=='__main__':main()
