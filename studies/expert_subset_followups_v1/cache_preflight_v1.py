from common_v1 import *
import adapter_v1 as adapter

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return set(a)==set(b)and all(same(v,b[k])for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
 return a==b

def main():
 guard();torch.set_num_threads(1);cases=[]
 for ds in ['hc2','vidstg']:
  for row in rows(ds)[:2]:
   cap,raw,frames,t,refs,proof=read_inputs(ds,row);a,c=build(cap,raw,t,ds);b,cc=build(cap,raw,t,ds);assert all(np.array_equal(v,b[k])for k,v in a.items());assert np.array_equal(a['Current'],refs['CurrentOurs'])and np.array_equal(a['Current_interval'],refs['CurrentOurs_interval'])
   causal=dict(t);causal.update(native_same_track_all8=False,native_causal_state_carried_all8=True,episodic_reset=True,model_steps=8,selected_queries=[int(t['selected_query'])]*8,expert_model='OnlineRefer-R50-RefYTVOS-sparse-eight-causal-adapter');del causal['selected_query'];ca,cb=build(cap,raw,causal,ds);assert all(np.array_equal(v,ca[k])for k,v in a.items())
   cases.append(dict(dataset=ds,original_global_expert_Current_all_arrays_exact=True,causal_contract_only_same_selected_masks_all_arrays_exact=True,no_new_actual_expert_validation_claim=True))
 logits=torch.full((1,8,5,1),-3.);logits[:,:,2]=4;logits[:,0,4]=8;masks=torch.full((1,8,1,4,4),-10.);masks[:,1:,0,:2,1:3]=10
 out=dict(pred_logits=logits[:,:,2:3].clone(),pred_masks=masks,audit_all_native_query_logits=logits);t=adapter.select('sgmg_videoswin_t',out,[8,3,4,4],[4,4]);assert t['selected_query']==2 and t['empty_indices']==[0]and torch.equal(t['boxes'][0],torch.tensor([.5,.5,1.,1.]));assert same(t,adapter.select('sgmg_videoswin_t',out,[8,3,4,4],[4,4]))
 steps=[];track=None
 for i in range(8):
  q=5 if i==0 else 1;ll=torch.full((1,1,q,1),-3.);ll[:,:,3 if i==0 else 0]=4.;bb=torch.full((1,1,q,4),.4);mm=torch.full((1,1,q,4,4),-10.);j=3 if i==0 else 0;mm[:,:,j,:2,1:3]=10;emb=torch.arange(q*4,dtype=torch.float32).reshape(q,4);before=track if track is not None else dict(pos_embedding=emb,track_embedding=emb.clone(),ref_pts=emb[:,:2]);after=dict(pos_embedding=emb[j:j+1].clone(),track_embedding=emb[j:j+1].clone(),ref_pts=emb[j:j+1,:2].clone());steps.append(dict(frame_index=i,outputs=dict(pred_logits=ll,pred_boxes=bb,pred_masks=mm,pos_embedding=emb),track_before=before,track_after=after));track=after
 out=dict(steps=steps,model_steps=8,episodic_reset=True);t=adapter.select('onlinerefer_r50',out,[8,3,4,4],[4,4]);assert t['selected_queries']==[3]+[0]*7 and len(t['native_query_scores'][0])==5 and all(len(x)==1 for x in t['native_query_scores'][1:]);assert same(t,adapter.select('onlinerefer_r50',out,[8,3,4,4],[4,4]));assert not torch.cuda.is_initialized()
 atomic(D/'cache_and_interface_preflight.json',dict(status='passed_four_original_Source_cache_Current_exact_causal_interface_only_reverse_exact_and_synthetic_native_selection_masks_boxes',cases=cases,adapter_sha256=sha(S/'adapter_v1.py'),common_sha256=sha(S/'common_v1.py'),code_sha256=sha(Path(__file__)),new_model=0,new_GPU=0,new_Source=0,new_expert=0,new_GT=0,not_actual_new_expert_GPU_bridge=True));print('four actual cached Source/current cases and synthetic new expert interfaces passed')
if __name__=='__main__':main()
