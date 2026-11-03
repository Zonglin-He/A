"""No-model synthetic selected-track/empty-interface contract before target runs."""
from pathlib import Path
import torch,json,hashlib
import adapter_v1 as a
D=Path(__file__).resolve().parents[2]/'outputs/expert_subset_expansion_v1'
def main():
 torch.set_num_threads(1);scores=torch.zeros(8,1,50,2);scores[...,1]=2;scores[:,0,7,0]=8;scores[0,0,9,0]=20
 masks=torch.full((8,1,50,4,4),-10.);masks[1:,0,7,0:2,1:3]=10
 t=a.select('mttr',dict(pred_is_referred=scores,pred_masks=masks),[8,3,4,4],[4,4]);assert t['selected_query']==7 and t['empty_indices']==[0]and t['empty_box_conversion']==1
 assert torch.equal(t['boxes'][0],torch.tensor([.5,.5,1.,1.]))and torch.equal(t['boxes'][1:],torch.tensor([[.5,.25,.5,.5]]).repeat(7,1));assert torch.equal(t['native_mask_logits'],masks[:,0,7])
 twice=a.select('mttr',dict(pred_is_referred=scores,pred_masks=masks),[8,3,4,4],[4,4]);assert all(torch.equal(v,twice[k])if torch.is_tensor(v)else v==twice[k]for k,v in t.items());assert not torch.cuda.is_initialized()
 receipt=dict(status='passed_synthetic_global_track_not_perframe_threshold_halfopen_and_empty_no_Source_interface',new_model=0,new_GPU=0,new_Source=0,new_expert=0,new_GT=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest());(D/'interface_contract.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
