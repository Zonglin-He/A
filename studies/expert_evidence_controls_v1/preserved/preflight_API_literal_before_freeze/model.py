"""Pinned official SAM2.1 Hiera-L; no source edits, full strict state coverage."""
from common import *
def modules():
 sys.path.insert(0,str(V));import sam2
 assert Path(sam2.__file__).resolve().is_relative_to(V.resolve());sam2.__path__.append(str(EXT.parent))
 import sam2.sam2_video_predictor as predictor
 import sam2.utils.misc as misc
 import sam2._C as extension
 assert sha(EXT)==read(R/'outputs/edgetam_comparison_v1/extension_reuse_receipt.json')['module_sha256']
 assert sha(V/'sam2/csrc/connected_components.cu')==read(R/'outputs/edgetam_comparison_v1/extension_reuse_receipt.json')['original_cuda_source_SHA256']
 return predictor,misc,extension
def build_cpu():
 modules();from sam2.build_sam import build_sam2_video_predictor
 tick=time.perf_counter();model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(WEIGHT),device='cpu',mode='eval',apply_postprocessing=True,vos_optimized=False)
 state=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)['model'];actual=model.state_dict();assert set(actual)==set(state)
 for k,v in actual.items():assert torch.equal(v.cpu(),state[k]),k
 assert model.fill_hole_area==8;model.eval().requires_grad_(False)
 return model,dict(strict_all_keys_tensors_exact=True,tensors=len(state),parameters=sum(v.numel()for v in model.parameters()),seconds=time.perf_counter()-tick,checkpoint_sha256=sha(WEIGHT),source_commit=read(R/'outputs/review_strengthening_v1/sam2_source_lock.json')['commit'],actual_forward=0)
def mask_box(mask):
 # An independent implementation in report.py verifies all pixels/boxes.
 yy,xx=np.nonzero(mask);h,w=mask.shape
 if len(xx)==0:return np.zeros(4,np.float64)
 x0,x1,y0,y1=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
 return np.array([(x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h],np.float64)
def propagate(model,module,frames,positions,prompts,checkpoint):
 from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader
 assert prompts.dtype==np.bool_ and prompts.shape==(8,*frames.shape[1:3])
 original=module.load_video_frames
 try:
  module.load_video_frames=lossless_rgb_loader
  state=model.init_state(frames,offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
 finally:module.load_video_frames=original
 for j in sorted(range(8),key=lambda j:positions[j]):
  model.add_new_mask(state,frame_idx=int(positions[j]),obj_id=1,mask=prompts[j])
 assert set(state['temp_output_dict_per_obj'][0]['cond_frame_outputs'])==set(map(int,positions))
 assert not state['point_inputs_per_obj'][0] and set(state['mask_inputs_per_obj'][0])==set(map(int,positions))
 out={};h,w=frames.shape[1:3]
 # Start0 native forward with all8 conditioning masks, exactly as previous
 # validated native API; no custom fusion or best-direction selection.
 for i,obj,scores in model.propagate_in_video(state,start_frame_idx=0,reverse=False):
  cpu=checkpoint(i,scores)
  assert obj==[1]and cpu.shape==(1,1,h,w)and torch.isfinite(cpu).all()and int(i)not in out
  out[int(i)]=mask_box((cpu[0,0]>0).numpy())
 assert set(out)==set(range(len(frames)))
 return np.stack([out[i]for i in range(len(frames))])
