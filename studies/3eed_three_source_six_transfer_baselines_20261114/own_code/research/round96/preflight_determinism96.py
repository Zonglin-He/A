"""No-GT repeated-forward prerequisite with fixed FP32 evaluation kernels."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import configure,model,dataset,loader,to_cuda,torch

def main():
 configure({'seed':0,'torch_num_threads':4});torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);m=model().cuda().requires_grad_(False).eval();c=torch.load(R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth',map_location='cpu',weights_only=True);m.load_state_dict(c['model']);del c;ds=dataset('drone','val');g=torch.Generator().manual_seed(0);dl=loader(ds,'val',g,{'num_workers':4});b=None
 for b in dl:pass
 b=to_cuda(b);outs=[]
 with torch.no_grad():
  for k in range(2):
   o=m({'point_clouds':b['point_clouds'].float(),'text':b['utterances']});outs.append({h:torch.cat([o[h+'center'],o[h+'pred_size']],-1).cpu().clone() for h in ['last_','proposal_']})
 err=max(float(abs(outs[0][h]-outs[1][h]).max()) for h in outs[0]);out={'state':'PASS_R96_FIXED_FP32_REPEATED_FORWARD' if err<1e-5 else 'FAIL_R96_FIXED_FP32_REPEATED_FORWARD','max_box_error':err,'batch_n':len(b['utterances']),'task_GT_numeric_scoring':False,'source_weights_changed':False,'CUDNN_benchmark':False,'TF32':False,'deterministic_algorithms':True};(H/'experiment/FP32_EVAL_PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n');print(out);assert err<1e-5
if __name__=='__main__':main()
