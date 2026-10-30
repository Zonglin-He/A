"""Parent-scheduled GPU source inference only; reads GT-free input manifests."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parent;P=R.parent
sys.path[:0]=[str(P),str(P/'upstream')]
from tta import text_span
from models.bdetr import BeaUTyDETR

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args()
    planpath=R/'PLAN_LOCK.json';plan=json.loads(planpath.read_text());assert sha(planpath)==(R/'PLAN_LOCK.sha256').read_text().strip()
    for name,h in plan['source_hashes'].items():assert sha(P/name)==h,name
    for name,h in plan['round2_code_hashes'].items():assert sha(R/name)==h,name
    manifest=R/'inputs'/(a.cohort+'_manifest.json');assert sha(manifest)==plan['input_manifest_sha256'][a.cohort];samples=json.loads(manifest.read_text())
    # Confirmation model evaluation is gated by final freeze before its execution.
    if a.cohort=='confirm':assert (R/'CONFIRMATION_RELEASE.json').is_file(),'Need explicit final-method freeze'
    dest=R/(a.cohort+'_source');dest.mkdir(exist_ok=False);(dest/'registered_plan.json').write_bytes(planpath.read_bytes())
    torch.set_num_threads(4);torch.manual_seed(plan['seed']);os.chdir(P/'upstream')
    model=BeaUTyDETR(num_class=256,num_obj_class=485,input_feature_dim=3,num_queries=256,num_decoder_layers=6,contrastive_align_loss=True,butd=False,self_attend=True,roberta_config_only=True)
    checkpoint=P/'assets/final_6384/ckpt_6384.pth';assert sha(checkpoint)==plan['checkpoint_sha256']
    with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    state={k.removeprefix('module.'):v for k,v in state.items()};assert len(state)==1034
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point());model.load_state_dict(state,strict=True);del state
    model.requires_grad_(False).cuda().eval();cache={}
    def hook(module,arguments):cache['features']=arguments[0].detach().transpose(1,2).contiguous()
    handle=model.prediction_heads[-1].register_forward_pre_hook(hook);traces=[];first=None
    def inference(sample):
        pp=Path(sample['point_path']);assert sha(pp)==sample['point_sha256'];pc=np.load(pp,allow_pickle=False)
        inp={'point_clouds':torch.as_tensor(pc,dtype=torch.float32,device='cuda')[None],'text':[sample['text']]}
        with torch.inference_mode():
            output=model(inp);boxes=torch.cat([output['last_center'],output['last_pred_size']],-1)[0]
            tokens=((output['last_proj_queries']@output['proj_tokens'].transpose(-1,-2))/.07).softmax(-1)[0]
            mask,noun=text_span(model.tokenizer,sample['text']);score=(tokens*mask.to(tokens.device)[None]).sum(-1)
            assert boxes.shape==(256,6) and cache['features'].shape==(1,256,288)
            assert torch.isfinite(boxes).all() and torch.isfinite(score).all() and torch.isfinite(cache['features']).all()
            return {'boxes':boxes.cpu().numpy(),'semantic_scores':score.cpu().numpy(),'token_scores':tokens.cpu().numpy(),'query_features':cache['features'][0].cpu().numpy()},noun
    try:
        for sample in samples:
            torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats();arrays,noun=inference(sample);torch.cuda.synchronize()
            np.savez_compressed(dest/(sample['id']+'.npz'),**arrays)
            trace={'id':sample['id'],'platform':sample['platform'],'top_query':int(arrays['semantic_scores'].argmax()),'noun':noun,'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'invalid_source_size_candidates':int(np.any(arrays['boxes'][:,3:6]<=0,axis=1).sum())};traces.append(trace);print(json.dumps(trace),flush=True)
            if first is None:first={k:v.copy() for k,v in arrays.items()}
        repeated,_=inference(samples[0]);diff={k:float(np.max(np.abs(v-repeated[k]))) for k,v in first.items()};assert max(diff.values())<=1e-6,diff
        (dest/'source_repeat_control.json').write_text(json.dumps({'maximum_differences':diff,'bitexact':all(x==0 for x in diff.values()),'no_adaptation_in_source_process':True},indent=2))
        (dest/'trace.json').write_text(json.dumps(traces,indent=2));seal={'n':len(traces),'input_manifest_sha256':sha(manifest),'plan_sha256':sha(planpath),'files':{p.name:sha(p) for p in dest.glob('*.npz')},'model_strict_loaded':1034,'numeric_GT_read':False,'source_repeat_control_sha256':sha(dest/'source_repeat_control.json')};(dest/'sealed.json').write_text(json.dumps(seal,indent=2));print('SOURCE_SEALED',len(traces),flush=True)
    finally:handle.remove()
if __name__=='__main__':main()
