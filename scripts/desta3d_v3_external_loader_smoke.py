"""Two fixed source inputs: actual official vs efficient loaders and decode recipes."""
import argparse,gc,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import *

def register(name):
    files=read(DOWNLOAD/'MODEL_DOWNLOAD_COMPLETE.json');assert files['all_lfs_hashes_match']
    dep=read(OUT/'siglip_download/MODEL_DOWNLOAD_COMPLETE.json');assert dep['all_lfs_hashes_match']
    paths=[Path(__file__),ROOT/'vg_tta/llava_st_teacher.py',ROOT/'vg_tta/external_privileged_views.py',
           ROOT/'vg_tta/exact_frame_decode_audit_v2.py',DOWNLOAD/'MODEL_DOWNLOAD_COMPLETE.json',OUT/'siglip_download/MODEL_DOWNLOAD_COMPLETE.json']
    paths += [CK/f['path'] for f in files['files']]+[SIGLIP/f['path'] for f in dep['files']]
    paths += [OFFICIAL/'inference/config.yaml']
    paths += list((OFFICIAL/'llava').rglob('*.py'))+list((OFFICIAL/'inference').rglob('*.py'))
    register_base(name,{'stage':'loader_decode_smoke','phase_seconds':1800,'loaders':['efficient','official'],
        'decodes':['greedy','official'],'seed':20260928,'vision_atol':0.,'vision_rtol':0.,
        'acceptance':'identical model parameter hashes, processor tensor, first vision features and generated IDs for each recipe',
        'qualification_decode_rule':'greedy only if both fixed queries token-exact to official recipe; otherwise official temp .01',
        'native_predictions':False},read(PANEL/'INPUTS.json')[:2],paths)

def run(name):
    dest=OUT/name
    with allocation(dest) as (cfg,guard):
        import torch,numpy as np
        from vg_tta.llava_st_teacher import load,load_official,predict
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.external_privileged_views import repeated_frame_indices,parse_teacher_text
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        results={};parameter_hashes={}
        for loader in cfg['loaders']:
            guard()
            tokenizer,model,processor,info=load(CK) if loader=='efficient' else load_official(CK,SIGLIP,OFFICIAL)
            parameter_hashes[loader]={n:tensor_sha(p) for n,p in model.named_parameters()}
            write(dest/(loader+'_LOADING.json'),{'info':info,'max_frame':model.config.max_frame,'parameter_hashes':parameter_hashes[loader]})
            results[loader]=[]
            for i,row in enumerate(read(dest/'INPUTS.json')):
                guard();frames,ids=decode(row['input']);bounds=[row['input']['start_frame'],row['input']['end_frame']]
                indices,repeated=repeated_frame_indices(ids,clip_bounds=bounds)
                ep=dest/'episodes'/loader/f'{i:02}'
                write(ep/'INPUT.json',{'key':row['key'],'frame_ids':ids,'bounds':bounds,'repeat_indices':indices.tolist(),
                    'physical_slots':np.linspace(*bounds,100).tolist(),'pixel_sha256':__import__('hashlib').sha256(frames.tobytes()).hexdigest()})
                outputs={}
                for recipe in cfg['decodes']:
                    guard();captured=[]
                    def vision_hook(module,args,result):
                        if not captured:captured.append(result[0].detach().cpu())
                    handle=model.get_vision_tower().register_forward_hook(vision_hook)
                    try:pred,pixels=predict(tokenizer,model,processor,frames[indices],row['input']['caption'],OFFICIAL,decode=recipe,seed=cfg['seed'])
                    finally:handle.remove()
                    write(ep/(recipe+'_RAW.json'),pred)
                    assert len(captured)==1 and torch.isfinite(captured[0]).all()
                    torch.save({'first_vision_features':captured[0],'pixels':pixels},ep/(recipe+'_FEATURES.pt'))
                    write(ep/(recipe+'_EVIDENCE.json'),parse_teacher_text(pred['raw_text'],ids,clip_bounds=bounds))
                    outputs[recipe]={'tokens':pred['output_token_ids'],'input_ids':pred['input_token_ids'],
                        'pixels_sha':tensor_sha(pixels),'vision_sha':tensor_sha(captured[0])}
                results[loader].append(outputs)
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            del model,tokenizer,processor,pixels,captured;gc.collect();torch.cuda.empty_cache()
        write(dest/'COMPARISON.json',{'results':results,'full_parameter_hashes_exact':parameter_hashes['efficient']==parameter_hashes['official']})
        assert parameter_hashes['efficient']==parameter_hashes['official'],'loader parameter mismatch'
        assert results['efficient']==results['official'],'actual loader output mismatch'
        same=all(r['greedy']['tokens']==r['official']['tokens'] for r in results['official'])
        write(dest/'COMPLETE.json',{'status':'engineering_equivalence_passed','seal_sha':seal(dest,8),'loader_exact':True,
            'decode_same_on_two_inputs':same,'qualification_decode':'greedy' if same else 'official','source_GT_read':False,'target_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True)
    a=p.parse_args();globals()[a.action](a.name)
