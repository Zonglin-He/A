"""Corrected physical-time source Q0; official loader/decode smoke is required."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from vg_tta.external_qualification_io import *

def register(name,smoke):
    smoke=OUT/smoke;done,_=verify_seal(smoke);assert done['loader_exact']
    files=read(DOWNLOAD/'MODEL_DOWNLOAD_COMPLETE.json');assert files['all_lfs_hashes_match']
    paths=[Path(__file__),ROOT/'vg_tta/llava_st_teacher.py',ROOT/'vg_tta/external_privileged_views.py',ROOT/'vg_tta/exact_frame_decode_audit_v2.py',
        smoke/'COMPLETE.json',smoke/'COMPARISON.json',smoke/'PREDICTIONS_SEAL.json',DOWNLOAD/'MODEL_DOWNLOAD_COMPLETE.json']
    paths += [CK/f['path'] for f in files['files']]+list((OFFICIAL/'llava').rglob('*.py'))+list((OFFICIAL/'inference').rglob('*.py'))
    register_base(name,{'stage':'source_external_evidence_Q0','phase_seconds':3600,'seed':20260928,
        'decode':done['qualification_decode'],'smoke':str(smoke),'teacher_revision':files['revision'],
        'time_contract':'linear physical clip bounds; nearest observed RGB for each uniform physical slot; no unseen pixels',
        'aggregation':'per-observation valid-box coordinate median; errors and raw tuples retained',
        'gap_limit':None,'disagreement_threshold':None},read(PANEL/'INPUTS.json'),paths)

def run(name):
    dest=OUT/name
    with allocation(dest) as (cfg,guard):
        import torch,numpy as np,hashlib
        from vg_tta.llava_st_teacher import load,predict
        from vg_tta.external_privileged_views import repeated_frame_indices,parse_teacher_text,dense_spatial_support
        from vg_tta.exact_frame_decode_audit_v2 import decode
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        tokenizer,model,processor,info=load(CK)
        write(dest/'MODEL_LOADING.json',{'info':info,'max_frame':model.config.max_frame,'all_parameters_frozen':all(not p.requires_grad for p in model.parameters())})
        for i,row in enumerate(read(dest/'INPUTS.json')):
            guard();frames,ids=decode(row['input']);bounds=[row['input']['start_frame'],row['input']['end_frame']]
            indices,repeated=repeated_frame_indices(ids,clip_bounds=bounds);ep=dest/'episodes'/f'{i:02}'
            write(ep/'INPUT.json',{'key':row['key'],'source':row['source'],'frame_ids':ids,'clip_bounds':bounds,
                'pixel_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),'original_video_sha256':row['input']['video_sha256'],
                'repeated_observation_indices':indices.tolist(),'physical_slots':np.linspace(*bounds,100).tolist(),
                'teacher_pixel_frame_ids':repeated.tolist(),'frame_shape':list(frames.shape),'no_unseen_pixels':True})
            pred,pixels=predict(tokenizer,model,processor,frames[indices],row['input']['caption'],OFFICIAL,decode=cfg['decode'],seed=cfg['seed'])
            pred['processed_pixel_sha256']=tensor_sha(pixels);write(ep/'RAW_PREDICTION.json',pred)
            evidence=parse_teacher_text(pred['raw_text'],ids,clip_bounds=bounds);_,diagnostic=dense_spatial_support(ids,evidence)
            write(ep/'EVIDENCE.json',evidence);write(ep/'SUPPORT.json',diagnostic)
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            write(ep/'COMPLETE.json',{'key':row['key'],'optimizer_steps':0,'format_errors':evidence['errors']})
            print('TEACHER_Q0',i+1,16,'errors',evidence['errors'],flush=True)
        write(dest/'COMPLETE.json',{'status':'inference_complete_not_scored','queries':16,'parents':16,'optimizer_steps':0,
            'seal_sha':seal(dest,16),'source_GT_read':False,'target_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);p.add_argument('--smoke')
    a=p.parse_args();register(a.name,a.smoke) if a.action=='register' else run(a.name)
