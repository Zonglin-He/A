"""Capture frozen source-model features without target boxes or annotated classes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
import numpy as np
import torch
from paths import ROOT, PRIV, OUT, PLATFORMS, read, write, digest

sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'studies/3eed_v2_ordinary_val_subject_ranking_20261119'))
from scope183 import ordinary, VIEWS
from crossplatform.candidate_frontend import CandidateFrontend

@torch.inference_mode()
def run():
    roster=read(PRIV/'panel.json')
    assert digest(PRIV/'panel.json')==read(OUT/'panel_seal.json')['panel']
    original=ordinary()
    folder=PRIV/'frozen'
    folder.mkdir(exist_ok=True)
    files={}
    for source in PLATFORMS:
        front=CandidateFrontend(source)
        from runner_helpers import sample_points
        from utils.transform_waymo import transform_to_front_view
        from model51 import SENTINEL
        for item in roster:
            t=item['target']
            if t==source:continue
            row=original[t][item['ordinal']]
            points,_,_=sample_points(row)
            if t=='waymo':points,_=transform_to_front_view(points,np.empty((0,7)),VIEWS[row['suffix']])
            pred=front.base({'point_clouds':torch.as_tensor(np.ascontiguousarray(points,dtype=np.float32),device='cuda')[None],
                             'text':[row['caption']+SENTINEL]})
            key=source+'_'+item['uid']
            np.savez(folder/(key+'.npz'),q=pred['last_proj_queries'][0].cpu().numpy(),
                token=pred['proj_tokens'][0].cpu().numpy(),raw=torch.cat([pred['last_center'],pred['last_pred_size']],-1)[0].cpu().numpy())
            files[key]=digest(folder/(key+'.npz'))
        print('frozen source completed',source,flush=True)
        del front,pred
        torch.cuda.empty_cache()
    assert len(files)==2*len(roster)
    write(PRIV/'frozen_hashes.json',files)
    write(OUT/'frozen_seal.json',dict(features_sha=digest(PRIV/'frozen_hashes.json'),source_features=len(files),
        panel=digest(PRIV/'panel.json'),code=digest(__file__),source_assets=digest(ROOT/'crossplatform/single_frame_sources.json'),
        flag='Published last-layer BBF category mask applied only in post-seal independent baseline decoding.'))

if __name__=='__main__':run()
