"""Reproduce the released baseline's category-assisted token mask on sealed features."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import re
import sys
from unittest.mock import patch

import ijson
import numpy as np
import torch
from transformers import RobertaTokenizerFast
from paths import ROOT,PRIV,OUT,read,write,digest,PLATFORMS

sys.path[:0]=[str(ROOT),str(ROOT/'research/round160'),str(ROOT/'crossplatform/round3/p0/upstream'),
              str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round20/runtime')]
from decode160 import (GroundingEvaluator, Joint3DDataset, WAYMO_SYNONYMS, M3ED_SYNONYMS,
                       get_positive_map, grounding_evaluator, skip_metric, top_ten)
from crossplatform.candidate_frontend import sensor_frame_box

@torch.inference_mode()
def run():
    roster=read(PRIV/'panel.json'); fseal=read(OUT/'frozen_seal.json')
    assert fseal['features_sha']==digest(PRIV/'frozen_hashes.json') and fseal['panel']==digest(PRIV/'panel.json')
    files=read(PRIV/'frozen_hashes.json')
    tok=RobertaTokenizerFast.from_pretrained(ROOT/'crossplatform/upstream/data/roberta_base',local_files_only=True)
    evaler=GroundingEvaluator(prefixes=['last_'])
    stub={'gt_bboxes':torch.zeros((1,1,7),device='cuda'),'meta_path':['dummy/p/f/m'],'utterances':['']}
    torch.backends.cuda.matmul.allow_tf32=False
    pred={}
    with patch.object(grounding_evaluator,'iou3d_rotated_vs_aligned',skip_metric):
        for row in roster:
            uid=row['uid'];target=row['target']
            with open(row['calibration'],'rb') as f:
                cat=next(v for j,v in enumerate(ijson.items(f,'ground_info.item.class')) if j==row['slot']).lower()
            phrase=Joint3DDataset._format_caption(None,row['text'])
            syn=WAYMO_SYNONYMS if target=='waymo' else M3ED_SYNONYMS
            span=next(m.span() for w in syn.get(cat,[cat]) for m in re.finditer(r'\b'+re.escape(w)+r'\b',phrase,re.I))
            encoded=tok.batch_encode_plus([phrase],padding='longest',return_tensors='pt')
            mask=torch.as_tensor(get_positive_map(encoded,[torch.tensor(span)]),device='cuda')
            for src in PLATFORMS:
                if src==target:continue
                name=src+'_'+uid;p=PRIV/'frozen'/(name+'.npz')
                assert digest(p)==files[name]
                with np.load(p) as d:
                    raw=d['raw']
                    out=dict(last_proj_queries=torch.as_tensor(d['q'],device='cuda')[None],
                         proj_tokens=torch.as_tensor(d['token'],device='cuda')[None],
                         last_center=torch.as_tensor(raw[:,:3],device='cuda')[None],
                         last_pred_size=torch.as_tensor(raw[:,3:],device='cuda')[None].clamp(min=1e-6),
                         box_label_mask=torch.ones((1,1),device='cuda'),positive_map=mask[None])
                best=int(top_ten(out,mask)[0][0])
                evaler.evaluate_bbox_by_contrast(stub,out,'last_')
                box=np.r_[raw[best,:3],np.maximum(raw[best,3:],1e-6),0.]
                np.testing.assert_array_equal(np.array(evaler.prediction_records.pop()['pred_box']),box[:6])
                pred[name]=sensor_frame_box(box,target,row['suffix']).tolist()
    assert len(pred)==192
    write(PRIV/'official.json',pred)
    write(OUT/'official_seal.json',dict(pred=digest(PRIV/'official.json'),n=len(pred),
          feature_source=digest(OUT/'frozen_seal.json'),target_category_span=True,target_box_read=False,
          exact_released_evaluator_top1_agreement=len(pred),code=digest(__file__)))

if __name__=='__main__':run()
