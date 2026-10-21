"""Read-only source-specific feedback audit after v0 final prediction seal."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stvg.runtime import sha
from stvg.metrics import box_iou
from stvg.metrics_base import xyxy
from tools.tubedetr_propagation_policy import inputs,caches,labels,OUT,PUBLIC


def main():
    p,old,rows=inputs();fs=caches();final=PUBLIC/'final_seal.json';s=json.loads(final.read_text());assert s['status']=='complete'
    for f,h in s['files'].items():assert sha(ROOT/f)==h
    truth=labels(old);details=[]
    for q in p['query_ids']:
        c=torch.load(OUT/fs['files'][q]['path'],map_location='cpu',weights_only=True);gt={int(k):v for k,v in truth[q]['boxes'].items()};pos=set(c['positions'].tolist());chosen=[];regret=[];sourcepairs=[]
        for t,f in enumerate(c['frame_ids'].tolist()):
            if f not in gt or t in pos:continue
            b=c['boxes'][t].numpy();valid=c['valid'][t].numpy();iou=box_iou(xyxy(b,rows[q]['metadata']['width'],rows[q]['metadata']['height']),np.array(gt[f]));feedback=c['feedback_source'][t].numpy();j=int(feedback.argmax());regret.append(float(iou[valid].max()-iou[j]))
            if j!=0:chosen.append(float(iou[j]-iou[0]))
            for k in range(1,3):
                if not valid[k]or abs(iou[k]-iou[0])<1e-6:continue
                sourcepairs.append(float((feedback[k]-feedback[0])*(iou[k]-iou[0])>0))
        details.append(dict(id=q,changed_top1_count=len(chosen),top1_alternative_wins=sum(v>0 for v in chosen),top1_alternative_losses=sum(v<0 for v in chosen),top1_alt_mean_delta=float(np.mean(chosen))if chosen else None,top1_regret=float(np.mean(regret))if regret else None,Source_pair_accuracy=float(np.mean(sourcepairs))if sourcepairs else None))
    def mean(v):
        v=[a for a in v if a is not None];return float(np.mean(v))if v else None
    result=json.loads((PUBLIC/'analysis.json').read_text());result['feedback_source_specific']=dict(unobserved_top1_alternative_decisions=sum(d['changed_top1_count']for d in details),unobserved_top1_alternative_wins=sum(d['top1_alternative_wins']for d in details),unobserved_top1_alternative_losses=sum(d['top1_alternative_losses']for d in details),equal_query_alt_delta_pp=100*mean([d['top1_alt_mean_delta']for d in details]),equal_query_top1_regret_pp=100*mean([d['top1_regret']for d in details]),equal_query_Source_pair_accuracy=mean([d['Source_pair_accuracy']for d in details]),no_GT_method_gates=True)
    (OUT/'source_feedback_private.json').write_text(json.dumps(details,indent=2)+'\n');(PUBLIC/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['feedback_source_specific'],indent=2))


if __name__=='__main__':main()
