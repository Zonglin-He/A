"""Run one complete video/text request, without any independent detector."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .runtime import ROOT,setup,lease
from .predictor import SingleReferDINOOPDPredictor
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    request=json.loads(Path(args.request).read_text());assert set(request)<= {'dataset','key','metadata'}
    ds=request['dataset'];assert ds in ['hc2','vidstg'];metadata=request['metadata'];setup();lock=lease();start=time.perf_counter()
    try:
        if ds=='hc2':from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
        else:from vg_tta.exact_frame_decode_audit_v2 import decode
        frames,ids,_=decode(metadata)
        from methods.decota_final_simplified_v1.backbone import validate_input
        validate_input(frames,ids,metadata);predictor=SingleReferDINOOPDPredictor(ds);out=predictor.predict(frames,ids,metadata,request.get('key','single-query'))
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez(dest,Source=out['Source'],E_Ref=out['E_Ref'],Live=out['Live'],frame_ids=np.array(ids),Native_interval=out['intervals']['Native'],Strong_interval=out['intervals']['Strong'])
        dest.with_suffix('.cost.json').write_text(json.dumps(dict(expert_count=1,independent_GroundingDINO_calls=0,total_wall_seconds=time.perf_counter()-start,capture=out['capture_cost'],adapt=out['cost']),indent=2)+'\n')
    finally:lock.close()
if __name__=='__main__':main()
