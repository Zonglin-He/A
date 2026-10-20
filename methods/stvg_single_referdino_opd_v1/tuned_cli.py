"""Run the selected one-expert configuration, with no GT/scorer dependency."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .runtime import setup,lease
from .tuned import TunedSingleReferDINOOPDPredictor

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();req=json.loads(Path(args.request).read_text());assert set(req)<= {'dataset','key','metadata'};ds=req['dataset'];assert ds in ['hc2','vidstg'];setup();lock=lease();beg=time.perf_counter()
 try:
  if ds=='hc2':from vg_tta.tastvg_paper48_hc2_decode_v1 import decode
  else:from vg_tta.exact_frame_decode_audit_v2 import decode
  from methods.decota_final_simplified_v1.backbone import validate_input
  frames,ids,_=decode(req['metadata']);validate_input(frames,ids,req['metadata']);predictor=TunedSingleReferDINOOPDPredictor(ds);out=predictor.predict(frames,ids,req['metadata'],req.get('key','single-query'));dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);np.savez(dest,Source=out['Source'],E_Ref=out['E_Ref'],Live_raw=out['Live_raw'],Live=out['Live'],frame_ids=np.array(ids),Native_interval=out['intervals']['Native'],Strong_interval=out['intervals']['Strong']);dest.with_suffix('.cost.json').write_text(json.dumps(dict(expert_count=1,configuration=predictor.config,independent_GroundingDINO_calls=0,total_wall_seconds=time.perf_counter()-beg,capture=out['capture_cost'],adapt=out['cost']),indent=2)+'\n')
 finally:lock.close()
if __name__=='__main__':main()
