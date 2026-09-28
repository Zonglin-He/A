"""All native blocks, including failures; independent of coordinate-restricted logits."""
import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha
from scripts.desta3d_v3_context_mask import ARMS

def run(name):
 d=OUT/name;assert read(d/'independent_readback_v1/PRE_SCORE_AUDIT.json')['status']=='passed'
 p=ROOT/'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/tokenizer.json';tok=read(p)
 vocab=dict(tok['model']['vocab']);vocab.update({x['content']:x['id'] for x in tok['added_tokens']})
 reverse={v:k for k,v in vocab.items()};start=vocab['<|box_start|>'];end=vocab['<|box_end|>']
 coord={vocab[f'<{i}>'] for i in range(1001)}
 assert len(coord)==1001
 rows=[]
 for i in range(16):
  for arm in ARMS:
   pred=torch.load(d/'episodes'/f'{i:02}'/(arm+'.pt'),weights_only=False,map_location='cpu');raw=pred['readout'].get('raw_blocks');bad=[]
   if raw is not None:
    for k,block in enumerate(raw.tolist()):
     issues=[]
     for j,t in enumerate(block):
      okay=t==start if j==0 else t==end if j==5 else t in coord
      if not okay:issues.append(dict(slot=j,id=t,token=reverse.get(t)))
     if len(block)!=6:issues.append(dict(length=len(block)))
     if issues:bad.append(dict(block=k,nonconforming=issues,token_ids=block))
   if pred['format_ok']:assert raw is not None and not bad
   rows.append(dict(panel_index=i,arm=arm,format_ok=bool(pred['format_ok']),raw_blocks=len(raw) if raw is not None else 0,bad_blocks=bad))
 result=dict(status='passed',predictions=80,rows=rows,bad_outputs=[r for r in rows if not r['format_ok']],tokenizer_sha=sha(p),
  official_six_token_validation_sha=sha(ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py'),
  interpretation='No token repair/retry/filter; unchanged official whole-output scoring. Spatial event equality audited separately.',GPU=False,target=False)
 write(d/'ROOT_NATIVE_GRAMMAR_READBACK.json',result);print({'bad_outputs':result['bad_outputs']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
