"""Read every saved native block; separate restricted CE from full-vocab grammar."""
import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha,verify_seal

def run(name):
 d=OUT/name;verify_seal(d);report=read(d/'ROOT_DECISION.json');cases=[]
 for index,c in enumerate(report['cases']):
  ep=d/'episodes'/f'{index:02}';states=[]
  for step in range(31):
   p=torch.load(ep/f'NATIVE_{step:02}.pt',weights_only=False,map_location='cpu')
   tr=torch.load(ep/f'TRACE_{step:02}.pt',weights_only=False,map_location='cpu');ids=tr['token_ids'];coord=set(tr['coordinate_ids']);bad=[]
   raw=p['readout'].get('raw_blocks')
   if raw is not None:
    for i,block in enumerate(raw.tolist()):
     for j,tok in enumerate(block):
      good=tok==ids['box_start'] if j==0 else tok==ids['box_end'] if j==5 else tok in coord
      if not good:bad.append({'block':i,'slot':j,'token_id':tok,'kind':'coordinate' if 1<=j<=4 else 'structure'})
   states.append({'step':step,'format_ok':bool(p['format_ok']),'bad_tokens':bad,'interval':p['interval'],'CE':c['all_steps'][step]['CE']})
  cases.append({'key':c['key'],'branch':c['branch'],'states':states,'bad_states':sum(bool(s['bad_tokens']) for s in states),
                'fixed_final':states[-1]})
 write(d/'ROOT_GRAMMAR_READBACK.json',{'cases':cases,'source':'all native full-vocab sampled block IDs, not coordinate-restricted argmax',
  'code_fact':'Official sample_token_ids uses full-vocabulary argmax; old coordinate CE normalizes only1001 classes, excluding non-coordinate competitors',
  'not_claimed':'No saved full-vocab logit gaps for old run; no claim about every oracle case or universal loss failure',
  'pins':{str(ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py'):sha(ROOT/'external/ParallelTubeDecoding/src/model/ptd_generation.py')},'new_GPU':False})
 print([(c['branch'],c['bad_states'],c['fixed_final']) for c in cases])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
