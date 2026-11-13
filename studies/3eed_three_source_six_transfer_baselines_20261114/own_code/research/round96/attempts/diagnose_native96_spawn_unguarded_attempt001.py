"""No-GT numerical diagnosis of native replay mismatch; not a successful audit."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import configure,model,dataset,loader,to_cuda,torch
from source_validation import LEXICON
import re
rd=lambda p:json.loads(p.read_text());p=argparse.ArgumentParser();p.add_argument('--source',default='waymo');p.add_argument('--target',default='drone');a=p.parse_args();configure({'seed':0,'torch_num_threads':4});m=model().cuda().requires_grad_(False).eval();ck=R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth' if a.source=='waymo' else Path(rd(B/(a.source+'_source100')/'TRAINING_COMPLETE.json')['primary_checkpoint']);c=torch.load(ck,map_location='cpu',weights_only=True);m.load_state_dict(c['model']);del c;ds=dataset(a.target,'val');g=torch.Generator().manual_seed(0);dl=loader(ds,'val',g,{'num_workers':4});out=[]
with torch.no_grad():
 for ib,b in enumerate(dl):
  if ib not in [0,len(dl)//2,len(dl)-1]:continue
  bb=to_cuda(b);o=m({'point_clouds':bb['point_clouds'].float(),'text':bb['utterances']});T=o['proj_tokens'].shape[1]
  for ih,h in enumerate(['last_','proposal_']):
   raw=torch.cat([o[h+'center'],o[h+'pred_size']],-1).cpu().numpy();v=o[h+'proj_queries']@o['proj_tokens'].transpose(-1,-2)/.07
   for j in range(len(raw)):
    z=np.load(B/'frozen_matrix'/(a.source+'_to_'+a.target)/(str(ib*24+j).zfill(6)+'.npz'));old=z['boxes_raw'][ih];err=np.abs(raw[j]-old);maxer=float(err.max());row={'source':a.source,'target':a.target,'batch':ib,'row_in_batch':j,'head':h,'max_box_error':maxer}
    if maxer>1e-5:
     cost=((raw[j][:,None]-old[None])**2).sum(-1);ii,jj=linear_sum_assignment(cost);row.update(set_match_max_error=float(np.abs(raw[j][ii]-old[jj]).max()),moved_queries=int((ii!=jj).sum()),selected_saved_box_error=float(err[int(z['text_logodds'][ih].argmax())].max()),saved_query=int(z['text_logodds'][ih].argmax()))
    out.append(row)
  print('DIAG96',a.source,a.target,ib,max(r['max_box_error'] for r in out),flush=True)
(H/('NATIVE_DIAGNOSTIC_'+a.source+'_'+a.target+'_PRIVATE.json')).write_text(json.dumps(out,indent=2)+'\n');print({'state':'DIAGNOSTIC_ONLY_NOT_AUDIT_PASS','rows':len(out),'mismatch_rows':sum(r['max_box_error']>1e-5 for r in out),'worst':sorted(out,key=lambda x:-x['max_box_error'])[:2]})
