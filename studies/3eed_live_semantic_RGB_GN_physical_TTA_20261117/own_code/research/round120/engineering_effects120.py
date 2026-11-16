from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round120';B=R/'crossplatform/round120/engineering';rd=lambda p:json.loads(Path(p).read_text());out=[]
assert rd(H/'ENGINEERING.json')['n']==24
for st in ['zero','depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']:
 a=[x for p in B.glob('*/*/records_PRIVATE/*.json') for x in rd(p)['evaluation']['engineering'] if x['setting']==st];assert len(a)==24;v=np.array([x['source_context_residual_change_L2'] for x in a]);out.append({'setting':st,'n':24,'source_context_residual_change_L2_mean':float(v.mean()),'median':float(np.median(v)),'minimum':float(v.min()),'maximum':float(v.max()),'nonzero':int((v>0).sum()),'foundation_h_u_bitexact':all(x['native_foundation_h_u_bitexact'] for x in a)})
(H/'ENGINEERING_PATH_SUMMARY.json').write_text(json.dumps({'GT_used':False,'task_success_claim':False,'states':out},indent=2)+'\n');print(out,flush=True)
