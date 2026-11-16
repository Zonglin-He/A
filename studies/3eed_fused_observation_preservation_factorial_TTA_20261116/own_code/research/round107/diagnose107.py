"""Postseal descriptive diagnosis; no new scores, filters or candidate method."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round107';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 rows=[r for p in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];out=[];cases=[]
 for name,rr in [('all',rows)]+[(d,[r for r in rows if r['source']+'_to_'+r['target']==d]) for d in rd(R/'crossplatform/round107/MATRIX_SEALED.json')['directions']]:
  ar=lambda k:np.array([r['arms'][k]['IoU'] for r in rr]);q=lambda k:np.array([r['arms'][k]['query_PRIVATE'] for r in rr]);p='real_mean_keep_fusedABC';f='real_frozen_fusedABC';o='real_TTA_fusedABC';a='prepared_frozen';d=ar(p)-ar(f)
  out.append({'scope':name,'n':len(rr),'primary_better_than_fused':int((d>1e-12).sum()),'primary_worse_than_fused':int((d< -1e-12).sum()),'primary_improvements_exceed_prepared':int(((d>1e-12)&(ar(p)>ar(a)+1e-12)).sum()),'primary_improvements_only_repair_fusion':int(((d>1e-12)&(ar(p)<=ar(a)+1e-12)).sum()),'primary_harms_erase_fusion_improvement':int(((d< -1e-12)&(ar(f)>ar(a)+1e-12)).sum()),'old_TTA_harms_reverted_to_fusion_query':int(((ar(o)<ar(f)-1e-12)&(q(p)==q(f))).sum()),'old_TTA_harms_still_present':int(((ar(o)<ar(f)-1e-12)&(ar(p)<ar(f)-1e-12)).sum()),'old_TTA_benefits_preserved':int(((ar(o)>ar(f)+1e-12)&(ar(p)>ar(f)+1e-12)).sum())})
  if name=='all':continue
  for sign in [-1,1]:
   ss=sorted([r for r in rr if sign*(r['arms'][p]['IoU']-r['arms'][f]['IoU'])>1e-12],key=lambda r:abs(r['arms'][p]['IoU']-r['arms'][f]['IoU']),reverse=True)
   for j in sorted({0,min(1,len(ss)-1),len(ss)//2}) if ss else []:
    r=ss[j];cases.append({'direction':name,'row_PRIVATE':r['id_PRIVATE'],'kind':'good' if sign>0 else 'failure','all_arms':r['arms'],'fits':r['fits'],'status':r['status'],'identity':'UNKNOWN unless previously independently viewed; IoU is not identity label','selection':'postseal extremes+median, no case filter in predictions'})
 dump(H/'BENEFIT_PROVENANCE.json',out);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);p=H/'REPORT.md';p.write_text(p.read_text()+'\n## Good / failure 的总体来源\n\n'+json.dumps(out,indent=2)+'\n\n案例索引从每方向正负尾部与中位数事后生成；没有新看图，不将IoU替代实例身份。R106已实际查看的4页仍只提供历史机制线索，本轮不声称新增视觉身份复核。\n');print('DIAGNOSE107',out[0],len(cases))
if __name__=='__main__':main()
