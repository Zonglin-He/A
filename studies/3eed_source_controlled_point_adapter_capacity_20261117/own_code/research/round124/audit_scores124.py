from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round124';B=R/'crossplatform/round124';rd=lambda p:json.loads(Path(p).read_text())
def main():
 rr=rd(H/'scores_attempt001/ALL_ROWS_PRIVATE.json');reg=rd(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json');assert len(rr)==len(reg)==384;count=0
 for e,r in zip(reg,rr):
  y=np.load(R/e['files']['label']['path'])['ious'];a=np.load(B/'formal'/e['source']/'samples_PRIVATE'/(e['row']['id']+'.npz'));assert r['source']==e['source'] and r['id_PRIVATE']==e['row']['id']
  for k,q in zip(a['arm_names'],a['scores'].argmax(1)):assert r['arms'][k]['IoU']==float(y[q]);count+=1
 def mean(rows,k):v=[r['arms'][k]['IoU'] for r in rows];return [sum(v)/len(v)*100,sum(x>.25 for x in v)/len(v)*100,sum(x>.5 for x in v)/len(v)*100]
 for x in rd(H/'scores_attempt001/AGGREGATE_ROWS.json'):
  rows=rr if x['frame_group']=='all' else [r for r in rr if r['frame_group']==int(x['frame_group'][-1])]
  if x['source']=='all_three_equal':got=[sum(mean([r for r in rows if r['source']==s],x['arm'])[i] for s in ['waymo','drone','quad'])/3 for i in range(3)]
  else:rows=[r for r in rows if r['source']==x['source']];got=mean(rows,x['arm'])
  assert all(abs(got[i]-x[k])<1e-9 for i,k in enumerate(['mIoU_percent','Acc25_percent','Acc50_percent']))
 for x in rd(H/'scores_attempt001/ALL_PAIRED_ROWS.json'):
  rows=[r for r in rr if r['source']==x['source']];d=[r['arms'][x['method']]['IoU']-r['arms'][x['control']]['IoU'] for r in rows];assert abs(sum(d)/len(d)*100-x['gain_pp'])<1e-9 and sum(y>1e-12 for y in d)==x['better'] and sum(y< -1e-12 for y in d)==x['worse']
 for x in rd(H/'scores_attempt001/ALL_SCENE_ROWS_PRIVATE.json'):
  rows=[r for r in rr if r['source']==x['source'] and r['scene_PRIVATE']==x['scene_PRIVATE']];got=mean(rows,x['arm']);assert len(rows)==x['n'] and all(abs(got[i]-x[k])<1e-9 for i,k in enumerate(['mIoU_percent','Acc25_percent','Acc50_percent']))
 for x in rd(H/'CONCENTRATION.json'):
  d=[r['arms'][x['arm']]['IoU']-r['arms']['zero']['IoU'] for r in rr];assert abs(sum(d)/384*100-x['gain_pp'])<1e-9 and sum(v>1e-12 for v in d)==x['better'] and sum(v< -1e-12 for v in d)==x['worse']
 out={'state':'PASS_ALL_SOURCE_CACHED_IOU_VALUES_AND_EQUAL_SOURCE_REDUCTIONS','n':384,'arms':7,'values':count,'fresh_GT_geometry':False,'target_data_read':False,'FINAL_access':False};(H/'SCALAR_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
