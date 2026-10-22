"""Postseal target coverage and native safety-corner audit, no online GT gate."""
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_support_view_v1.run import D,P,read,put,sha,records,B
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.metrics import box_iou
from stvg.metrics_base import xyxy
import torch,numpy as np

def main():
 torch.set_num_threads(1);s=read(D/'sealed.json');p=read(D/'plan.json');r=read(P/'report.json');assert read(D/'score_receipt.json')['report_sha256']==sha(P/'report.json');labels=R/'data/evaluation/development_labels.jsonl';assert sha(labels)==p['labels_sha256'];gt={x['id']:x for x in map(json.loads,labels.read_text().splitlines())if x['dataset']=='hc2'};vals={x['query']:x for x in r['rows']};rows=[]
 for row in records():
  k=row['id'];e=s['files'][k];assert sha(D/e['trace_path'])==e['trace_sha256'];tr=torch.load(D/e['trace_path'],weights_only=False,map_location='cpu');f=B/'captures'/f'{k}.pt';assert sha(f)==p['captures'][k]['capture_sha256'];cap=torch.load(f,weights_only=False,map_location='cpu');geom=SpatialGeometry.__new__(SpatialGeometry);geom.__dict__.update(cap['geometry']);ids=cap['frame_ids'];aux=evidence_dp(geom.for_interval([ids[0],ids[-1]+1]))['final'].numpy();m=row['metadata'];pix=xyxy(aux,m['width'],m['height']);expanded=aux.copy();expanded[:,2:]*=1.5;roi=xyxy(expanded,m['width'],m['height']);truth={int(f):np.asarray(b)for f,b in gt[k]['boxes'].items()};iou=[];cover=[]
  for i,f in enumerate(ids):
   if f not in truth:continue
   g=truth[f];x=roi[i];inter=np.maximum(0,np.minimum(x[2:],g[2:])-np.maximum(x[:2],g[:2])).prod();area=np.maximum(0,g[2:]-g[:2]).prod();iou.append(box_iou(pix[i],g));cover.append(float(inter/area)if area else 0.)
  st=tr['support_statistics'];actual=tr['actual_native_validcounts'];postdiff=max(abs(a-b)for xa,xb in zip(actual['SupportBoth'],actual['ReflectedBoth'])for a,b in zip(xa,xb));v=vals[k]['values'];rows.append(dict(query=k,auxiliary_target_sIoU=float(np.mean(iou)),expanded_context_GT_target_coverage=float(np.mean(cover)),mean_keep_cells=float(np.mean([x['keep_cells']for x in st])),mean_initial_overlap=float(np.mean([x['overlap_fraction']for x in st])),emptyROI_fallbacks=sum(x['empty_ROI_fallback']for x in st),target_corner_unmask=sum(not x['target_corner_kept']for x in st),control_corner_unmask=sum(not x['control_corner_kept']for x in st),max_actual_target_control_keycount_difference=postdiff,delta_t_primary_G6_pp=100*(v['SupportViewFusion']['t']-v['NativeG6']['t']),delta_t_fill_G6_pp=100*(v['FillOnlyFusion']['t']-v['NativeG6']['t'])))
 out=dict(scope='Postseal offline target coverage; no relation-object GT and no inference gate. High targetcoverage cannot prove all described relations visible.',queries=64,mean_aux_sIoU=float(np.mean([x['auxiliary_target_sIoU']for x in rows])),mean_target_context_coverage=float(np.mean([x['expanded_context_GT_target_coverage']for x in rows])),mean_keep_cells=float(np.mean([x['mean_keep_cells']for x in rows])),mean_initial_target_control_overlap=float(np.mean([x['mean_initial_overlap']for x in rows])),empty_ROI_fallbacks=sum(x['emptyROI_fallbacks']for x in rows),maximum_actual_keycount_difference=max(x['max_actual_target_control_keycount_difference']for x in rows),rows=rows);assert out['maximum_actual_keycount_difference']<=1;put(P/'support_audit.json',out);put(D/'support_audit_receipt.json',dict(code_sha256=sha(Path(__file__)),seal_sha256=sha(D/'sealed.json'),labels_sha256=sha(labels),report_sha256=sha(P/'report.json'),audit_sha256=sha(P/'support_audit.json'),online_GT=False));print(json.dumps({k:v for k,v in out.items()if k!='rows'},indent=2));
 for x in rows:
  if x['query']in ['RC042','RC041','RC015','RC011','RC020','RC060','RC064','RC023']:print(json.dumps(x))
if __name__=='__main__':main()
