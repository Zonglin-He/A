"""Postseal scalar-only gallery selection. No media/GT/scores/models/GPU.
Selection is reliability-extreme/undefined diagnostic, not benefit selection.
"""
from pathlib import Path
import argparse,collections,hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
SALT='R11_reliability_gallery_v1'
def main(a):
 assert sha(a.predictions/'SEALED.json')==a.prediction_seal_sha256;s=read(a.predictions/'SEALED.json');assert s['state']=='sealed_R11_reliability11_and_atomic6_before_GT' and s['n_samples']==320 and s['n_arms']==11 and s['all_reliability_and_rank_weights_sealed_before_GT']
 assert sha(a.manifest)==s['manifest_sha256'];rows=read(a.manifest);inputmap={r['id']:r for r in rows};assert len(inputmap)==320
 assert sha(a.predictions/'trace.json')==s['files']['trace.json'];traces=read(a.predictions/'trace.json');assert len(traces)==320 and {r['id'] for r in traces}==set(inputmap)
 candidates=[]
 for t in traces:
  sid=t['id'];row=inputmap[sid];p=a.predictions/(sid+'.npz');assert sha(p)==s['files'][p.name]
  with np.load(p,allow_pickle=False) as z:
   active=bool(z['atomic_active6'].any());defined=bool(z['case_defined']);r=float(z['case_reliability']);status=str(z['association_status'])
  assert active==t['base_actual_change'] and defined==(t['reliability']['case_reliability'] is not None)
  assert not defined or np.isfinite(r) and r==t['reliability']['case_reliability']
  if active:candidates.append({'id':sid,'platform':row['platform'],'scene':row['scene'],'case_defined':defined,'case_r':r if defined else None,'association_status':status})
 selected=[];counts={}
 tie=lambda x:hashlib.sha256((SALT+'|'+x['id']).encode()).hexdigest()
 for platform in ('drone','quad'):
  pp=[x for x in candidates if x['platform']==platform];defined=[x for x in pp if x['case_defined']];top=sorted(defined,key=lambda x:(-x['case_r'],tie(x),x['id']))[:2];taken={x['id'] for x in top};bottom=sorted([x for x in defined if x['id'] not in taken],key=lambda x:(x['case_r'],tie(x),x['id']))[:2];na=sorted([x for x in pp if not x['case_defined']],key=lambda x:(tie(x),x['id']))[:2]
  for role,subset in (('top_defined',top),('bottom_defined_after_top_unique',bottom),('undefined_C_hash',na)):
   for x in subset:selected.append({**x,'selection_role':role,'case_alias':f'reliability_case_{len(selected)+1:02d}'})
  counts[platform]={'C':len(pp),'defined_C':len(defined),'undefined_C':len(pp)-len(defined),'top_kept':len(top),'bottom_kept':len(bottom),'undefined_kept':len(na)}
 assert len({x['id'] for x in selected})==len(selected)<=12
 plan={'state':'R11_postseal_reliability_extremes_and_undefined_gallery_registered_no_task_scores','n':len(selected),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'source_seal_sha256':s['source_seal_sha256'],'manifest_sha256':sha(a.manifest),'selection_program_sha256':sha(__file__),'renderer_sha256':sha(Path(__file__).with_name('render_reliability_gallery.py')),'selection_salt':SALT,'selection_rule':'Per platform within originalG actualchanged C: top2 defined r, then bottom2 among remaining distinct defined cases, plus up to2 undefined C by fixedhash. Reliability ties use samehash thenID; missing/short buckets retain actual counts. Defined0 kept distinct fromNA.','selection_not_by_GT_task_IoU_gain_harm_or_pixels':True,'output_conditioned_not_performance_validation':True,'by_platform_counts':counts,'rows_PRIVATE':selected,'mask_reproduction_requirement':'Exact frozen physical_partitions + make_rgb_support + camera helper + OpenCV threads1; both A/B masks SHA must equal sealed trace before plotting.','same_RGB_sourceq0_ROI_backbone_not_independent_identity':True,'rawcaption_media_coords_private_only':True,'GT_overlay':False,'method_update_from_gallery':False}
 assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'n':len(selected),'plan_sha256':sha(a.output),'counts':counts}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('predictions','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--prediction-seal-sha256',required=True);main(p.parse_args())
