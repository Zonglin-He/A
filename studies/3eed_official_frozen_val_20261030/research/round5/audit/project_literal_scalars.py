"""Explicit public scalar projection of sealed literal/audited results; no GT scoring.
No parent IDs/scenes, candidate vectors, pixel/box values, captions or events copied.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main(a):
 assert not a.output.exists();seal=read(a.literal/'SEALED.json')
 for n,h in seal['files'].items():assert sha(a.literal/n)==h
 d=read(a.literal/'RESULTS_PRIVATE.json');prep=read(a.preparation);rows=d['rows'];table=[]
 for platform in ['waymo','drone','quad']:
  for condition in ['full','noun']:
   for head in ['last_','proposal_']:
    for stream in ['official_bbf','official_bbs','firstnoun_bbf','firstnoun_bbs']:
     rr=[x for x in rows if (x['platform'],x['condition'],x['head'],x['stream'])==(platform,condition,head,stream)]
     if not rr:continue
     assert all(x['n']==10 for x in rr);item={'platform':platform,'condition':condition,'head':head,'stream':stream,'n':len(rr),'scene_clusters':len({x['scene'] for x in rr}),'GTclass_assisted_readout':stream.startswith('official'),'source_literal_top1_nonfinite_count':sum(x['literal_top1_iou'] is None for x in rr),'audited_float64_mean_top1_iou':float(np.mean([x['audited_top1_iou'] for x in rr])),'literal_mean_top1_iou':float(np.mean([x['literal_top1_iou'] for x in rr])) if all(x['literal_top1_iou'] is not None for x in rr) else None,'topk':{}}
     for k,key in [(1,'literal_top1_iou'),(5,'literal_best_top5_iou'),(10,'literal_best_top10_iou')]:
      values=[x[key] for x in rr];finite=all(x is not None for x in values);entry={'nonfinite_parent_count':sum(v is None for v in values),'acc25_success_count':sum(v>.25 for v in values) if finite else None,'acc50_success_count':sum(v>.5 for v in values) if finite else None,'denominator':len(rr),'mean_best_iou_GTassisted_if_k_gt1':float(np.mean(values)) if finite else None};item['topk'][str(k)]=entry
     table.append(item)
 legacy=[]
 for platform in ['waymo','drone','quad']:
  rr=[x for x in rows if x['platform']==platform and x['condition']=='full' and x['stream']=='official_bbf'];assert len(rr)%7==0;v=[x['literal_top1_iou'] for x in rr];legacy.append({'platform':platform,'eligible_target_n':len(rr)//7,'object_head_observations':len(rr),'literal7head_mean_iou':float(np.mean(v)) if all(x is not None for x in v) else None,'audited7head_mean_iou':float(np.mean([x['audited_top1_iou'] for x in rr])),'nonfinite_observation_count':sum(x is None for x in v),'heads_are_not_independent_samples':True})
 coverage=[{'platform':p,'official_split_listed_scenes':v['official_split_listed_scenes'],'missing_listed_scene_directory_count':len(v['missing_listed_scene_dirs']),'on_disk_raw_target_count':v['raw_ground_info_objects'],'forwarded_raw_target_count':v['ready_model_inputs']} for p,v in prep['platforms'].items()]
 out={'scope':'Descriptive availableofficialVAL, releasedunioncheckpoint, B1fixedsampling, noTTA/training; GTclassassistedspan distinctfromtextallraw','prediction_before_label_scope':'Allfull-run predictioncaches sealed before this fullscoringtransaction, not firstever historicalGTaccess;48smoke andpriorstudies exposed','numerics':'Literaloriginalfloat32IoU andindependentauditedfloat64 keptseparate; neither silently substitutes forother','source_result_sha256':sha(a.literal/'RESULTS_PRIVATE.json'),'source_seal_sha256':sha(a.literal/'SEALED.json'),'n_raw_targets':d['n_parents'],'per_platform_execution_coverage':coverage,'scalar_tables':table,'legacy7head_BBF_miou':legacy,'nonfinite_policy':'Neverdrop denominator ornanmean. If requested literalvector contains nonfinite, scalar best/accuracy projection isnull and flagged; exactnative threshold counts would require retainedperpositionbooleans.','extra_field_schema':'OfficialGTfirst7 only; remainingrawextra fields are unnamed andunused','whitelist':'Only scope/hash/count/bool/aggregateIoU fields; noparentIDs/sceneidentities/rawannotations/captions/media/arrays/events'};a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps([x for x in table if x['condition']=='full' and x['head']=='last_' and x['stream'].startswith('official')],indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--literal',type=Path,required=True);ap.add_argument('--preparation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
