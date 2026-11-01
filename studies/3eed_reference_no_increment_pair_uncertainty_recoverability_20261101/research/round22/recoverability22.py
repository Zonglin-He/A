"""Offline eligibility only. GT never selects runtime candidates or filters G."""
import sys,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round22'),str(ROOT/'research/round5/audit')]
from common22 import *
from independent_core_v2 import transform_gt,candidate_ious
def main():
    index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};out=[];count={}
    for r in read(ROWS):
        z=feature(r);ix=index[r['id']];assert sha(ix['meta_path'])==ix['metadata_container_sha256'];gt=transform_gt(read(ix['meta_path'])['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);iou=candidate_ious(b,gt)[0];p=old(r,'pairs');offered=[int(z['scores'].argmax())]+[x['qB_PRIVATE'] for x in p['pairs']];g=prior(r,'text');offered_recoverable=float(iou[offered].max())>.25 and iou[offered[0]]<=.25;has_rel=bool(g['reference_relations'])
        out.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'source_iou':float(iou[offered[0]]),'all256_oracle_iou':float(iou.max()),'offered_oracle_iou':float(iou[offered].max()),'offered_recoverable25':bool(offered_recoverable),'all256_recoverable25':bool(iou.max()>.25 and iou[offered[0]]<=.25),'has_reference_relation':has_rel,'paired_queries_PRIVATE':offered,'candidate_ious_PRIVATE':iou.tolist()})
    for p in ['drone','quad']:
        rr=[r for r in out if r['platform']==p];count[p]={'n':len(rr),'all256_recoverable25':sum(r['all256_recoverable25'] for r in rr),'offered_recoverable25':sum(r['offered_recoverable25'] for r in rr),'offered_recoverable25_with_reference_relation':sum(r['offered_recoverable25'] and r['has_reference_relation'] for r in rr),'offered_recoverable_aliases':[r['alias'] for r in rr if r['offered_recoverable25']]}
    dump(EXP/'OFFLINE_RECOVERABILITY_PRIVATE.json',out);dump(EXP/'OFFLINE_RECOVERABILITY_COUNTS.json',{'counts':count,'label_selected_subgroups_diagnostic_only':True,'not_runtime_eligibility':True,'all208_historically_exposed':True});print(count)
if __name__=='__main__':main()
