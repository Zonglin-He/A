"""Record actually viewed atlas pages and bounded postseal observations."""
from inventory105 import *
import numpy as np
def main():
 p=R/'crossplatform/round105/atlas_PRIVATE/INDEX_PRIVATE.json';z=rd(p)
 for x in z:
  if x['case']=='CASE0001':x['identity_review']='Viewed: both predictions project over the same visible white van; differences resemble overlapping box hypotheses, not a visible different-instance correction. GT completeness/3D center not visually certified.'
  if x['case']=='CASE0003':x['identity_review']='Viewed: overlapping hypotheses at the right-edge partially visible light vehicle; no visually established target-identity correction. Nearly unchanged camera context; exact instance/3D boundary remains uncertain.'
 dump(p,z);s={'atlas_pages_generated':len(z),'pages_actually_viewed':2,'same_information_TTA_vs_frozen_jointABC_good_IoU_cases':0,'all_Top1_changes_vs_prepared':len(z),'before_after_AABB_pair_IoU_min':float(min(x['before_after_box_IoU'] for x in z)),'before_after_AABB_pair_IoU_max':float(max(x['before_after_box_IoU'] for x in z)),'same_instance_not_certified_by_pair_IoU':True,'review':'Two viewed pages are consistent with overlapping candidate changes on the same visible object/area. Others remain visually UNKNOWN; no claim of complete identity audit.','actual_viewpoint_note':'Sensor estimates have sub-degree median rotation and small median translation; distinct image bytes do not ensure new discriminating target evidence. Not all variation is JPEG compression: resized RGB differences are nonzero, but target-specific information is unproven.'};dump(H/'ATLAS_REVIEW_SUMMARY.json',s)
 report=H/'REPORT.md';report.write_text(report.read_text()+'\n## 新观测与实际看图复核\n\n'+json.dumps(s,indent=2)+'\n\n'+json.dumps(rd(H/'OBSERVATION_DIAGNOSTICS.json'),indent=2)+'\n\n这一轮没有相对自身冻结模型的good IoU案例，不能将与较弱历史或打乱对照的差异包装成有效纠错。新增观测仅使5条Top1变化，其中4条IoU下降、1条仍为零；5对所选框的AABB重叠均高于0.52，说明不能直接把query变化解释成换了物体。只实际查看2页，其余保持未知。真实额外RGB接入工程可用，但这份原生读取没有产生总体有益的排序信息，参数适应也没有额外收益；不再对本配方增加更新强度。\n');print('REVIEW105_RECORDED',s['pages_actually_viewed'])
if __name__=='__main__':main()
