"""Describe resolution and intervention effects after sealed predictions; no fitting."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
from common27 import *
def main():
 scored=read(EXP/'ALL528_ROWS_PRIVATE.json');resolution={x['alias']:x for x in read(EXP/'RESOLUTION_ROWS_PRIVATE.json')};modelrows=[];geometry_replay=0;counts=collections.Counter();oldgroups=collections.defaultdict(list)
 for x,r in zip(scored,rows()):
  assert x['alias']==r['alias'];pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));a=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));e=read(BASE/'joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'));pool=prep['mask_pool_PRIVATE']
  for m in pool:
   ve=e['targets_PRIVATE'][str(m)];base=ve['scene_base'];high=ve['scene_hires'];repeat=ve['scene_repeat'];joint=ve['crop_scene_joint'];assert base['image_sha256'][0]==high['image_sha256'][0]==repeat['image_sha256'][0]==repeat['image_sha256'][1]==joint['image_sha256'][1];assert base['prompt_sha256']==high['prompt_sha256'] and repeat['prompt_sha256']==joint['prompt_sha256'];assert len({v['literal_caption_sha256'] for k,v in ve.items()})==1;assert base['image_grid_thw'][0]==repeat['image_grid_thw'][0]==repeat['image_grid_thw'][1]==joint['image_grid_thw'][1];counts['actual_candidate_image_caption_prompt_binding_checks']+=1
  q=x['q']['legacy_complex'];j=int(np.argmax(a['weights'][q])) if pool and a['weights'][q].sum()>0 else None
  if r['alias'] in resolution:
   for item in resolution[r['alias']]['instance_sizes']:
    m=pool[item['pool_slot']];v=e['targets_PRIVATE'][str(m)]
    for n,key,vi in [('scene_base','scene_resized_wh',0),('scene_hires','hires_resized_wh',0),('crop_scene_joint','crop_resized_wh',0)]:assert v[n]['view_geometry'][vi]['resized_size_wh']==item[key]
    geometry_replay+=1
  counts['observations']+=1;counts['with_pool']+=bool(pool)
  if j is not None:
   v=e['targets_PRIVATE'][str(pool[j])];rec={'alias':r['alias'],'platform':r['platform'],'stage':r['stage'],'R26_vs_legacy_delta_pp':100*(x['ious']['R26_scene_fixed2']-x['ious']['legacy_complex']),'joint_vs_scene_delta_pp':100*(x['ious']['crop_scene_joint']-x['ious']['scene_base']),'joint_vs_legacy_delta_pp':100*(x['ious']['crop_scene_joint']-x['ious']['legacy_complex']),'base_subject_short_px':v['scene_base']['view_geometry'][0]['subject_bbox_short_side_px'],'hires_subject_short_px':v['scene_hires']['view_geometry'][0]['subject_bbox_short_side_px'],'joint_crop_subject_short_px':v['crop_scene_joint']['view_geometry'][0]['subject_bbox_short_side_px'],'joint_context_subject_short_px':v['crop_scene_joint']['view_geometry'][1]['subject_bbox_short_side_px'],'base_visual_tokens':sum(z['visual_tokens'] for z in v['scene_base']['view_geometry']),'hires_visual_tokens':sum(z['visual_tokens'] for z in v['scene_hires']['view_geometry']),'repeat_visual_tokens':sum(z['visual_tokens'] for z in v['scene_repeat']['view_geometry']),'joint_visual_tokens':sum(z['visual_tokens'] for z in v['crop_scene_joint']['view_geometry'])};modelrows.append(rec)
   if r['alias'] in resolution:
    group='R26_harm' if rec['R26_vs_legacy_delta_pp']< -1e-8 else 'R26_gain' if rec['R26_vs_legacy_delta_pp']>1e-8 else 'R26_tie';oldgroups[r['platform']+'_'+group].append(rec)
 summary={}
 for name,rs in oldgroups.items():
  summary[name]={'n':len(rs),'base_subject_short_median_px':float(np.median([r['base_subject_short_px'] for r in rs])),'joint_crop_short_median_px':float(np.median([r['joint_crop_subject_short_px'] for r in rs])),'joint_vs_scene_mIoU_delta_pp':float(np.mean([r['joint_vs_scene_delta_pp'] for r in rs]))}
 counts['resolution_grid_vs_actual_processor_exact_checks']=geometry_replay
 dump(EXP/'RESOLUTION_EFFECT_ROWS.json',modelrows);dump(EXP/'RESOLUTION_EFFECT_SUMMARY.json',{'all_summary_groups_descriptive_not_causal':True,'old400_groups':summary,'counts':dict(counts),'upsampling':'more model spatial tokens do not add raw sensor information','control_budget':'repeat uses two identical fullscenes with same perimage cap,not necessarily equal actual tokens to crop_scene;actual counts retained'})
 pooln={r['alias']:read(paths(r)['prep']/'cases_PRIVATE'/(r['id']+'.json'))['objects'] for r in rows()};op={}
 for phase,rr in [('OLD400',scored[:400]),('Q64',scored[464:])]:
  for platform in ['drone','quad']:
   for kind in ['all','multiple_instances']:
    rs=[x for x in rr if x['platform']==platform and (kind=='all' or pooln[x['alias']]>=2)]
    op[phase+'_'+platform+'_'+kind]={'n':len(rs),'joint_mIoU_percent':float(np.mean([x['ious']['crop_scene_joint'] for x in rs])*100),'joint_vs_legacy_pp':float(np.mean([x['ious']['crop_scene_joint']-x['ious']['legacy_complex'] for x in rs])*100),'joint_vs_scene_pp':float(np.mean([x['ious']['crop_scene_joint']-x['ious']['scene_base'] for x in rs])*100)} if rs else {'n':0}
 dump(EXP/'OPPORTUNITY_TASK_SUMMARY.json',{'subgroups_descriptive_only_no_runtime_exclusions':True,'groups':op})
 print('DIAGNOSTICS27',geometry_replay,summary)
if __name__=='__main__':main()
