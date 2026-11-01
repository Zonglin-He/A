"""Descriptive platform/scene aggregates from independently reconstructed rows."""
import json,collections,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
rr=json.loads((P/'ALL3800_RECONSTRUCTED_ROWS_PRIVATE.json').read_text());rows=collections.defaultdict(list)
for r in rr:rows[r['feedback_alias']].append(r)
scenes={p:sorted({r['target_scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};scene_rows=[];global_rows=[];zero=[]
for alias,rs in rows.items():
 if all(abs(r['delta_remove_pp'])<=1e-8 for r in rs):zero.append({'feedback_alias':alias,'query_changes':sum(r['query_changed'] for r in rs),'all200_task_deltas_zero':True})
 for split in ['A208','DEV64','R64','S64','R64_S64']:
  chosen=[r for r in rs if not r['is_removed_observation'] and (r['split'] in ['R64','S64'] if split=='R64_S64' else r['split']==split)];d=float(np.mean([r['delta_remove_pp'] for r in chosen]));global_rows.append({'feedback_alias':alias,'affected_platform':rs[0]['platform'],'split':split,'affected_platform_n':len(chosen),'same_platform_mean_delta_pp':d,'balanced_two_platform_delta_pp':d/2,'opposite_platform_unchanged_by_construction':True})
 for split in ['A208','DEV64','R64','S64']:
  chosen=[r for r in rs if r['split']==split and not r['is_removed_observation']]
  for scene in sorted({r['target_scene_PRIVATE'] for r in chosen}):
   ss=[r for r in chosen if r['target_scene_PRIVATE']==scene];p=rs[0]['platform'];delta=np.array([r['delta_remove_pp'] for r in ss]);scene_rows.append({'feedback_alias':alias,'platform':p,'split':split,'scene_alias':p+'_s'+str(scenes[p].index(scene)),'n':len(ss),'mean_delta_remove_pp':float(delta.mean()),'improves_if_removed':int((delta>1e-8).sum()),'worsens_if_removed':int((delta< -1e-8).sum()),'query_changes':sum(r['query_changed'] for r in ss)})
(P/'SCENE_INFLUENCE_INDEPENDENT_SUMMARY.json').write_text(json.dumps(scene_rows,indent=2)+'\n');(P/'BALANCED_SYSTEM_EFFECTS.json').write_text(json.dumps(global_rows,indent=2)+'\n');(P/'ZERO_TASK_EFFECT_DELETIONS.json').write_text(json.dumps({'n':len(zero),'models':zero,'interpretation':'zero IoU influence on this exposed cohort does not imply zero parameter or query change'},indent=2)+'\n')
print('DESCRIPTIVE_SUMMARY',len(scene_rows),len(global_rows),len(zero))
