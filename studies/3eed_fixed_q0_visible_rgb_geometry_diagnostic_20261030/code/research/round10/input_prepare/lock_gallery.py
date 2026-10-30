"""Input-only R10 gallery registration; never reads media, caches or labels."""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=D/'diagnostic160_gpu_inputs.json';rows=json.loads(manifest.read_text());assert len(rows)==160
salt='R10_rgb_support_gallery_v1';groups=collections.defaultdict(list)
for r in rows:groups[(r['platform'],r['scene'])].append(r)
assert len(groups)==5
selected=[]
for (platform,scene),items in sorted(groups.items()):
 ranked=sorted(items,key=lambda r:(hashlib.sha256((salt+'|'+r['id']).encode()).hexdigest(),r['id']))
 for r in ranked[:2]:selected.append({'id':r['id'],'platform':platform,'scene':scene,'case_alias':f'case_{len(selected)+1:02d}'})
assert len(selected)==10 and len({r['id'] for r in selected})==10 and len({r['case_alias'] for r in selected})==10
plan={'state':'R10_input_only_gallery10_frozen_before_task_GT_score','input_manifest_sha256':sha(manifest),'selection_program_sha256':sha(__file__),'selection_salt':salt,'rule':'Each platform-scene choose first2 ascending SHA256(salt|id), tie id; no score/color/GT screening.','n':10,'per_scene':2,'rows_PRIVATE':selected,'primary_panels':['original_RGB','q0_seeded_RGB_mask','source_q0_projection','full_rgb_projection','no_rgb_coordinate_projection','rgb_gated_pointmask_ablation_projection'],'GT_overlay_default':False,'GT_overlay_requires_separate_postseal_CPU_score_and_render_release':True,'caption_and_actualmedia_private_only':True,'use_for_method_change':False,'scope':'Qualitative diagnostics; not random independent scene evidence or performance-selected examples.'}
p=D/'GALLERY_PLAN_LOCK_v2.json';assert not p.exists();p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'n':10,'gallery_plan_sha256':sha(p),'file':str(p)}))
