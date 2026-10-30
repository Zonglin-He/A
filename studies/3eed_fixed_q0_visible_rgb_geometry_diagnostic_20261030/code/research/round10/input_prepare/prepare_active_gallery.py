"""Select every changed full-RGB output postseal; no score, GT or media access."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path(__file__).resolve().parents[3];E=B/'research/round10/experiment';D=E/'active_gallery12';P=B/'crossplatform/round10/geometry160_attempt001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=P/'SEALED.json';assert sha(seal)=='f12b0892f11f44a9f71865798e04502e8e08d1d27173e3ab081883b183d0f6da';s=json.loads(seal.read_text());assert s['n_samples']==160 and s['no_GT_or_parameter_update']
manifest=B/'research/round10/input_prepare/diagnostic160_gpu_inputs.json';assert sha(manifest)==s['manifest_sha256'];rows=json.loads(manifest.read_text());selected=[];flags={}
for r in rows:
 p=P/(r['id']+'.npz');assert sha(p)==s['files'][p.name]
 with np.load(p,allow_pickle=False) as z:
  assert tuple(z['arm_names'].tolist())==('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation');changed=bool(z['actual_changed_axes'][1].any())
 flags[r['id']]=changed
 if changed:selected.append(r)
assert len(selected)==12
selected.sort(key=lambda r:(r['platform'],r['scene'],hashlib.sha256(('R10_active_gallery_v1|'+r['id']).encode()).hexdigest(),r['id']))
plan={'state':'R10_postseal_all_changed_fullRGB12_mechanism_gallery_plan_no_GT_scores','n':12,'input_manifest_sha256':sha(manifest),'prediction_seal_sha256':sha(seal),'selection_program_sha256':sha(__file__),'renderer_sha256':sha(B/'research/round10/input_prepare/render_active_gallery.py'),'selection':'ALL sealed160 cases where actual_changed_axes[full_rgb].any(); no filtering of gain/harm, score, GT or image content. Order by platform,scene,SHA256(R10_active_gallery_v1|id).','selection_requires_all12_no_omissions':True,'rows_PRIVATE':[{'id':r['id'],'platform':r['platform'],'scene':r['scene'],'case_alias':f'active_case_{i+1:02d}'} for i,r in enumerate(selected)],'output_conditioned_postseal_not_performance_validation':True,'primary_original10_gallery_immutable':True,'GT_overlay_default':False,'caption_and_actualmedia_private_only':True,'use_for_method_change':False}
assert not D.exists();D.mkdir();p=D/'PLAN_LOCK.json';p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'n':12,'plan_sha256':sha(p),'renderer_sha256':plan['renderer_sha256']}))
