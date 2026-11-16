"""Postseal parameter identity audit; no new prediction or GT access."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round111';B=R/'crossplatform/round111';out=[]
for src in ['waymo','drone','quad']:
 A=dict(np.load(B/(src+'_HEAD_NATIVE_PRIVATE.npz')));P=dict(np.load(B/(src+'_HEAD_PREPARED_PRIVATE.npz')));assert set(A)==set(P);d=np.concatenate([(P[k].astype(float)-A[k].astype(float)).ravel() for k in A]);out.append({'source':src,'all_native_and_prepared_projection_values_bitexact':bool(np.all(d==0)),'parameter_L2':float(np.linalg.norm(d)),'max_abs_change':float(abs(d).max()),'changed_values':int(np.count_nonzero(d))})
(H/'HEAD_STATE_IDENTITY.json').write_text(json.dumps(out,indent=2)+'\n');p=H/'REPORT.md';p.write_text(p.read_text()+'\n## 投影副本的实际参数核对（预测封存后）\n\n```json\n'+json.dumps(out,indent=2)+'\n```\n\n若两个投影权重本来位等，互换分数一致是预期结果，不是判别性语义修复。该核对只排除投影副本漂移，不能据此把全部差异归给某一条RGB或点云模块。原native与prepared的差异发生在送入投影的表示/文字条件，以及额外late与竞争读出；后两者在当前基础项之上有正也有负作用。\n');print(out)
