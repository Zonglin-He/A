"""Metadata-only source-module activity, not a task or parameter sweep."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round111';out=[]
for src in ['waymo','drone','quad']:
 p=R/('crossplatform/round91/trained_source_attempt001/states_PRIVATE/R76_full_disc_e1.npz' if src=='waymo' else 'crossplatform/round97/'+src+'/SELECTED_SOURCE_ONLY.npz');s=dict(np.load(p));x={'source':src,'gate_P_tanh':float(np.tanh(s['gate_P'])),'gate_I_tanh':float(np.tanh(s['gate_I'])),'gate_late_tanh':float(np.tanh(s['gate_late']))}
 for key in ['point_adapter.up.weight','point_adapter.up.bias','image_adapter.up.weight','image_adapter.up.bias']:x[key.replace('.','_')+'_L2']=float(np.linalg.norm(s[key].astype(float)))
 out.append(x)
(H/'SOURCE_MODULE_ACTIVITY.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
