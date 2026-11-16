"""No model or labels: inspect existing source projection scope before hypothesis lock."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round104';coll=json.loads((R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json').read_text());out=[]
for src in ['waymo','drone','quad']:
 p=R/coll['modules'][src]['state_path']
 with np.load(p,allow_pickle=False) as z:
  W=z['late.weight'];b=z['late.bias'];gate=float(z['gate_late']);out.append({'source':src,'weight_shape':list(W.shape),'bias_shape':list(b.shape),'parameters':W.size+b.size,'gate_late':gate,'tanh_gate':float(np.tanh(gate)),'projection_norm':float(np.sqrt(np.square(W.astype(float)).sum()+np.square(b.astype(float)).sum())),'source_updated':False})
(H/'SOURCE_SCOPE.json').write_text(json.dumps({'state':'EXISTING_SOURCE_LATE_RGB_SCOPE_INSPECTED_NO_UPDATE','sources':out,'GT_used':False},indent=2)+'\n');print(out)
