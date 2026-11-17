from pathlib import Path
import torch,json,hashlib
R=Path(__file__).resolve().parents[2];p=R/'crossplatform/round128/checkpoints/cloudfixer_modelnet40.pt';v=torch.load(p,map_location='cpu',weights_only=True);print(type(v),len(v),[(k,tuple(x.shape)) for k,x in list(v.items())[:12]])
(R/'research/round128/CLOUDFIXER_WEIGHTS_AUDIT.json').write_text(json.dumps({'weights_only_safe_load':True,'file':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'state_keys':len(v),'tensor_elements':sum(x.numel() for x in v.values())},indent=2)+'\n')
