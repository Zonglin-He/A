from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round122';B=R/'crossplatform/round122';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert rd(H/'NATIVE_AUDIT.json')['n']==2304 and rd(H/'HEADMEAN_AUDIT.json')['n']==2304
m=rd(B/'MATRIX_SEALED.json');extra=rd(B/'headmean_controls/MATRIX_SEALED.json');m['arms']+=extra['arms'];m['new_arms']+=extra['arms'];m['original_prediction_seal_sha256']=sha(B/'MATRIX_SEALED.json');m['strong_direct_prediction_seal_sha256']=sha(B/'headmean_controls/MATRIX_SEALED.json');(B/'EVALUATION_MATRIX.json').write_text(json.dumps(m,indent=2)+'\n');print('ASSEMBLED122_22ARMS')
