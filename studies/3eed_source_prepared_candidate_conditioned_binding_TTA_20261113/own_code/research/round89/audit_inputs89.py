"""Validate saved source labels/features and fixed original checkpoint, no new scoring."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(R/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json')+read(R/'research/round76/input/SOURCE_VAL128_PRIVATE.json');L=R/'research/round25/experiment/source_labels_attempt002';ss=read(L/'SEALED.json');C=R/'crossplatform/round89/source_features640_attempt001';cs=read(C/'SEALED.json');ts=read(R/'crossplatform/round89/trained_source_attempt001/SEALED.json');n=0
for r in rows:
 f=L/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==ss['files'][f.relative_to(L).as_posix()];a=np.load(f,allow_pickle=False)['ious'];assert a.shape==(256,) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1;n+=1
for root,se in [(C,cs),(R/'crossplatform/round89/trained_source_attempt001',ts)]:
 for f,h in se['files'].items():assert sha(root/f)==h
assert sha(R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';assert sha(R/'crossplatform/round76/trained_source_attempt002/states_PRIVATE/bidirectional_e06.npz')=='84f3455b52c1ca9ded0dca394af17b9ea9949cafb720eb24728c505e79ed9f3e';out={'state':'PASS_R89_ORIGINAL_SOURCE_LABEL_FEATURE_TRAINING_SEALS_AND_CHECKPOINTS','source_label_rows':n,'source_feature_files':len(cs['files']),'source_training_files':len(ts['files']),'original_A256_label_count':256,'new_task_scoring':False,'target_labels_read':False};(H/'INPUT_SEAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
