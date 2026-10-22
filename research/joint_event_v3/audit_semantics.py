"""Post-seal role/margin recapture, no visual forward/fit/labels."""
import os,sys,json,hashlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,numpy as np
from transformers import AutoTokenizer
from stvg.query import QuerySubjectParser
from stvg.runtime import guard
from research.joint_event_v1.run import R,D,OLD,read,put,records,text_groups,verify
from research.joint_event_v1.core import relative_event
torch.set_num_threads(1);verify();sys.addaudithook(guard)
paper=read(R/'outputs/tastvg_paper_v1/plan.json');resources={}
for path,digest in paper['assets'].items():
    if 'checkpoints/stanza/'not in path and not ('runtime/referdino/bert-from-frozen-teacher/'in path and not path.endswith('model.safetensors')):continue
    h=hashlib.sha256()
    with(R/path).open('rb')as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    assert h.hexdigest()==digest;resources[path]=digest
parser=QuerySubjectParser(R/'checkpoints/stanza');tok=AutoTokenizer.from_pretrained(str(R/'runtime/referdino/bert-from-frozen-teacher'),local_files_only=True,use_fast=True)
plan=read(D/'plan.json');count=0;private={}
for row in records():
    k=row['id'];e=plan['bindings'][k]['semantic'];sem=torch.load(OLD/e['path'],map_location='cpu',weights_only=False);tr=torch.load(D/'traces'/f'{k}.pt',map_location='cpu',weights_only=False)
    vg,ng,tm=text_groups(row['metadata']['caption'],parser,tok,sem['groups']);assert tm==tr['text_metadata']
    y=relative_event(sem['premax'].numpy(),vg,ng,sem['selected_query']);y=y[:,np.std(y,axis=0)>1e-9]
    assert np.array_equal(y,tr['semantic']['Relative']['labels'])
    private[k]=dict(predicate_groups=vg,nominal_groups=ng);count+=1
put(D/'postseal_role_groups.json',private)
proof=dict(stage='post-seal no-GT/no-fit audit',queries=count,old_event_token_mapping_exact=count,new_role_margin_bitwise_exact=count,pinned_text_resource_files_verified=len(resources),resource_digests=resources,
           new_visual_or_expert_calls=0,new_projection_fits=0,CUDA_initialized=torch.cuda.is_initialized())
put(R/'results/joint_event_hc64_v3/semantic_audit.json',proof);print(json.dumps({k:v for k,v in proof.items()if k!='resource_digests'}))
