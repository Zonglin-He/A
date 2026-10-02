"""CPU query parser and immutable CLIP configuration before token inference/GT."""
import os,sys,time,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.tastvg_transfer_aligned_common_v1 import *
def run():
 import torch,stanza,transformers
 from vg_tta.tastvg_aligned_token_binding_v1 import phrases
 torch.set_num_threads(2);inputs={};bind=lambda f:inputs.update({str(f.relative_to(ROOT)):sha(f)})
 asset=ROOT/'checkpoints/clip_vit_b16_token_v1';r=read(asset/'DOWNLOAD_RECEIPT.json');assert r['revision']=='57c216476eefef5ab752ec549e440a49ae4ae5f3'
 for f in r['files']:assert sha(asset/f['file'])==f['sha256'];bind(asset/f['file'])
 bind(asset/'DOWNLOAD_RECEIPT.json');bind(ROOT/'methods/CURRENT_METHOD.json');bind(OLD/'FINAL_COMPLETION.json')
 nlp=stanza.Pipeline('en',dir=str(ROOT/'.cache/stanza'),package=None,processors={'tokenize':'ewt','mwt':'ewt','pos':'ewt_nocharlm','lemma':'ewt_nocharlm','depparse':'ewt_nocharlm'},use_gpu=False,download_method=None,verbose=False)
 parsed={}
 for ds in ['hc2','vidstg']:
  qp=read(QUAL/ds/'PLAN.json');p=read(OLD/ds/'PLAN.json');bind(QUAL/ds/'PLAN.json');bind(OLD/ds/'PLAN.json');records=[]
  for parent in sorted({c['parent'] for c in qp['cells']}):
   caption=p['rows'][parent]['input']['caption'];doc=nlp(caption);words=[];offset=0
   for sent in doc.sentences:
    for w in sent.words:words.append(dict(id=w.id+offset,text=w.text,lemma=w.lemma,upos=w.upos,head=w.head+offset if w.head else 0,deprel=w.deprel))
    offset+=len(sent.words)
   records.append(dict(parent=parent,phrases=phrases(words),words=words,GT_read=False))
  write(BASE/'token'/ds/'PHRASES_PRIVATE.json',records);bind(BASE/'token'/ds/'PHRASES_PRIVATE.json');parsed[ds]=dict(sources=len(records),object_empty=sum(not r['phrases']['object'] for r in records),event_empty=sum(not r['phrases']['event'] for r in records),rules=[r['phrases']['referent_rule'] for r in records])
  for c in qp['cells']:
   bind(ROOT/c['payload']);f=POOL/ds/'capture'/c['condition']/f'{c["parent"]:05}.json';bind(f);cr=read(f);assert sha(POOL/ds/cr['cache'])==cr['sha256']
 names=['protocols/tastvg_transfer_aligned_v1.md','scripts/tastvg_transfer_aligned_common_v1.py','scripts/prepare_tastvg_aligned_token_v1.py','scripts/run_tastvg_aligned_token_v1.py','scripts/test_tastvg_transfer_aligned_v1.py','vg_tta/tastvg_aligned_token_binding_v1.py','scripts/run_tastvg_full_b1_experts_v1.py','vg_tta/tastvg_deployment_corruption_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py','vg_tta/exact_frame_decode_audit_v2.py']
 module=Path(transformers.__file__).parent/'models/clip/modeling_clip.py';bind(module) if module.is_relative_to(ROOT) else None
 write(BASE/'TOKEN_RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},inputs=inputs,checkpoint=dict(repository=r['repository'],revision=r['revision']),environment=dict(transformers=transformers.__version__,stanza=stanza.__version__,torch=torch.__version__,clip_modeling_sha256=sha(module)),parser_summary=parsed,cells=60,candidates=540,GT_read=False,time=time.time()))
 status(BASE/'token/STATUS.json',dict(status='ready_after_transfer_GPU',done=0,total=60,time=time.time()));print('TOKEN parser and runtime locked',parsed)
if __name__=='__main__':run()
