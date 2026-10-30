"""CPU-only source-Waymo caption capacity/first12batch audit. No model/GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from transformers import RobertaTokenizerFast
R=Path(__file__).resolve().parent;P=R.parent;J=P.parent/'research/round6/source_data_audit/OFFICIAL_TRAIN_ANNO_ORDER_PRIVATE.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=json.loads(J.read_text());texts=[];metadata={}
 for r in rows:
  path=Path(r['meta_path']);metadata.setdefault(str(path),json.loads(path.read_text()));caption=metadata[str(path)]['ground_info'][r['object_index']]['caption'];assert hashlib.sha256(caption.encode()).hexdigest()==r['caption_sha256'];texts.append(' '.join(caption.replace(',',' ,').split())+' . not mentioned')
 tok=RobertaTokenizerFast.from_pretrained(str((P/'upstream/data/roberta_base').resolve()));lengths=[]
 for i in range(0,len(texts),256):lengths.extend(len(x) for x in tok(texts[i:i+256],padding=False,truncation=False)['input_ids'])
 assert len(lengths)==3682;order=torch.randperm(3682,generator=torch.Generator().manual_seed(1)).tolist();batches=[max(lengths[i] for i in order[k:k+24]) for k in range(0,3672,24)];assert len(batches)==153
 out={'state':'CPU_source_train_token_lengths_complete','n':3682,'actual_text_format':'comma-spacing normalized+ . not mentioned','max_actual_tokens':max(lengths),'token_percentiles_50_90_95_99':np.percentile(lengths,[50,90,95,99]).tolist(),'original_BBF_256score_capacity_exceeded':sum(t>256 for t in lengths),'epoch1_B24_max_tokens_minmax':[min(batches),max(batches)],'timing_first12_B24_token_maxima':batches[:12],'timing_first12_max_tokens':max(batches[:12]),'global_max_minus_timing_max':max(lengths)-max(batches[:12]),'source_join_sha256':sha(J),'code_sha256':sha(Path(__file__)),'no_model_or_GPU':True,'no_target_Drone_Quad':True,'risk':'Token length audit is not a guarantee of allocator peak; retainbatch24 unless actualOOM, never preemptively changeBN/microbatch.'};(R/'TRAIN_TOKEN_LENGTH_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
