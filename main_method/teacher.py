"""One frozen Qwen3.5, unchanged Caption/pairwise/Direct interfaces."""
import hashlib
import time
from .runtime import ROOT, BASE, put

def load():
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from scripts.stvg_qwen35_geometry32_v1 import CHECKPOINT
    p = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=False)
    m = Qwen3_5ForConditionalGeneration.from_pretrained(CHECKPOINT, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, device_map={'':'cuda:0'}, low_cpu_mem_usage=True).eval()
    m.requires_grad_(False)
    assert m.config.model_type == 'qwen3_5' and not getattr(m,'is_quantized',False)
    assert all(x.dtype in (torch.bfloat16,torch.float32) for x in m.parameters() if x.is_floating_point())
    return m,p

def caption_batch(p, video, fps):
    import torch
    from transformers.video_utils import VideoMetadata
    from scripts.stvg_qwen_b3_temporal32_448_v2 import configure448
    from scripts.stvg_table1_temporal_adapter_v1 import caption_prompt
    configure448(p)
    text=p.apply_chat_template([dict(role='user',content=[dict(type='video'),dict(type='text',text=caption_prompt())])],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    b=p(text=[text],videos=[torch.from_numpy(video.copy())],return_tensors='pt',add_special_tokens=False,
        do_sample_frames=False,video_metadata=[VideoMetadata(total_num_frames=8,fps=float(fps),frames_indices=list(range(8)),duration=8/float(fps))],return_metadata=True)
    b.pop('video_metadata',None)
    assert b['video_grid_thw'][0].tolist()==[4,28,28]
    assert int((b['input_ids']==248057).sum())==784
    return b,text

def generate(m,p,b,cap):
    import torch
    n=b['input_ids'].shape[1]; tick=time.perf_counter()
    with torch.inference_mode():
        out=m.generate(**b.to(m.device),max_new_tokens=cap,do_sample=False,use_cache=True)
    torch.cuda.synchronize()
    tokens=out[0,n:].cpu().tolist()
    # Preserve the original Caption truncation convention, including EOS at cap.
    return dict(raw=p.tokenizer.decode(tokens,skip_special_tokens=True).strip(),
        tokens=len(tokens),token_ids=tokens,truncated=len(tokens)>=cap,wall_seconds=time.perf_counter()-tick)

def pairwise(m,p,query,captions,*,last_token_only=False):
    import torch
    from scripts.stvg_pairwise_primitives_v1 import prompt,combine,LABELS
    ids=[p.tokenizer.encode(x,add_special_tokens=False) for x in LABELS]
    assert all(len(x)==1 for x in ids)
    ids=[x[0] for x in ids]; orders=[]
    for order in ([0,1],[1,0]):
        text=prompt(query,[captions[i] for i in order])
        rendered=p.apply_chat_template([dict(role='user',content=[dict(type='text',text=text)])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        b=p(text=[rendered],return_tensors='pt',add_special_tokens=False).to(m.device)
        assert b['input_ids'].shape[1]<=4096
        with torch.inference_mode():
            out=m(**b,use_cache=False,**({'logits_to_keep':1} if last_token_only else {}))
        z=out.logits[0,-1].float().log_softmax(-1)[ids]
        orders.append(dict(order=list(order),posterior=z.softmax(0).cpu().double().tolist(),
            raw_label_logprobs=z.cpu().double().tolist(),prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
        del b,out
    posterior,rewards=combine(*[x['posterior'] for x in orders])
    return dict(orders=orders,posterior=posterior,rewards=rewards)

def direct_batch(p,job,image,context):
    from PIL import Image
    from scripts.stvg_qwen35_geometry_contract_v1 import messages
    from scripts.stvg_qwen35_geometry_run_v5 import native_video_adapter
    # A dedicated processor prevents Temporal's total-video budget mutating Direct.
    text=p.apply_chat_template(messages(job),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return native_video_adapter(p,text,Image.fromarray(image),context,job['context'])

def direct(m,p,job,image,context):
    from scripts.stvg_qwen35_geometry_run_v4 import parse_final
    b,a=direct_batch(p,job,image,context)
    d=generate(m,p,b,256)
    eos=m.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else (eos or [])
    d['truncated']=len(d['token_ids'])>=256 and (not d['token_ids'] or d['token_ids'][-1] not in eos)
    d['parsed']=parse_final(d['raw'],job['job_id'],truncated=d['truncated'])
    d['input_audit']=a
    return d
