"""Isolated official LLaVA-ST loader and matched-evidence inference interface.

Run only with the dedicated pinned Transformers runtime. Never import into a
live PTD worker. No external output is treated as a PTD probability distribution.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import torch

@contextmanager
def cwd(path):
    old=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)

def imports(official):
    # Official inference module resolves config.yaml relative to its repository.
    # Preserve that import behavior without changing process cwd during inference.
    with cwd(official):
        from inference.multi_task_inference import preprocess_multimodal,preprocess_qwen
        from inference.src.utils import get_variables
    return preprocess_multimodal,preprocess_qwen,get_variables

def load(checkpoint):
    from transformers import AutoTokenizer,AutoConfig
    from llava.model.language_model.llava_qwen import LlavaQwenConfig,LlavaQwenForCausalLM
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel,SigLipVisionConfig
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    config=AutoConfig.from_pretrained(checkpoint,local_files_only=True)
    # Match official builder: released composite config delegates vocabulary to text_config.
    assert config.vocab_size==config.text_config.vocab_size
    assert len(tokenizer)==config.vocab_size+config.num_temporal_tokens+2*config.num_spatial_tokens
    # The released full checkpoint contains all 421 vision tensors. The official
    # constructor first downloads an unrelated initial copy of SigLIP then loads
    # these tensors. Build its exact class/shape locally instead; strict loading
    # info below forbids leaving any fresh/random parameter in the teacher.
    original=SigLipVisionModel.__dict__['from_pretrained'] if 'from_pretrained' in SigLipVisionModel.__dict__ else None
    SigLipVisionModel.from_pretrained=classmethod(lambda cls,*a,**kw:cls(SigLipVisionConfig()))
    try:
        model,info=LlavaQwenForCausalLM.from_pretrained(checkpoint,config=config,local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='sdpa',device_map='cuda',low_cpu_mem_usage=True,output_loading_info=True)
    finally:
        if original is None:delattr(SigLipVisionModel,'from_pretrained')
        else:SigLipVisionModel.from_pretrained=original
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys'),info
    model.eval().requires_grad_(False);model.model.init_vision_config(tokenizer)
    model.config.max_frame=100
    assert model.config.max_frame==100
    assert sum('vision_tower' in n for n,_ in model.named_parameters())==421
    return tokenizer,model,model.get_vision_tower().image_processor,info

@torch.inference_mode()
def predict(tokenizer,model,processor,frames100,caption,official,*,decode='greedy',seed=20260928):
    if decode not in ('greedy','official'):raise ValueError(decode)
    assert len(frames100)==100 and model.config.max_frame==100
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    preprocess_multimodal,preprocess_qwen,get_variables=imports(official)
    prompt=("<video>\nAt which time interval in the video can we see "+caption.rstrip('.!?')+
        " occurring? Please describe the location of the corresponding subject/object in this video."
        "Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.")
    conversations=[{'from':'human','value':prompt},{'from':'gpt','value':None}]
    conversations,variables=get_variables(conversations)
    sources=preprocess_multimodal([conversations],model.model.vision_config)
    ids=preprocess_qwen([sources[0][0],{'from':'gpt','value':None}],tokenizer,has_image=True).cuda()
    pixels=processor.preprocess(frames100,return_tensors='pt')['pixel_values'].cuda().half()
    output=model.generate(ids,images=[pixels],variables=[variables],modalities=['video'],
        do_sample=decode=='official',num_beams=1,max_new_tokens=1024,use_cache=True,
        **({'temperature':0.01,'top_p':None} if decode=='official' else {}))
    result=dict(prompt=prompt,input_token_ids=ids.cpu().tolist(),output_token_ids=output.cpu().tolist(),
        raw_text=tokenizer.batch_decode(output,skip_special_tokens=False)[0],
        processed_shape=list(pixels.shape),processed_dtype=str(pixels.dtype),
        decode=decode,seed=seed,temperature=.01 if decode=='official' else None,
        max_frame=model.config.max_frame,max_new_tokens=1024,possibly_truncated=output.shape[-1]>=1024)
    return result,pixels.cpu()

def load_official(checkpoint,siglip_path,official):
    """Unmodified official loader, with the official vision dependency cached locally.

    Only the dependency's location changes; no SigLIP from_pretrained monkeypatch.
    The full checkpoint still loads final trained vision weights.
    """
    with cwd(official):
        from llava.model.builder import load_lora_model
        tokenizer,model,processor,_=load_lora_model(None,str(checkpoint),'llava_st_qwen',
            device_map='cuda',attn_implementation='sdpa',
            overwrite_config={'mm_vision_tower':str(siglip_path)},local_files_only=True)
    model.config.max_frame=100
    assert model.config.max_frame==100
    model.eval().requires_grad_(False)
    return tokenizer,model,processor,{'loader':'official load_lora_model','local_official_siglip':str(siglip_path)}
