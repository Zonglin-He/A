"""Pinned official LLaVA-ST backbone; no annotation loader or evaluation code."""
import ast,copy,gc,hashlib,json,re,sys,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'runtime/llava_st/upstream'
CHECKPOINT=ROOT/'checkpoints/llava-st/LLaVA-ST-Qwen2-7B'
REVISION='bacf6d61e1de27a78fc0025083aa445d3d41b0b3'


def native_functions():
    """Use precisely the two official prompt functions, not LocalDataset/GT code."""
    import transformers
    import llava.constants as constants
    from llava import conversation as conversation_lib
    p=UPSTREAM/'inference/multi_task_inference.py';tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in {'preprocess_qwen','preprocess_multimodal'}]
    assert len(nodes)==2
    scope=dict(torch=torch,transformers=transformers,re=re,conversation_lib=conversation_lib,**{k:v for k,v in vars(constants).items()if k.isupper()})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope);return scope


@contextmanager
def embedded_vision_initialization():
    """Construct exact SigLIP architecture, restore all weights from merged shard.

    Official builder downloads an unnecessary base SigLIP before overwriting
    it with the vision weights already present in the merged checkpoint.
    """
    import llava.model.multimodal_encoder.siglip_encoder as siglip
    original=siglip.SigLipVisionTower.load_model
    def construct(self,device_map=None):
        if self.is_loaded:return
        self.vision_tower=siglip.SigLipVisionModel(self.config)
        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head=torch.nn.Identity();self.vision_tower.requires_grad_(False);self.is_loaded=True
    siglip.SigLipVisionTower.load_model=construct
    try:yield
    finally:siglip.SigLipVisionTower.load_model=original


def load():
    sys.path.insert(0,str(UPSTREAM))
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM,LlavaQwenConfig
    raw=json.loads((CHECKPOINT/'config.json').read_text());vocab=raw.get('vocab_size',raw['text_config']['vocab_size']);config=LlavaQwenConfig.from_dict({**{k:v for k,v in raw.items()if k!='text_config'},'vocab_size':vocab,'model_type':'llava_qwen'})
    assert config.hidden_size==3584 and config.num_hidden_layers==28 and vocab==151660
    tokenizer=AutoTokenizer.from_pretrained(str(CHECKPOINT),local_files_only=True,use_fast=False)
    assert len(tokenizer)==vocab+300
    with embedded_vision_initialization():
        model,info=LlavaQwenForCausalLM.from_pretrained(str(CHECKPOINT),config=config,torch_dtype=torch.float16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    learned=set(dict(model.named_parameters()));assert not(set(info['missing_keys'])&learned),info
    assert not info['unexpected_keys']and not info.get('mismatched_keys'),info
    model.eval().requires_grad_(False);model.model.init_vision_config(tokenizer);model.config.max_frame=100
    return model,tokenizer,model.get_vision_tower().image_processor,dict(loading_info=info,vocab_size=vocab,coordinate_tokens=300,precision='fp16_no_quantization',attention='sdpa',upstream_revision=REVISION,vision_weights='entirely from same merged checkpoint; no auxiliary base weights',configuration_compatibility='flatten legacy text_config.vocab_size into Qwen2 config; 151660 frozen base rows+300 coordinate rows, no resize/reinitialization')


def parse_output(text,frame_ids_100):
    """Original 100-bin coordinates -> actual physical frames, no label timing."""
    ids=np.asarray(frame_ids_100);assert len(ids)==100
    span=re.search(r'\{\s*<TEMP-(\d+)>\s*,?\s*<TEMP-(\d+)>\s*,?\s*\}',text)
    if span is None:return dict(valid=False,reason='missing_native_temporal_span')
    a,b=map(int,span.groups())
    if not(0<=a<b<=99):return dict(valid=False,reason='invalid_native_temporal_span')
    pattern=r'<TEMP-(\d+)>\s*,?\s*:\s*\[\s*<WIDTH-(\d+)>\s*,?\s*<HEIGHT-(\d+)>\s*,?\s*<WIDTH-(\d+)>\s*,?\s*<HEIGHT-(\d+)>\s*,?\s*\]'
    points={};rejected=0
    for match in re.finditer(pattern,text):
        t,x,y,u,v=map(int,match.groups())
        if not(0<=t<=99 and 0<=x<u<=99 and 0<=y<v<=99):rejected+=1;continue
        points[int(ids[t])]=np.asarray([x,y,u,v],dtype=np.float64)/99.
    if not points:return dict(valid=False,reason='missing_or_invalid_native_spatial_track',Native_interval=[int(ids[a]),int(ids[b])+1],rejected_boxes=rejected)
    fids=np.unique(ids);keys=sorted(points);boxes=np.stack([points[t]for t in keys]);xy=np.column_stack([np.interp(fids,keys,boxes[:,k])for k in range(4)])
    source=np.concatenate(((xy[:,:2]+xy[:,2:])/2,xy[:,2:]-xy[:,:2]),1).astype(np.float32)
    return dict(valid=True,Source=source,frame_ids=fids.tolist(),Native_interval=[int(ids[a]),int(ids[b])+1],generated_track_points=len(points),rejected_boxes=rejected,source_interpolation='linear xyxy between native generated points; nearest endpoint hold outside generated-point hull, common to Source/Ours; not official ST-Align sparse metric')


@torch.inference_mode()
def capture(model,tokenizer,processor,frames,frame_ids,caption):
    from llava.constants import DEFAULT_VIDEO_TOKEN
    from stvg.positions import observation_positions
    assert len(frames)==len(frame_ids)and len(frame_ids)>=8
    # Same old input support, sampled to the official 100-frame token interface.
    chosen=np.rint(np.linspace(0,len(frames)-1,100)).astype(int);ids=np.asarray(frame_ids)[chosen];images=frames[chosen]
    prompt=DEFAULT_VIDEO_TOKEN+'\nGiven the textual query: "'+caption+'". Locate the corresponding subject/object in this video. Please firstly give the timestamps, and then give the spatial bounding box corresponding to each timestamp in the time period.'
    functions=native_functions();vc=model.model.vision_config
    conversations=[{'from':'human','value':prompt},{'from':'gpt','value':None}]
    source=functions['preprocess_multimodal'](copy.deepcopy([conversations]),vc)
    input_ids=functions['preprocess_qwen'](source[0],tokenizer,has_image=True).cuda()
    variables={key:[]for key in ['temporal_input_locations','temporal_output_locations','spatial_height_input_locations','spatial_height_output_locations','spatial_width_input_locations','spatial_width_output_locations']}
    pixels=processor.preprocess(images,return_tensors='pt')['pixel_values'].cuda().half();captured=[]
    hook=model.model.vision_resampler.token_packer.register_forward_hook(lambda _m,_a,o:captured.append(o.detach().float().cpu()))
    seed=int.from_bytes(hashlib.sha256(('41\0'+caption+'\0'+str(list(frame_ids))).encode()).digest()[:4],'big');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);tick=time.perf_counter()
    try:
        result=model.generate(input_ids,images=[pixels],modalities=['video'],variables=[variables],do_sample=True,temperature=.01,top_p=None,num_beams=1,max_new_tokens=1024,use_cache=True)
    finally:hook.remove()
    torch.cuda.synchronize();seconds=time.perf_counter()-tick;text=tokenizer.batch_decode(result,skip_special_tokens=False)[0].replace('<|im_end|>','');parsed=parse_output(text,ids)
    assert len(captured)==1 and captured[0].shape==(100,81,3584),[x.shape for x in captured]
    unique=np.unique(ids);features=torch.stack([captured[0][np.flatnonzero(ids==t)].mean(0)for t in unique]);assert torch.isfinite(features).all()
    parsed.update(frame_ids=unique.tolist(),native_text=text,seed=seed,seconds=seconds,model_frames_100=ids.tolist(),features=features,valid_cells=torch.ones((len(unique),81),dtype=torch.bool),grid_shape=(9,9),feature_interface='actual first spatial STP 9x9 grid,3584ch,before temporal compression; LAPE retained; duplicate physical frame features averaged',new_teacher_calls=0)
    if not parsed['valid']:
        parsed['Source']=np.zeros((len(unique),4),dtype=np.float32);parsed.setdefault('Native_interval',[-1,-1])
    if parsed['valid']:
        extent=[list(unique).index(parsed['Native_interval'][0]),list(unique).index(parsed['Native_interval'][1]-1)];parsed['positions8']=observation_positions(unique,extent)
    del pixels,result,captured;gc.collect();torch.cuda.empty_cache();return parsed
