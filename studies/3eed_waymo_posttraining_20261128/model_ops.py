"""Lightweight LoRA and masked Qwen3-VL completion likelihoods."""
from pathlib import Path

from PIL import Image
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from bridge import CFG, MODEL, RUN, digest


class LowRank(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        rank = CFG['lora_rank']
        self.scale = CFG['lora_alpha']/rank
        self.a = nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=5**.5)

    def forward(self, value):
        residual = F.linear(F.linear(value,self.a.to(value.dtype)),self.b.to(value.dtype))
        return self.base(value)+self.scale*residual


def adapter(model):
    layers=model.model.language_model.layers[-CFG['lora_last_layers']:]
    for layer in layers:
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            attn=layer.self_attn
            setattr(attn,name,LowRank(getattr(attn,name)))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert params and all(k.endswith(('.a','.b')) for k in params)
    return params


def state(params):
    return {k:p.detach().to('cpu',copy=True) for k,p in params.items()}


@torch.no_grad()
def restore(params, snapshot):
    assert params.keys()==snapshot.keys()
    for name,value in params.items():
        value.copy_(snapshot[name])


def save(params, name):
    out=RUN/f'{name}.safetensors'
    temp=out.with_suffix('.partial')
    save_file(state(params),str(temp))
    temp.replace(out)
    return dict(file=out.name,sha256=digest(out),parameters=sum(x.numel() for x in params.values()))


def load(params, name):
    restore(params,load_file(str(RUN/f'{name}.safetensors')))


def make_model(*, trainable):
    from run_full_qwen import load_model
    processor,model,limit=load_model(MODEL)
    model.eval()
    return processor,model,limit,adapter(model) if trainable else {}


def image(path):
    with Image.open(path) as f:
        return f.convert('RGB')


def visual(processor, jobs):
    prompt=[]
    for _,text in jobs:
        chat=[dict(role='user',content=[dict(type='image'),dict(type='text',text=text)])]
        prompt.append(processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True))
    return processor(text=prompt,images=[im for im,_ in jobs],padding=True,return_tensors='pt').to('cuda')


def text_tokens(processor, values):
    end=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    return [processor.tokenizer.encode(v,add_special_tokens=False)+[end] for v in values]


def completion_logp(model, context, tokens, pad, temperature=1.):
    width=max(map(len,tokens))
    device=context['input_ids'].device
    ids=torch.tensor([x+[pad]*(width-len(x)) for x in tokens],device=device)
    mask=torch.arange(width,device=device)[None,:]<torch.tensor([len(x) for x in tokens],device=device)[:,None]
    extended=dict(context)
    extended['input_ids']=torch.cat((context['input_ids'],ids),dim=1)
    for field in ('attention_mask','mm_token_type_ids','token_type_ids'):
        if field in context:
            added=mask.to(context[field].dtype) if field=='attention_mask' else torch.zeros_like(ids,dtype=context[field].dtype)
            extended[field]=torch.cat((context[field],added),dim=1)
    logits=model(**extended,use_cache=False,logits_to_keep=width+1).logits[:,:-1,:].float()/temperature
    selected=logits.gather(2,ids[...,None]).squeeze(2)-torch.logsumexp(logits,dim=-1)
    return selected,mask


@torch.no_grad()
def generate(processor, model, jobs, maximum, *, count=1, temp=None):
    context=visual(processor,jobs)
    options=dict(do_sample=temp is not None,max_new_tokens=maximum,num_return_sequences=count)
    if temp is not None:
        options.update(temperature=temp,top_k=0,top_p=1.)
    else:
        options.update(temperature=None,top_p=None,top_k=None)
    output=model.generate(**context,**options)[:,context.input_ids.shape[1]:]
    ends=model.generation_config.eos_token_id
    ends={ends} if isinstance(ends,int) else set(ends)
    pad=processor.tokenizer.pad_token_id
    results=[]
    for seq in output.tolist():
        end=next((i+1 for i,v in enumerate(seq) if v in ends),len(seq))
        results.append([v for v in seq[:end] if v!=pad or v in ends])
    texts=processor.batch_decode(results,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return texts,results


@torch.no_grad()
def choices(processor, model, jobs, marks):
    ctx=visual(processor,jobs)
    logits=model(**ctx,use_cache=False,logits_to_keep=1).logits[:,0].float()
    numbers={i:processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(9)}
    assert all(len(x)==1 for x in numbers.values())
    return [valid[logits[j,[numbers[k][0] for k in valid]].argmax().item()] for j,valid in enumerate(marks)]
