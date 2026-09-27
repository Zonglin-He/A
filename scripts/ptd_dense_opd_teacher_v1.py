"""FP32 decoder, canonical coordinate trie, immutable conditioning prefix. No GT."""
import gc
import os
import re
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from scripts.ptd_spatial_adapter_ab_v1 import frames_for

TEACHER=Path(__file__).resolve().parents[1]/'checkpoints/Qwen3-VL-8B-Instruct'


class Teacher:
    def __init__(self,budget):
        self.budget=budget
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        # The default fused SDPA path drifted even on repeated identical FP32
        # prefixes. Explicit math SDPA makes independent chunking reproducible.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.pr=AutoProcessor.from_pretrained(TEACHER,local_files_only=True)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(TEACHER,local_files_only=True,dtype=torch.bfloat16,
                    device_map='cpu',attn_implementation='sdpa').eval().requires_grad_(False)
        lm=self.model.model.language_model
        for layer in lm.layers:
            layer.to(device='cuda',dtype=torch.float32)
        lm.norm.to(device='cuda',dtype=torch.float32)
        lm.rotary_emb.to('cuda')
        self.model.lm_head.to('cuda')
        self.events=[self.pr.tokenizer.encode(str(i)+'\n',add_special_tokens=False) for i in range(1001)]
        assert len(set(map(tuple,self.events)))==1001
        # All paths end in a separate newline; trie nodes contain only numeric prefixes.
        newline=self.pr.tokenizer.encode('\n',add_special_tokens=False)
        assert len(newline)==1 and all(e[-1:]==newline for e in self.events)
        prefixes={tuple(e[:k]) for e in self.events for k in range(1,len(e))}
        self.nodes=sorted(prefixes,key=lambda p:(len(p),p))
        self.node_index={p:i for i,p in enumerate(self.nodes)}
        self.keep=sorted({t for e in self.events for t in e})
        self.columns={t:i for i,t in enumerate(self.keep)}
        self.budget.check()

    def embed(self,ids):
        return self.model.model.language_model.embed_tokens(ids.cpu()).float().cuda()

    @torch.inference_mode()
    def lognext(self,h):
        # BF16 checkpoint values promoted exactly; FP32 multiplication and denominator.
        weight=self.model.lm_head.weight
        output=[]
        for a in range(0,len(h),32):
            logits=torch.cat([torch.nn.functional.linear(h[a:a+32].float(),weight[b:b+8192].float())
                              for b in range(0,len(weight),8192)],-1)
            output.append(logits.double().log_softmax(-1)[:,self.keep].cpu())
        return torch.cat(output)

    def inputs(self,row,z,branch):
        ref=re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>',z['completion'],re.S).group(1)
        s,e=z['interval'];content=[];images=[]
        if branch=='plus':
            frames,_=frames_for(row,'clean');h,w=frames.shape[1:3];scale=min(1.,448/max(h,w))
            hh=max(32,min(448,round(h*scale/32)*32));ww=max(32,min(448,round(w*scale/32)*32))
            # Match PTD's actual resized pixels; visual encoders remain model-specific.
            x=torch.from_numpy(frames).permute(0,3,1,2).float()
            x=torch.nn.functional.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False,antialias=True)
            images=[v for v in x]
        for i,fid in enumerate(row['input']['frame_ids']):
            content.append(dict(type='text',text=f'Frame {i+1}; time {fid/row["input"]["fps"]:.6f} seconds.'))
            if branch=='plus':content.append(dict(type='image'))
        content.append(dict(type='text',text=(f'Query: {row["input"]["caption"]}\nObject reference: {ref}\n'
            f'Fixed event time: frames {s+1} through {e+1}.\n'
            'Localize this object at the requested frame. Coordinates are normalized integers from 0 to 1000. '
            'x1 is left, y1 is top, x2 is right, y2 is bottom. Return only the requested coordinate as one '
            'canonical decimal integer followed by a newline. Do not output other coordinates or explanations.')))
        text=self.pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        inp=self.pr(text=[text],images=images or None,do_resize=False,return_tensors='pt')
        if branch=='minus':
            assert not any(k in inp for k in ['pixel_values','image_grid_thw'])
            assert not any(t in inp['input_ids'] for t in [self.pr.tokenizer.convert_tokens_to_ids(x) for x in ['<|vision_start|>','<|vision_end|>','<|image_pad|>','<|video_pad|>']])
        return inp,text

    @torch.inference_mode()
    def prepare(self,inp):
        core=self.model.model;ids=inp['input_ids'].cuda();emb=self.embed(ids);mask=None;deep=None
        if 'pixel_values' in inp:
            core.visual.to('cuda')
            visual=core.get_image_features(inp['pixel_values'].cuda(),inp['image_grid_thw'].cuda(),return_dict=True)
            feats=torch.cat(visual.pooler_output,0).float();mask=ids==core.config.image_token_id
            assert int(mask.sum())==len(feats)
            emb[mask]=feats;deep=[v.float() for v in visual.deepstack_features]
            del visual,feats
            core.visual.to('cpu');torch.cuda.empty_cache()
            pos,delta=core.get_rope_index(ids,mm_token_type_ids=inp['mm_token_type_ids'].cuda(),
                                          image_grid_thw=inp['image_grid_thw'].cuda(),attention_mask=inp['attention_mask'].cuda())
        else:
            pos=torch.arange(ids.shape[1],device='cuda').view(1,1,-1).expand(3,1,-1)
            delta=torch.zeros((1,1),device='cuda',dtype=torch.long)
        return dict(emb=emb,pos=pos,mask=mask,deep=deep,delta=delta,ids=ids)

    @torch.inference_mode()
    def prefill(self,p,extra=None,chunk=256):
        emb=p['emb'];pos=p['pos'];n=emb.shape[1]
        if extra:
            ids=torch.tensor([extra],device='cuda');emb=torch.cat([emb,self.embed(ids)],1)
            ep=torch.arange(n,n+len(extra),device='cuda').view(1,1,-1).expand(3,1,-1)+p['delta'].view(1,1,1)
            pos=torch.cat([pos,ep],-1)
        cache=None;hidden=[];lm=self.model.model.language_model
        for a in range(0,emb.shape[1],chunk):
            b=min(emb.shape[1],a+chunk);vm=None;dd=None
            if p['mask'] is not None and a<n:
                vm=p['mask'][:,a:min(b,n)]
                if b>n:vm=torch.cat([vm,torch.zeros((1,b-n),device='cuda',dtype=torch.bool)],1)
                lo=int(p['mask'][:,:a].sum());hi=int(p['mask'][:,:min(b,n)].sum())
                dd=[d[lo:hi] for d in p['deep']] if hi>lo else None
            causal=torch.zeros((b-a,b),device='cuda',dtype=torch.float32)
            causal.masked_fill_(torch.arange(b,device='cuda')[None,:]>torch.arange(a,b,device='cuda')[:,None],-torch.inf)
            out=lm(inputs_embeds=emb[:,a:b],position_ids=pos[:,:,a:b],past_key_values=cache,attention_mask=causal[None,None],
                   use_cache=True,visual_pos_masks=vm,deepstack_visual_embeds=dd,return_dict=True)
            cache=out.past_key_values
            if extra and b>=n:
                hidden.append(out.last_hidden_state[:,max(0,n-1-a):].cpu())
            last=out.last_hidden_state[:,-1].cpu()
            self.budget.calls+=1;self.budget.tokens+=b-a;self.budget.check()
        return cache,torch.cat(hidden,1) if extra else last

    @torch.inference_mode()
    def suffix(self,cache,ids,delta,mask=None,offset=None):
        n=cache.get_seq_length();m=len(ids)
        if mask is None:
            mask=torch.zeros((m,n+m),device='cuda',dtype=torch.float32)
            mask.masked_fill_(torch.arange(n+m,device='cuda')[None,:]>(n+torch.arange(m,device='cuda'))[:,None],-torch.inf)
            mask=mask[None,None]
        pos=(torch.arange(m,device='cuda')+n if offset is None else offset).view(1,1,-1).expand(3,1,-1)+delta.view(1,1,1)
        out=self.model.model.language_model(inputs_embeds=self.embed(torch.tensor([ids],device='cuda')),
            position_ids=pos,past_key_values=cache,use_cache=True,attention_mask=mask,
            visual_pos_masks=None,deepstack_visual_embeds=None,return_dict=True)
        self.budget.calls+=1;self.budget.tokens+=m
        return out.last_hidden_state[0]

    @torch.inference_mode()
    def distribution(self,p,cache,selector):
        base_n=cache.get_seq_length();ids=self.pr.tokenizer.encode(selector,add_special_tokens=False)
        first=self.lognext(self.suffix(cache,ids,p['delta'])[-1:])[0];n=cache.get_seq_length()
        # Trie ancestors are the only visible suffix nodes, never sibling coordinate values.
        node_lp=[];node_raw=[]
        for a in range(0,len(self.nodes),64):
            nodes=self.nodes[a:a+64];b=a+len(nodes)
            att=torch.full((len(nodes),n+b),-torch.inf,device='cuda',dtype=torch.float32);att[:,:n]=0
            for j,path in enumerate(nodes):
                for k in range(1,len(path)+1):att[j,n+self.node_index[path[:k]]]=0
            offsets=torch.tensor([n+len(v)-1 for v in nodes],device='cuda')
            lp=self.lognext(self.suffix(cache,[v[-1] for v in nodes],p['delta'],att[None,None],offsets))
            for j,path in enumerate(nodes):
                parent=self.node_index.get(path[:-1]);raw=first[self.columns[path[0]]] if parent is None else node_raw[parent]+node_lp[parent][self.columns[path[-1]]]
                node_raw.append(raw);node_lp.append(lp[j])
            self.budget.check()
        raw=torch.stack([node_raw[self.node_index[tuple(e[:-1])]]+node_lp[self.node_index[tuple(e[:-1])]][self.columns[e[-1]]] for e in self.events])
        cache.crop(base_n);mass=torch.logsumexp(raw,0);q=raw-mass
        assert torch.isfinite(q).all() and abs(float(q.exp().sum())-1)<1e-12
        return dict(logp=q,raw=raw,log_event_mass=float(mass),prefix_tokens=n,numeric_trie_nodes=len(self.nodes))

    @torch.inference_mode()
    def independent(self,p,selector,value):
        sid=self.pr.tokenizer.encode(selector,add_special_tokens=False);event=self.events[value]
        # Fresh cache, ordinary full causal sequence, different chunk boundary; no trie.
        cache,h=self.prefill(p,sid+event,chunk=191)
        logits=self.lognext(h[0,-len(event)-1:-1].cuda())
        result=float(logits.gather(-1,torch.tensor([self.columns[v] for v in event])[:,None]).sum())
        del cache,h;gc.collect();torch.cuda.empty_cache()
        return result
