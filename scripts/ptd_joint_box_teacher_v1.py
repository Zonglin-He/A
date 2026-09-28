"""Full canonical box-event likelihoods on sampled student prefixes, frozen FP32 teacher."""
import gc,re
import torch
from scripts.ptd_dense_opd_teacher_v1 import Teacher
from scripts.ptd_spatial_adapter_ab_v1 import frames_for

class JointTeacher(Teacher):
    def __init__(self,budget):
        super().__init__(budget)
        self.keep=sorted(set(self.keep)|set(self.pr.tokenizer.encode(',',add_special_tokens=False)))
        self.columns={t:i for i,t in enumerate(self.keep)}
    def inputs_joint(self,row,z):
        ref=re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>',z['completion'],re.S).group(1)
        s,e=z['interval'];frames,_=frames_for(row,'clean');h,w=frames.shape[1:3];scale=min(1.,448/max(h,w))
        hh=max(32,min(448,round(h*scale/32)*32));ww=max(32,min(448,round(w*scale/32)*32))
        x=torch.from_numpy(frames).permute(0,3,1,2).float();x=torch.nn.functional.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False,antialias=True)
        content=[]
        for i,fid in enumerate(row['input']['frame_ids']):content.extend([dict(type='text',text=f'Frame {i+1}; time {fid/row["input"]["fps"]:.6f} seconds.'),dict(type='image')])
        content.append(dict(type='text',text=f'Query: {row["input"]["caption"]}\nObject reference: {ref}\nFixed event time: frames {s+1} through {e+1}.\n'
            'Localize this object at the requested frame. Coordinates are normalized integers from 0 to 1000. '
            'x1 is left, y1 is top, x2 is right, y2 is bottom. Return the complete box as x1,y1,x2,y2 followed by a newline, '
            'using canonical decimal integers, commas, and no spaces. Do not output explanations.'))
        text=self.pr.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)
        inp=self.pr(text=[text],images=list(x),do_resize=False,return_tensors='pt')
        return inp,text
    def encode_box(self,box):
        chunks=[self.pr.tokenizer.encode(str(int(v))+(',' if c<3 else '\n'),add_special_tokens=False) for c,v in enumerate(box)]
        event=sum(chunks,[]);canonical=','.join(str(int(v)) for v in box)+'\n'
        assert self.pr.tokenizer.encode(canonical,add_special_tokens=False)==event
        assert all(t in self.columns for t in event)
        return event,[len(c) for c in chunks],canonical
    @torch.inference_mode()
    def score_boxes(self,p,cache,selector,boxes,text):
        encoded=[self.encode_box(b) for b in boxes];events=[v[0] for v in encoded]
        base=self.pr.tokenizer.encode(text+selector,add_special_tokens=False)
        assert base==self.pr.tokenizer.encode(text,add_special_tokens=False)+self.pr.tokenizer.encode(selector,add_special_tokens=False)
        for event,_,canonical in encoded:assert self.pr.tokenizer.encode(text+selector+canonical,add_special_tokens=False)==base+event
        base_n=cache.get_seq_length();sel=self.pr.tokenizer.encode(selector,add_special_tokens=False)
        first=self.lognext(self.suffix(cache,sel,p['delta'])[-1:])[0];n=cache.get_seq_length()
        nodes=sorted({tuple(event[:k]) for event in events for k in range(1,len(event))},key=lambda p:(len(p),p))
        index={path:i for i,path in enumerate(nodes)};node_lp=[]
        for start in range(0,len(nodes),32):
            batch=nodes[start:start+32];end=start+len(batch)
            att=torch.full((len(batch),n+end),-torch.inf,device='cuda');att[:,:n]=0
            for j,path in enumerate(batch):
                for k in range(1,len(path)+1):att[j,n+index[path[:k]]]=0
            offsets=torch.tensor([n+len(v)-1 for v in batch],device='cuda')
            lp=self.lognext(self.suffix(cache,[v[-1] for v in batch],p['delta'],att[None,None],offsets));node_lp.extend(lp)
            self.budget.check()
        segments=[];token_lps=[]
        for event,lengths,_ in encoded:
            values=torch.stack([(first if j==0 else node_lp[index[tuple(event[:j])]])[self.columns[t]] for j,t in enumerate(event)])
            token_lps.append(values);offset=0;parts=[]
            for length in lengths:parts.append(values[offset:offset+length].sum());offset+=length
            assert offset==len(event);segments.append(torch.stack(parts))
        cache.crop(base_n)
        return dict(segments=torch.stack(segments),raw=torch.stack([v.sum() for v in token_lps]),events=events,token_logprobs=token_lps,
            canonical=[v[2] for v in encoded],segment_lengths=[v[1] for v in encoded],prefix_tokens=n,trie_nodes=len(nodes),unconstrained_LM=True,grammar_mass_normalized=False)
    @torch.inference_mode()
    def independent_box(self,p,selector,box):
        event,_,_=self.encode_box(box);sid=self.pr.tokenizer.encode(selector,add_special_tokens=False)
        cache,h=self.prefill(p,sid+event,chunk=191);lp=self.lognext(h[0,-len(event)-1:-1].cuda())
        val=float(lp.gather(-1,torch.tensor([self.columns[v] for v in event])[:,None]).sum())
        del cache,h;gc.collect();torch.cuda.empty_cache();return val
