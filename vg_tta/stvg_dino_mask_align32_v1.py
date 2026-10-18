"""High-content conditional DINO: paired exclusion masks, saved six-layer traces."""
import time
import numpy as np
import torch
from torchvision.ops import box_iou,box_convert
from vg_tta.stvg_dino_content_match32_v1 import ContentDecoder,digest
from vg_tta.stvg_dino_conditional_opd32_v1 import legal
from vg_tta.stvg_error_state_correction_v1 import encode,decode


class MaskDecoder(ContentDecoder):
    @torch.no_grad()
    def readout(self,out,initial):
        bb=[];ss=[]
        for l in range(out.intermediate_hidden_states.shape[1]):
            h=out.intermediate_hidden_states[:,l]
            ref=initial if l==0 else out.intermediate_reference_points[:,l-1]
            bb.append((self.model.bbox_embed[l](h)+torch.special.logit(ref,eps=1e-5)).sigmoid())
            ss.append(self.model.class_embed[l](h,self.kw['text_encoder_hidden_states'],self.textmask).sigmoid()[:,:,self.entity_tokens].mean(-1))
        return torch.stack(bb,1),torch.stack(ss,1)

    @torch.no_grad()
    def native_trace(self):
        bb,ss=self.readout(self.original,self.kw['reference_points'])
        assert torch.equal(bb[:,-1],self.original.pred_boxes)
        assert torch.equal(ss[:,-1],self.original.logits.sigmoid()[:,:,self.entity_tokens].mean(-1))
        high=int(ss[0,-1].argmax())
        return dict(initial_references=self.kw['reference_points'][0].cpu().numpy(),layer_boxes=bb[0].cpu().numpy(),
            layer_scores=ss[0].cpu().numpy(),layer_references=self.original.intermediate_reference_points[0].cpu().numpy(),
            high_slot=high,high_layer_hidden=self.original.intermediate_hidden_states[0,:,high].cpu().numpy())

    @torch.no_grad()
    def run_trace(self,boxes,aligned=False,content_slots=None,mask_slots=None):
        b=torch.as_tensor(boxes,dtype=torch.float32,device='cuda').reshape(1,-1,4);k=b.shape[1]
        assert k and legal(b[0].cpu().numpy()).all()
        initial=self.kw['reference_points'];nearest=box_iou(box_convert(b[0],'cxcywh','xyxy'),box_convert(initial[0],'cxcywh','xyxy')).argmax(-1)
        native_score=self.original.logits[0].sigmoid()[:,self.entity_tokens].mean(-1);high=int(native_score.argmax())
        cs=torch.full_like(nearest,high)if content_slots is None else torch.as_tensor(content_slots,device=b.device,dtype=torch.long)
        ms=(cs if aligned else nearest)if mask_slots is None else torch.as_tensor(mask_slots,device=b.device,dtype=torch.long)
        kw=dict(self.kw);kw['inputs_embeds']=torch.cat([kw['inputs_embeds'],kw['inputs_embeds'][:,cs]],1);kw['reference_points']=torch.cat([initial,b],1)
        n=self.N+k;mask=b.new_full((1,1,n,n),torch.finfo(b.dtype).min);mask[:,:,:self.N,:self.N]=0
        for j,slot in enumerate(ms.tolist()):
            mask[:,:,self.N+j,:self.N]=0;mask[:,:,self.N+j,slot]=torch.finfo(b.dtype).min;mask[:,:,self.N+j,self.N+j]=0
        kw['self_attn_mask']=mask
        invariant=dict(reference_sha256=digest(kw['reference_points']),content_sha256=digest(kw['inputs_embeds']),
            other_tensors={key:digest(v)for key,v in kw.items()if torch.is_tensor(v)and key not in ['inputs_embeds','reference_points','self_attn_mask']})
        old=self.model.model.decoder.config.num_queries;torch.cuda.synchronize();t=time.perf_counter()
        try:
            self.model.model.decoder.config.num_queries=n;out=self.model.model.decoder(**kw)
        finally:self.model.model.decoder.config.num_queries=old
        bb,ss=self.readout(out,kw['reference_points']);torch.cuda.synchronize();self.decoder_seconds+=time.perf_counter()-t;self.decoder_calls+=1
        native_err=float((bb[0,-1,:self.N]-self.original.pred_boxes[0]).abs().max());assert native_err<1e-4
        return dict(boxes=bb[0,:,-k:].cpu().numpy(),scores=ss[0,:,-k:].cpu().numpy(),
            references=out.intermediate_reference_points[0,:,-k:].cpu().numpy(),hidden=out.intermediate_hidden_states[0,:,-k:].cpu().numpy(),
            initial_reference=b[0].cpu().numpy(),content_slots=cs.cpu().numpy(),mask_slots=ms.cpu().numpy(),nearest_slots=nearest.cpu().numpy(),
            invariant=invariant,mask_sha256=digest(mask),native_context_max_error=native_err)

    @torch.no_grad()
    def qualify_mask(self):
        native=self.native_trace();bb,ss=self.readout(self.model.model.decoder(**self.kw),self.kw['reference_points'])
        replay_box=float((bb[0]-torch.as_tensor(native['layer_boxes'],device='cuda')).abs().max());replay_score=float((ss[0]-torch.as_tensor(native['layer_scores'],device='cuda')).abs().max())
        slots=np.array([native['high_slot'],self.N//2,self.N-1]);refs=native['initial_references'][slots]
        clone=self.run_trace(refs,True,content_slots=slots,mask_slots=slots)
        boxerr=float(abs(clone['boxes']-native['layer_boxes'][:,slots]).max());scoreerr=float(abs(clone['scores']-native['layer_scores'][:,slots]).max())
        assert max(replay_box,replay_score)==0
        assert max(boxerr,scoreerr)<1e-4,(boxerr,scoreerr)
        return dict(status='pass',layers=6,clone_slots=slots.tolist(),native_replay_box_error=replay_box,native_replay_score_error=replay_score,
            clone_all_layers_box_error=boxerr,clone_all_layers_score_error=scoreerr,tolerance=1e-4,initial_reference_saved=True)


def feedback(caches,source,actions,scale,positions,paired=True):
    actions=np.asarray(actions,np.float64);scale=np.asarray(scale);z=encode(source)
    boxes=np.stack([decode(z+a*scale)for a in actions]);valid=legal(boxes[:,positions]).all(1)
    modes=['A_unaligned','B_aligned']if paired else ['B_aligned']
    result={m:dict(targets=actions.copy(),details=[dict(legal=bool(ok),anchors={})for ok in valid])for m in modes};traces={}
    for pos,c in zip(positions,caches):
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        unique,inverse=np.unique(boxes[ids,pos],axis=0,return_inverse=True);traces[pos]=dict(candidate_indices=ids,inverse=inverse)
        for m in modes:
            trace=c.run_trace(unique,m=='B_aligned');traces[pos][m]=trace
            for j,ix in enumerate(ids):
                h=inverse[j];v=float(trace['scores'][-1,h]);result[m]['details'][ix]['anchors'][pos]=dict(box=trace['boxes'][-1,h],score=v,accepted=v>=.35,
                    slot=int(trace['content_slots'][h]),mask_slot=int(trace['mask_slots'][h]))
        if paired:
            a,b=traces[pos]['A_unaligned'],traces[pos]['B_aligned'];assert a['invariant']==b['invariant']
            same=a['mask_slots']==b['mask_slots']
            if same.any():assert abs(a['boxes'][:,same]-b['boxes'][:,same]).max()<1e-5 and abs(a['scores'][:,same]-b['scores'][:,same]).max()<1e-5
    for m,r in result.items():
        for i,d in enumerate(r['details']):
            good=[encode(np.array([v['box']]))[0]-z[pos]for pos,v in d['anchors'].items()if v['accepted']]
            if good:r['targets'][i]=np.mean(good,0)/scale
            d['accepted_anchors']=len(good)
    return result,traces


def update(mu,targets,actions,denominator=24):
    assert len(actions)<=denominator and denominator==24
    correction=np.asarray(targets)-np.asarray(actions);new=np.asarray(mu)+correction.sum(0)/denominator
    if not correction.any():assert np.array_equal(new,mu)
    return new,dict(denominator=denominator,candidates=len(actions),mean_before=np.asarray(mu).copy(),mean_after=new.copy(),
        correction_sum=correction.sum(0),step_l2=float(np.linalg.norm(new-mu)),all_identity_exact_noop=not bool(correction.any()))


def contract():
    rng=np.random.default_rng(20261018);mu=rng.normal(size=4);a=mu+rng.normal(size=(24,4));y=a+rng.normal(size=(24,4))*.1
    one,_=update(mu,y,a);m=mu.copy()
    for aa,yy in zip(np.split(a,3),np.split(y,3)):m,_=update(m,yy,aa)
    assert np.allclose(one,m,rtol=0,atol=1e-15)
    for aa in np.split(a,3):
        n,_=update(mu,aa.copy(),aa);assert np.array_equal(n,mu)
    return dict(status='pass',count=4,no_feedback_exact_noop=True,fixed24_equals_three_fixed8_correction_sums=True,dynamic_denominator=24)
