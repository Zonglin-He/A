"""Matched content intervention: references, context and attention support fixed."""
import time,hashlib
import numpy as np
import torch
from torchvision.ops import box_iou,box_convert
from vg_tta.stvg_dino_conditional_opd32_v1 import ConditionalDecoder as Original,legal
from vg_tta.stvg_error_state_correction_v1 import encode,decode


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class ContentDecoder(Original):
    @torch.no_grad()
    def run_content(self,boxes,mode='nearest',mask_slots=None,content_slots=None):
        b=torch.as_tensor(boxes,dtype=torch.float32,device='cuda').reshape(1,-1,4)
        k=b.shape[1];assert k and legal(b[0].cpu().numpy()).all()
        native=self.kw['reference_points']
        nearest=box_iou(box_convert(b[0],'cxcywh','xyxy'),box_convert(native[0],'cxcywh','xyxy')).argmax(-1)
        mask_slots=nearest if mask_slots is None else torch.as_tensor(mask_slots,device=b.device)
        scores=self.original.logits[0].sigmoid()[:,self.entity_tokens].mean(-1)
        high=int(scores.argmax())
        if content_slots is None:content_slots=nearest if mode=='nearest'else torch.full_like(nearest,high)
        content_slots=torch.as_tensor(content_slots,device=b.device,dtype=torch.long)
        kw=dict(self.kw);kw['inputs_embeds']=torch.cat([kw['inputs_embeds'],kw['inputs_embeds'][:,content_slots]],1)
        kw['reference_points']=torch.cat([native,b],1)
        n=self.N+k;mask=b.new_full((1,1,n,n),torch.finfo(b.dtype).min)
        mask[:,:,:self.N,:self.N]=0
        for j,slot in enumerate(mask_slots.tolist()):
            mask[:,:,self.N+j,:self.N]=0;mask[:,:,self.N+j,slot]=torch.finfo(b.dtype).min
            mask[:,:,self.N+j,self.N+j]=0
        kw['self_attn_mask']=mask
        control=dict(reference_sha256=digest(kw['reference_points']),mask_sha256=digest(mask),
                     native_content_sha256=digest(kw['inputs_embeds'][:,:self.N]),
                     other_inputs={key:digest(v)for key,v in kw.items()if torch.is_tensor(v)and key not in ['inputs_embeds','reference_points','self_attn_mask']})
        old=self.model.model.decoder.config.num_queries;torch.cuda.synchronize();tick=time.perf_counter()
        try:
            self.model.model.decoder.config.num_queries=n
            out=self.model.model.decoder(**kw)
        finally:self.model.model.decoder.config.num_queries=old
        hidden=out.intermediate_hidden_states[:,-1];ref=out.intermediate_reference_points[:,-2]
        pred=(self.model.bbox_embed[-1](hidden)+torch.special.logit(ref,eps=1e-5)).sigmoid()
        logits=self.model.class_embed[-1](hidden,self.kw['text_encoder_hidden_states'],self.textmask)
        torch.cuda.synchronize();self.decoder_seconds+=time.perf_counter()-tick;self.decoder_calls+=1
        assert torch.isfinite(pred).all()
        meta=dict(controls=control,content_slots=content_slots.cpu().numpy(),mask_slots=mask_slots.cpu().numpy(),
                  native_top_slot=high,native_top_score=float(scores[high]),
                  chosen_native_scores=scores[content_slots].cpu().numpy())
        return pred[0,self.N:].cpu().numpy(),logits[0,self.N:].sigmoid()[:,self.entity_tokens].mean(-1).cpu().numpy(),meta

    @torch.no_grad()
    def qualify_content(self):
        q=self.qualify();slots=torch.tensor([0,self.N//2,self.N-1],device='cuda')
        b,v,_=self.run_content(self.kw['reference_points'][0,slots],mask_slots=slots,content_slots=slots)
        original=self.original.logits[0,slots].sigmoid()[:,self.entity_tokens].mean(-1).cpu().numpy()
        err=float(abs(v-original).max());assert err<1e-4
        q.update(cloned_native_target_score_max_error=err,content_and_mask_controls=True)
        return q


def matched_targets(caches,source,actions,scale,positions):
    actions=np.asarray(actions,np.float64);scale=np.asarray(scale);z=encode(source)
    boxes=np.stack([decode(z+a*scale)for a in actions]);valid=legal(boxes[:,positions]).all(1)
    result={mode:dict(targets=actions.copy(),details=[dict(legal=bool(ok),anchors={})for ok in valid])for mode in ['nearest','target_high']}
    controls=[]
    for pos,c in zip(positions,caches):
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        unique,inverse=np.unique(boxes[ids,pos],axis=0,return_inverse=True)
        pb,pv,pm=c.run_content(unique,'nearest');qb,qv,qm=c.run_content(unique,'target_high')
        assert pm['controls']==qm['controls'] and np.array_equal(pm['mask_slots'],qm['mask_slots'])
        controls.append(dict(position=pos,matched_controls=pm['controls'],native_top_slot=pm['native_top_slot'],native_top_score=pm['native_top_score']))
        for mode,bb,vv,mm in [('nearest',pb,pv,pm),('target_high',qb,qv,qm)]:
            for j,ix in enumerate(ids):
                h=inverse[j];v=float(vv[h]);result[mode]['details'][ix]['anchors'][pos]=dict(box=bb[h],score=v,accepted=v>=.35,
                    slot=int(mm['content_slots'][h]),mask_slot=int(mm['mask_slots'][h]),native_score=float(mm['chosen_native_scores'][h]))
    for mode,r in result.items():
        for i,d in enumerate(r['details']):
            good=[encode(np.array([v['box']]))[0]-z[pos]for pos,v in d['anchors'].items()if v['accepted']]
            if good:r['targets'][i]=np.mean(good,axis=0)/scale
            d['accepted_anchors']=len(good)
    return result,controls
