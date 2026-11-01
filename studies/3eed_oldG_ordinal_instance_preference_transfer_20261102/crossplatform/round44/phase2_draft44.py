"""DRAFT ONLY: loss/readout/probability-memory algebra; no execution entry point.

No teacher-posterior KL. All accepted Phase1 preferences must be kept unchanged.
Runtime driver and GPU execution require a separate root phase2 authorization.
"""
import numpy as np

def exact_z_from_logits(token_logits,positive_mask,padding_mask):
 import torch
 positive=positive_mask&~padding_mask;negative=~positive_mask&~padding_mask
 assert positive.any(-1).all() and negative.any(-1).all()
 return torch.logsumexp(token_logits.masked_fill(~positive[:,None],-torch.inf),-1)-torch.logsumexp(token_logits.masked_fill(~negative[:,None],-torch.inf),-1)

def proposed_loss(z,z0,frozen_q0,prefs,keep_rows,expected_denominator):
 import torch
 F=torch.nn.functional
 assert len(prefs)==expected_denominator and len(keep_rows)==104-len(prefs)
 gaps=torch.stack([z[f['row_index'],f['positive_queries']].max()-z[f['row_index'],f['negative_queries']].max() for f in prefs])
 # Same RankNet + relative source keep as R30; mean's denominator is fixed14/18.
 pref=F.softplus(-gaps/.1).sum()/expected_denominator
 d=z-z.gather(1,frozen_q0[:,None]);d0=z0-z0.gather(1,frozen_q0[:,None])
 keep=(d[keep_rows]-d0[keep_rows]).square().mean()
 return pref+keep,{'preference':pref,'relative_z_keep':keep,'gaps':gaps}

def fixed_source_group_readout(native_score,source_probability,full_groups):
 used=set();options=[]
 for group in full_groups.values():
  if not group:continue
  used.update(group);rep=max(group,key=lambda q:(float(source_probability[q]),-q));options.append((float(np.max(native_score[group])),-rep,rep))
 for q in range(256):
  if q not in used:options.append((float(native_score[q]),-q,q))
 return int(max(options)[2])

def original_probability_memory_values(source_probabilities,prefs):
 """Same R30 native-p delta; NEVER add this probability delta to token z."""
 values=np.zeros_like(source_probabilities,dtype=np.float32);count=np.zeros_like(values)
 for f in prefs:
  j=f['row_index'];P=f['positive_queries'];N=f['negative_queries'];sp=source_probabilities[j]
  delta=max(.1-float(sp[P].max()-sp[N].max()),0.)/2
  values[j,P]+=delta;values[j,N]-=delta;count[j,P]+=1;count[j,N]+=1
 values/=np.maximum(count,1)
 return values

def retrieve_probability_memory(B_keys,A_keys,values,A_library_indices):
 """Literal old-R30 CPU32 query-per-row then top4/exp kernel, fixed no scan.

 Inputs are frozen query*x-normalizednoun keys after row/query L2 normalization.
 U rows have zero values in allA library. E-only is an explicitly separate control.
 Returns all actual CPU32 neighbor indices, similarities and weights for audit.
 """
 import torch
 K=torch.tensor(np.asarray(A_keys[A_library_indices],np.float32),device='cpu');V=torch.tensor(np.asarray(values[A_library_indices],np.float32),device='cpu');q=torch.tensor(np.asarray(B_keys,np.float32),device='cpu')
 sim=torch.einsum('qc,anc->qan',q,K);near,which=sim.max(-1);best,obs=near.topk(min(4,len(A_library_indices)),dim=-1);pos=which.gather(1,obs);weights=((best-1)/.1).exp();corr=(V[obs,pos]*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)
 return corr.numpy(),{'A_row_indices':np.asarray(A_library_indices)[obs.numpy()],'A_query_indices':pos.numpy(),'cosine':best.numpy(),'weights':weights.numpy(),'topk_tie_contract':'literal torchCPU32 max/topk4, actual indices saved, no float64-neighbor replacement'}

if __name__=='__main__':
 raise SystemExit('DRAFT ONLY: root phase2 authorization and actual driver lock required')
