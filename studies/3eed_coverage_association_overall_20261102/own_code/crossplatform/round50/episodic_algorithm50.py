"""Pure current-input correspondence/readout and native matching-head losses.

No sample/label/model loader, hidden cache search, optimizer or new objective.
Geometry uses the exact existing R18 grouping/Hungarian implementation.
"""
from pathlib import Path
import importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
_spec=importlib.util.spec_from_file_location('R50_existing_R18_geometry',ROOT/'crossplatform/round18/geometry.py')
_geom=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_geom)

def group_values(score,groups):return np.array([np.max(np.asarray(score)[g]) for g in groups])
def pick_group(score,reps):return int(max(range(len(reps)),key=lambda j:(float(score[j]),-int(reps[j]))))
def native_head(head,h,tokens,mask):
 """Exact batch-1 source interface, avoiding an unregistered 2D GEMM path."""
 import torch
 F=torch.nn.functional
 assert h.ndim==tokens.ndim==2 and h.shape==(256,288) and tokens.shape[1]==64 and mask.shape==(len(tokens),)
 assert mask.any() and (~mask).any()
 q=F.normalize(head(h[None]),dim=-1);logits=q@tokens[None].transpose(-1,-2)/.07
 p=(logits.softmax(-1)*mask[None,None]).sum(-1)[0]
 z=(torch.logsumexp(logits.masked_fill(~mask[None,None],-torch.inf),-1)-torch.logsumexp(logits.masked_fill(mask[None,None],-torch.inf),-1))[0]
 assert torch.isfinite(p).all() and torch.isfinite(z).all(), 'STOP_nonfinite_matching_readout'
 return p,z
def original_ranknet(zA,zB,positiveA,negativeA,positiveB,negativeB):
 import torch
 ga=zA[positiveA].max()-zA[negativeA].max();gb=zB[positiveB].max()-zB[negativeB].max()
 return torch.nn.functional.softplus(-torch.stack([ga,gb])/.1).mean(),torch.stack([ga,gb])

def unique_mask(valid,q):
 ix=np.flatnonzero(np.asarray(valid,bool)[int(q)])
 return int(ix[0]) if len(ix)==1 else None

def frozen_structure(boxesA,scoresA,boxesB,scoresB,validA,validB,FA,FB):
 """Source geometry reps and frozen canonical-G output reps are distinct.

 Grouping, Hungarian and the identity gate use original source reps only.
 Output reps retain each A group's actual G-chosen box, frozen before update.
 """
 gA,rA=_geom.groups(boxesA,scoresA);gB,rB=_geom.groups(boxesB,scoresB)
 rows,cols,diag=_geom.match_boxes(np.asarray(boxesA)[rA],np.asarray(boxesB)[rB]);mapping={int(a):int(b) for a,b in zip(rows,cols)}
 membershipA=np.empty(256,np.int64);membershipB=np.empty(256,np.int64)
 for j,g in enumerate(gA):membershipA[g]=j
 for j,g in enumerate(gB):membershipB[g]=j
 q0=int(np.asarray(scoresA).argmax());qG=int(np.asarray(FA).argmax());sourcegroup=int(membershipA[q0]);positive=int(membershipA[qG])
 ma=unique_mask(validA,rA[sourcegroup]);mp=unique_mask(validA,rA[positive]);fga=group_values(FA,gA);gapA=float(fga[positive]-fga[sourcegroup])
 pre=bool(positive!=sourcegroup and ma is not None and mp is not None and ma!=mp and gapA>0)
 reason='A_EXACT_TIE_IS_NOT_A_STRICT_PREFERENCE' if positive!=sourcegroup and gapA==0 else 'NO_DISTINCT_UNIQUE_A_INSTANCE_PREFERENCE';accepted=False
 if pre:
  reason='UNKNOWN_A_GROUP_NOT_MATCHED_IN_B'
  if sourcegroup in mapping and positive in mapping:
   nb=mapping[sourcegroup];pb=mapping[positive];mnb=unique_mask(validB,rB[nb]);mpb=unique_mask(validB,rB[pb])
   reason='UNKNOWN_B_MASK_IDENTITY_MISMATCH'
   if mnb==ma and mpb==mp:
    fgB=group_values(FB,gB);winner=pick_group(fgB,rB)
    # The accepted statement is only P>N. A third B candidate may legitimately
    # win the global readout; it is never silently appended as a negative.
    accepted=bool(fgB[pb]>fgB[nb]);reason='ACCEPTED_LOCAL_TWO_VIEW_PREFERENCE' if accepted else 'G_B_DOES_NOT_SUPPORT_SAME_LOCAL_DIRECTION'
 outputA=np.array([max(g,key=lambda q:(float(np.asarray(FA)[q]),-int(q))) for g in gA],np.int64)
 rec={'groupsA':gA,'groupsB':gB,'repsA':outputA,'geometry_repsA':rA,'repsB':rB,'membershipA':membershipA,'membershipB':membershipB,
  'match_A':rows,'match_B':cols,'mapping':mapping,'matching_diagnostics':diag,'q0':q0,'qG':qG,
  'source_group':sourcegroup,'positive_group':positive,'pre_A_condition':pre,'accepted':accepted,'reason':reason,
  'strict_A_group_gap':gapA,
  'q0_group_has_B_mapping':sourcegroup in mapping,'mask_source_A':ma,'mask_positive_A':mp}
 return rec

def consensus(scoresA,scoresB,deltaA,deltaB,structure):
 """Same operator for 0/1/3 steps, same G increments, canonical A boxes only.

 Unmatched A groups retain their A score. No B geometry becomes an output box.
 A-only fallback is the literal currentG query, not a substitute group box.
 All learned-state outputs retain the same original canonical-G A reps.
 """
 def with_G(score,components):
  c=np.asarray(components,dtype=np.float64);assert c.shape==(3,256)
  # Same literal addition order as the already frozen legacy operator.
  return np.asarray(score,dtype=np.float64)+2*c[0]+2*c[1]+2*c[2]
 FA=with_G(scoresA,deltaA);FB=with_G(scoresB,deltaB)
 if not structure['pre_A_condition'] or not structure['q0_group_has_B_mapping']:
  return dict(query=int(FA.argmax()),scores=FA.copy(),group_scores=group_values(FA,structure['groupsA']),
   query_override=False,readout='LITERAL_A_G_FALLBACK',unmatched_A_group_n=len(structure['groupsA'])-len(structure['mapping']))
 ga=group_values(FA,structure['groupsA']);gb=group_values(FB,structure['groupsB']);q0g=structure['source_group'];q0b=structure['mapping'][q0g]
 da=ga-ga[q0g];db=gb-gb[q0b];c=da.copy()
 for ag,bg in structure['mapping'].items():c[ag]=(da[ag]+db[bg])/2
 winner=pick_group(c,structure['repsA']);q=int(structure['repsA'][winner]);lift=c[structure['membershipA']]
 return dict(query=q,scores=lift,group_scores=c,query_override=True,readout='MATCHED_TWO_VIEW_RELATIVE_GROUP_CONSENSUS_FIXED_A_REP',unmatched_A_group_n=len(ga)-len(structure['mapping']))
