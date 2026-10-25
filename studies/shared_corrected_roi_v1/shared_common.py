"""Immutable parent readers with independent selected-method registration."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'studies/no_shrink_main_v1'))
import ns_common as parent
from ns_common import *
S=Path(__file__).parent;D=R/'outputs/shared_corrected_roi_v1';P=R/'results/shared_corrected_roi_v1'
sys.path.insert(0,str(S))
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 parent.validate();check();return p
def arms(task):return ['Source','SourceJointMAP','EightReplacement','SharedCorrection','SpatialInference','HistoricalAuxDP','FullMAP','FinalIndependent']
def pairs(task):return [('EightReplacement','Source'),('SharedCorrection','EightReplacement'),('SpatialInference','SharedCorrection'),('FullMAP','SpatialInference'),('FullMAP','Source'),('FullMAP','SourceJointMAP'),('FullMAP','FinalIndependent'),('FullMAP','HistoricalAuxDP')]
def rows(task,ds):return [dict(row=r,key=r['id'])for r in records(ds)]
