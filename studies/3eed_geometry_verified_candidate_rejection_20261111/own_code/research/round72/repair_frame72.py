"""Pre-GT protocol repair from the official annotation contract, no gate sweep."""
from common72 import *
from text72 import parse,REL_RE,ALLREL
from constraints72 import build
from scipy.special import softmax,logsumexp
import time

def main():
 p=read(H/'experiment/FRAME_REPAIR_LOCK.json');pin(p);out=ROOT/p['evidence_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','cases_PRIVATE']:(out/f).mkdir()
 original=ROOT/read(H/'experiment/DESIGN_LOCK.json')['evidence_root'];ss=read(original/'SEALED.json');counts={'n':0,'active_rejection_inputs':0,'supported_inputs':0,'literal_relations':0,'camera_contract_simple_relations':0,'reference_color_parsed':0,'reference_witness_available':0,'relation_active_inputs':0,'source_top1_rejected':0};start=time.perf_counter()
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(original/rel)==ss['files'][rel];z=load(original/rel);a=prefix(r,('boxes','scores','tokens','query'));t=parse(r['caption'],len(a['tokens']));edge=REL_RE.search(r['caption']);first=ALLREL.search(r['caption']);simple=edge is not None and first is not None and edge.start()==first.start();t['relation']=edge.group(1).lower() if simple and t['ref_mask'].any() else None
  # Complex/chained relationship syntax remains UNKNOWN: no implicit role reversal.
  tm={k[7:]:v for k,v in z.items() if k.startswith('target_')};rm={k[10:]:v for k,v in z.items() if k.startswith('reference_') and k!='reference_prob'};qref=z['reference_prob'] if t['ref_mask'].any() else None
  N,rec=build(z,a['boxes'],tm,rm,t,qref,sid);z.update(N);np.savez_compressed(out/rel,**z)
  case=read(original/'cases_PRIVATE'/(sid+'.json'));case.update(reference_record=rec,camera_contract_simple_relation=t['relation'] is not None,main_rejected_groups=int(N['main_N'].sum()),source_rejected_mass=float(a['scores'][N['main_N'][z['query_group']]].sum()),source_top1_rejected=bool(N['main_N'][z['query_group'][a['scores'].argmax()]]));dump(out/'cases_PRIVATE'/(sid+'.json'),case)
  counts['n']+=1;counts['active_rejection_inputs']+=N['main_N'].any();counts['supported_inputs']+=tm['known'].any();counts['literal_relations']+=t['has_relation_literal'];counts['camera_contract_simple_relations']+=t['relation'] is not None;counts['reference_color_parsed']+=t['reference']['color'] is not None;counts['reference_witness_available']+=rec['anchor_groups']>0;counts['relation_active_inputs']+=N['relation_N'].any();counts['source_top1_rejected']+=case['source_top1_rejected']
 counts={k:int(v) for k,v in counts.items()};counts.update(seconds=time.perf_counter()-start,new_raw_media_reads=0,new_GT_lookups=0);dump(out/'COUNTS.json',counts);seal(out,'R72_OFFICIAL_CAMERA_CONTRACT_REPAIR_BEFORE_TASK_LOOKUP',n=2640);print(counts)
if __name__=='__main__':main()
