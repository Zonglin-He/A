"""Read audited payload metadata only; distinguish conversions from native empty masks."""
from pathlib import Path
import json,csv,hashlib
import torch
R=Path.cwd();D=R/'outputs/expert_subset_expansion_v1';P=R/'results/expert_subset_expansion_v1';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);records=[];counts={};selection=read(D/'selection.json')
 for expert in ['referformer_r101','mttr']:
  for ds in ['hc2','vidstg']:
   seal=read(D/expert/ds/'sealed.json');assert list(seal['files'])==[r['id']for r in selection[ds]];rows=[]
   for query,row in enumerate(selection[ds],1):
    e=seal['files'][row['id']];m=e['expert_payload'];assert sha(D/m['path'])==m['sha256'];payload=torch.load(D/m['path'],map_location='cpu',weights_only=False,mmap=True);shape=payload['input_shape'];assert shape[:2]==[8,3]
    r=dict(expert=expert,dataset=ds,query=query,frames=shape[0],channels=shape[1],input_height=shape[2],input_width=shape[3],selected_query=payload['selected_query'],empty_mask_conversion_to_full_frame_box=e['empty_mask_to_full_frame_boxes'],native_empty_mask_count=(e['empty_mask_to_full_frame_boxes']if expert=='mttr'else None),expert_payload_sha256=m['sha256']);records.append(r);rows.append(r)
   counts[expert+'/'+ds]=dict(queries=128,observations=1024,mask_to_box_adapter=expert=='mttr',full_frame_box_conversion_observations=sum(x['empty_mask_conversion_to_full_frame_box']for x in rows),queries_any_conversion=sum(x['empty_mask_conversion_to_full_frame_box']>0 for x in rows),queries_all8_converted=sum(x['empty_mask_conversion_to_full_frame_box']==8 for x in rows),native_empty_mask_count=(sum(x['native_empty_mask_count']for x in rows)if expert=='mttr'else None),R101_note='R101 has a direct box head: conversion0 does NOT claim its native masks were never empty'if expert=='referformer_r101'else'Official thresholded selected-track empty masks counted explicitly',input_sizes=sorted({(x['input_height'],x['input_width'])for x in rows}))
 with(P/'actual512_input_shapes_and_conversion_counts.tsv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(records)
 (P/'mask_to_box_interface_definition_v3.json').write_text(json.dumps(dict(authoritative_count_semantics=counts,earlier_empty_observations_field_in_cost_export_is_conversion_count_not_R101_native_mask_empty_measurement=True,new_model=0,new_Source=0,new_expert=0,new_GT=0),indent=2)+'\n')
 (D/'root_input_shape_supplement_receipt_v1.json').write_text(json.dumps(dict(status='all512_audited_input_sizes_and_conversion_semantics_readback',calls=512,shapes_sha256=sha(P/'actual512_input_shapes_and_conversion_counts.tsv'),definitions_sha256=sha(P/'mask_to_box_interface_definition_v3.json'),code_sha256=sha(Path(__file__)),new_model=0,new_Source=0,new_expert=0,new_GT=0),indent=2)+'\n');print('all512 actual shapes/conversion metadata verified; no inference/GT')
if __name__=='__main__':main()
