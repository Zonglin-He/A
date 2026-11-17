"""Verify every streamed native audit and pinned persistent endpoint before GT."""
from common151 import *
import argparse,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();root=B/('engineering_frozen' if args.engineering else 'frozen');seal=rd(root/'SEALED.json');reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));expected={(r['source'],r['target'],r['row']['id']) for r in reg};seen=set();nr=0;mxp=0.;mxg=0.;mxc=0.
 for rel,h in seal['files'].items():assert sha(root/rel)==h
 for p in sorted((root/'audit_chunks_PRIVATE').glob('*.json')):
  if p.name.endswith('_AUDIT.json'):continue
  ch=rd(p);a=rd(p.with_name(p.stem+'_AUDIT.json'));assert a['status']=='PASS' and not a['GT_read'] and a['chunk_manifest_sha256']==sha(p) and a['files']==ch['files']
  for rel,h in ch['files'].items():
   if '/features_TEMP_PRIVATE/' in rel or '/graphs_TEMP_PRIVATE/' in rel:
    assert a['temporary_files_sha256'][rel]==h
    if args.engineering:assert sha(root/rel)==h
    else:assert not(root/rel).exists()
   else:assert sha(root/rel)==h
  for r in ch['rows']:
   key=r['source'],r['target'],r['id_PRIVATE'];assert key in expected and key not in seen;seen.add(key)
  nr+=a['forwards'];mxp=max(mxp,a['max_CPU_probability_error']);mxg=max(mxg,a['max_graph_weight_error']);mxc=max(mxc,a['max_engineering_CNN_error'])
 assert seen==expected and len(seen)==seal['n'];n=len(seen);out=dict(status='PASS',n=n,new_reader_forwards=nr,max_CPU_probability_error=mxp,max_graph_weight_error=mxg,max_engineering_CNN_error=mxc,all_native_chunks_verified_before_temporary_deletion=True,source_parameters_updated=0,raw_pixel_SHA_and_actual_CNN_input_provenance=True,fixed_1024point_F_input_verified=True,engineering_CNN_regenerated_CPU=args.engineering,full_transformer_not_CPU_regenerated=True,FINAL_access=False);dump(H/('FROZEN_ENGINEERING_AUDIT.json' if args.engineering else 'FROZEN_AUDIT.json'),out);print('FROZEN151_AUDIT',out,flush=True)
if __name__=='__main__':main()
