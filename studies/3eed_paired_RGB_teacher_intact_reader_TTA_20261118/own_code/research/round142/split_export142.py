"""Transport-only split of large scene table; all row values/order preserved."""
from common142 import *
import csv,io,collections,re,shutil
def main():
 check();O=H/'publication/public_export_attempt002';p=O/'results/ALL_PHASE_SCENE_ROWS.csv';text=p.read_text();reader=csv.DictReader(io.StringIO(text));cols=reader.fieldnames;rows=list(reader);phases=list(dict.fromkeys(r['phase'] for r in rows));files=[];reassembled=[]
 for phase in phases:
  rr=[r for r in rows if r['phase']==phase];s=io.StringIO();w=csv.DictWriter(s,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows(rr);q=O/'results'/('SCENE_'+phase.upper()+'.csv');q.write_text(s.getvalue());assert list(csv.DictReader(io.StringIO(q.read_text())))==rr;reassembled+=rr;files.append({'path':str(q.relative_to(O)),'rows':len(rr),'bytes':q.stat().st_size})
 assert reassembled==rows and all(f['bytes']<500000 for f in files);backup=H/'publication/UNSPLIT_SCENE_TABLE_DRAFT.csv';assert not backup.exists();shutil.move(str(p),str(backup));q=O/'own_code/research/round142/split_export142.py';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(Path(__file__).read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'));(O/'REPRODUCTION.md').open('a').write('\nTransport: afterpackage142, runsplit_export142. The large all-phase sceneCSV is split byphase intoSCENE_IMAGE,SCENE_TTA,SCENE_SUPPLEMENTARY_MEAN CSVs, preservingallrow values/order; no prediction ormetric changes.\n');tokens=set()
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  for obj in [e['row'],e['A_sensor']]+([e['pair_PRIVATE']['B']] if e['has_B'] else []):
   for k in ['id','caption','scene','imagepath','pointpath','metapath']:
    v=obj.get(k)
    if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['entry']['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
 for f in O.rglob('*'):
  if f.is_file():
   t=f.read_text();assert 'LOCAL_HOME/' not in t and not any(x in t for x in tokens),f;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',t)
 receipt={'status':'PASS','original_rows':len(rows),'split_files':files,'all_row_values_and_order_unchanged':True,'original_unsplit_draft_preserved_locally':True,'not_a_new_experiment_or_task_score':True};dump(O/'TRANSPORT_TABLE_SPLIT.json',receipt);a=rd(H/'PUBLIC_EXPORT_AUDIT.json');a.update(transport_table_split=receipt,private_literals_scanned=len(tokens),rescan_after_split='PASS');dump(H/'PUBLIC_EXPORT_AUDIT.json',a);m=O/'MANIFEST.json';dump(m,{'prefix':'studies/3eed_paired_RGB_teacher_intact_reader_TTA_20261118','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file() and f!=m]});print(receipt,flush=True)
if __name__=='__main__':main()
