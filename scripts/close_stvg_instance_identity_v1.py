"""Finite closure after actual root, visual, anonymous publication and archive verification."""
import sys,json,shutil,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,read,write,sha

def main():
 names=['ROOT_ACTUAL_UPDATE_READBACK.json','ROOT_RAW_IOU_READBACK.json','ROOT_INDEPENDENT_GEOMETRY_READBACK.json','ROOT_POSTSEAL_READBACK.json','ROOT_VISUAL_REVIEW.json','ROOT_FINAL_PUBLIC_AUDIT.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json','PUBLIC_REF_READBACK.json','ARCHIVE_VERIFICATION_RECEIPT.json']
 for n in names:
  if read(BASE/n)['status']!='pass':raise RuntimeError('not verified '+n)
 p=read(BASE/'FINAL_GITHUB_RECEIPT.json');refs=read(BASE/'PUBLIC_REF_READBACK.json')
 if not p['all_remote_bytes_SHA256_Gitblob_mode_tree_exact']or set(refs['refs'].values())!={p['remote_commit']}:raise RuntimeError('remote incomplete')
 if sha(ROOT/'methods/CURRENT_METHOD.json')!=read(BASE/'DESIGN_LOCK.json')['CURRENT_sha256']:raise RuntimeError('CURRENT changed')
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 write(BASE/'ROOT_CLOSING.json',dict(status='complete',scope='8 historically exposed sources, WHO/WHEN separation, saved-step diagnosis, loss/rejection controls and matched geometry readout',time=stamp,case_count=8,actual_new_fit_paths=16,actual_accepted_updates=68,actual_proposed_updates=69,actual_full_model_redecodes=16,score_rows=112,summary_cells=140,paired_contrasts=120,pages_viewed=2,complete_current_finite_scope=True,independent32_or_full_data_launched=False,CURRENT_changed=False,root_receipts={n:sha(BASE/n)for n in names},public_measurement_commit=p['remote_commit'],public_measurement_tree=p['remote_tree']))
 write(BASE/'FINAL_COMPLETION.json',dict(status='complete',scope='current finite identity/update diagnostic only',complete_current_finite_scope=True,root_closing_sha256=sha(BASE/'ROOT_CLOSING.json'),production_validated=False,independent_confirmation_completed=False,future_research_complete=False,time=stamp))
 out=ROOT/'results/stvg_instance_identity/2026-10-14/closing';out.mkdir(parents=True,exist_ok=True);paths=[]
 for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ROOT_FINAL_PUBLIC_AUDIT.json','PUBLIC_SCALAR_AUDIT.json','FINAL_GITHUB_RECEIPT.json','PUBLIC_REF_READBACK.json','ARCHIVE_VERIFICATION_RECEIPT.json','SAVED_GRADIENT_DIAGNOSTIC.json','ROOT_DECODER_DEPENDENCY_READBACK.json']:
  p=out/n;shutil.copyfile(BASE/n,p);paths.append(p)
 paths +=[ROOT/'scripts/close_stvg_instance_identity_v1.py',ROOT/'scripts/review_stvg_instance_saved_gradients_v1.py',ROOT/'docs/STVG_INSTANCE_IDENTITY_GRADIENT_APPENDIX.md']
 stage=BASE/'closing_public_stage';stage.mkdir(exist_ok=False);manifest=[]
 for p in paths:
  if p.suffix=='.json' and any(x in p.read_text()for x in ['/home/wwww/','physical_frame_ids','query_text','raw_output','source_case_path']):raise RuntimeError('private marker')
  dest=stage/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,encoding='utf-8'))
 write(BASE/'CLOSING_PUBLIC_MANIFEST.json',manifest);print('closing staged',len(manifest))
if __name__=='__main__':main()
