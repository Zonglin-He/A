"""Root-approved anonymous finite-phase export with byte-exact scientific source pins."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'artifacts/stvg_spatial_causal_decomposition_v1';O=R/'results/stvg_spatial_causal_decomposition/2026-10-14'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 rt=json.loads((B/'phase1/RUNTIME.json').read_text());paths={'scripts/'+n+'.py'for n in ['run_stvg_spatial_update_amplitude_v1','launch_stvg_spatial_amplitude_v1','lock_stvg_spatial_amplitude_v1','review_stvg_spatial_amplitude_v1','review_stvg_spatial_oracle_custody_v1','diagnose_stvg_spatial_causal_pool_v1','report_stvg_spatial_causal_phase1_v1','prepare_stvg_spatial_causal_public_v1','stvg_spatial_causal_public_transfer_v1']};paths|={'protocols/stvg_spatial_causal_decomposition_v1.md','docs/STVG_SPATIAL_CAUSAL_DECOMPOSITION_V1.md','output/pdf/stvg_spatial_causal_phase1_v1.pdf'};bindings=[]
 for rel,h in {**rt['pins'],**rt['protected']}.items():
  assert sha(R/rel)==h,rel;public=rel.startswith(('scripts/','vg_tta/','methods/','protocols/'))and rel!='methods/CURRENT_METHOD.json'
  if public:paths.add(rel)
  bindings.append(dict(source=rel if public else'private_or_protected_binding',sha256=h,published=public))
 save(O/'runtime_source_bindings.json',dict(status='pass',bindings=bindings,phase1_runtime_sha256=sha(B/'phase1/RUNTIME.json'),phase1_lock_sha256=sha(B/'phase1/PHASE1_LOCK.json')))
 for src,name in [(B/'ROOT_VISUAL_REVIEW.json','root_visual_review.json'),(B/'ROOT_POOL_COMMON_SUPPORT_READBACK.json','root_pool_readback.json'),(B/'phase1/CPU_AMPLITUDE_CONTRACTS.json','cpu_contracts.json'),(B/'recovery/post_freeze_pin_constant001/CAPTURE_RESOLUTION.json','pin_constant_recovery.json'),(B/'recovery/root_reader_completeness002/RESOLUTION.json','root_reader_recovery.json')]:save(O/name,json.loads(src.read_text()))
 paths|={str(p.relative_to(R))for p in O.glob('*.json')};rows=[]
 for rel in sorted(paths):
  assert not any(x in rel for x in ['source_cases','private_inputs','private_views','.pt','.safetensors','attachments','conversation']);p=R/rel;data=p.read_bytes();target=B/'public_stage'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);rows.append(dict(path=rel,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64'if p.suffix=='.pdf'else'utf-8'))
 save(B/'PUBLIC_MANIFEST.json',rows);save(B/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',files=len(rows),bytes=sum(x['bytes']for x in rows),source_pin_count=len(bindings)));print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes']for x in rows))))
if __name__=='__main__':main()
