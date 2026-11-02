"""Verify worklist pixel bindings with the actual evidence PIL runtime, no model."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    exp=ROOT/'research/round50/experiment';a=json.loads((exp/'WORKLIST_RGB_RUNTIME_CHECK_AUTHORIZATION.json').read_text())
    assert a['state']=='ROOT_AUTHORIZED_R50_WORKLIST_RGB_BINDING_RUNTIME_CHECK_NO_MODELS'
    assert a['program_sha256']==sha(__file__)
    folder=ROOT/'crossplatform/round50/full5912_attempt001/worklists/dino'
    assert sha(folder/'SEALED.json')==a['worklist_seal_sha256']
    seal=json.loads((folder/'SEALED.json').read_text());p=folder/'DINO_CASE_REUSE_INDEX_PRIVATE.json'
    assert sha(p)==seal['files'][p.name]
    manifest=ROOT/'research/round8/experiment/full5912_gpu_inputs.json'
    assert sha(manifest)==a['manifest_sha256']
    rows={r['id']:r for r in json.loads(manifest.read_text())};entries=json.loads(p.read_text())
    assert Path(sys.prefix)==Path('LOCAL_HOME/mllm/.venv')
    start=time.perf_counter()
    for e in entries:
        b=e['binding'];r=rows[b['id_PRIVATE']]
        assert sha(r['imagepath'])==b['frame_binding']['image_sha256']
        rgb=np.array(Image.open(r['imagepath']).convert('RGB'))
        assert list(rgb.shape)==b['RGB_shape'] and hashlib.sha256(rgb.tobytes()).hexdigest()==b['RGB_array_sha256']
    result={'state':'PASS_R50_WORKLIST_IMAGE_ARRAY_BINDINGS_ACTUAL_EVIDENCE_RUNTIME','n':len(entries),
            'model_calls':0,'GT_or_rawXYZ_access':False,'runtime_executable':sys.executable,'seconds':time.perf_counter()-start}
    out=exp/'WORKLIST_RGB_RUNTIME_CHECK_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
