"""Pinned official SigLIP dependency solely for the official-loader comparison."""
import sys,json,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import download_llava_st_official as download

def main():
    download.OUT=ROOT/'artifacts/desta3d_v3/external_privileged_opd_v2/siglip_download'
    download.DEST=ROOT/'checkpoints/llava_official_siglip-so400m-patch14-384'
    download.OUT.mkdir(parents=True,exist_ok=True)
    reg=download.OUT/'MODEL_DOWNLOAD_REGISTRATION.json'
    if not reg.exists():
        reg.write_text(json.dumps(dict(repo='google/siglip-so400m-patch14-384',revision='9fdffc58afc957d1a03a25b10dba0329ab15c2a3',
            purpose='Official load_lora_model constructor equivalence; final full LLaVA checkpoint overwrites vision tensors',
            files=[dict(path='config.json',size=576,lfs_sha256=None),dict(path='model.safetensors',size=3511950624,
            lfs_sha256='ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a')]),indent=2)+'\n')
    try:download.main()
    except BaseException as e:
        (download.OUT/'HTTP_DOWNLOAD_FAILURE.json').write_text(json.dumps(dict(time=time.time(),error=repr(e))))
        raise
if __name__=='__main__':main()
