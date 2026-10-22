"""Recreate isolated dependencies without changing the frozen TubeDETR environment."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ENV=ROOT.parent/'envs/llava-st'
if __name__=='__main__':
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-m','venv','--system-site-packages',str(ENV)],check=True)
    site=ENV/'lib/python3.10/site-packages';(site/'frozen_tubedetr_dependencies.pth').write_text(str(ROOT/'.conda/tubedetr/lib/python3.10/site-packages')+'\n')
    subprocess.run([str(ENV/'bin/python'),'-m','pip','install','--no-deps','accelerate==1.6.0','decord==0.6.0','shortuuid','einops-exts==0.0.4','huggingface_hub==0.36.2','tokenizers==0.15.2','transformers @ https://github.com/huggingface/transformers/archive/1c39974a4c4036fd641bc1191cc32799f85715a4.zip'],check=True)
    upstream=ROOT/'runtime/llava_st/upstream'
    if not(upstream/'.git').exists():subprocess.run(['git','clone','https://github.com/appletea233/LLaVA-ST.git',str(upstream)],check=True)
    subprocess.run(['git','-C',str(upstream),'checkout','--detach','bacf6d61e1de27a78fc0025083aa445d3d41b0b3'],check=True)
