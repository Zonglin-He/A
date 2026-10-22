"""CPU engineering checks only. Never launch a model or a dataset experiment."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':subprocess.run([sys.executable,'-B','-m','pytest','-q','tests'],cwd=ROOT,check=True)
