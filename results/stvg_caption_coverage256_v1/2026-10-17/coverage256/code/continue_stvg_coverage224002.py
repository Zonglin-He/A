"""Resume after zero-WHO launcher failure, preserving completed qualification."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_coverage224_v1 as original
def main():
    source=inspect.getsource(original.main);assert source.count("'run_stvg_coverage224_v1.py'")==1;source=source.replace("'run_stvg_coverage224_v1.py'","'run_stvg_coverage224002.py'")
    ns=dict(original.__dict__);ns['UNIT']='stvg-coverage224-002.service';exec(compile(source,__file__,'exec'),ns);ns['main']()
if __name__=='__main__':main()
