"""Allow one extra completed support-diagnosis group, original publisher intact."""
from pathlib import Path
import ast,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=ROOT/'scripts/publish_stvg_instance_evidence32_v1.py';source=p.read_text();assert source.count("'closing001'")==1
m=types.ModuleType('scripts._instance_support_publisher');m.__dict__.update(__name__=m.__name__,__file__=str(p));exec(compile(ast.parse(source.replace("'closing001'","'support001'")),str(p),'exec'),m.__dict__)
if __name__=='__main__':m.main()
