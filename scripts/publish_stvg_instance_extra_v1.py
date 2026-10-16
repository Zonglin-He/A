"""Explicit extra-group routing; original publisher and old locks unchanged."""
from pathlib import Path
import ast,sys,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    group=sys.argv[1];assert group in ('interpretation001','backbone001')
    p=ROOT/'scripts/publish_stvg_instance_evidence32_v1.py';s=p.read_text();assert s.count("'closing001'")==1
    m=types.ModuleType('scripts._instance_extra_publisher');m.__dict__.update(__name__=m.__name__,__file__=str(p));exec(compile(ast.parse(s.replace("'closing001'",repr(group))),str(p),'exec'),m.__dict__);m.main()
