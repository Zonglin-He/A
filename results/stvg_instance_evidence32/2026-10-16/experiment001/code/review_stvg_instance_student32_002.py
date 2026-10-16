"""Reader-only float64 box conversion matching the complete model's original readout.

The original run, predictions, fit and source pins stay immutable. No tolerance
or scientific fit expression changes. Exactly one saved-box conversion changes.
"""
from pathlib import Path
import ast,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_instance_student32_v1 as original
from scripts.stvg_instance_evidence32_v1 import BASE,read,put,sha
def main():
    tid=sys.argv[1];source=textwrap.dedent(inspect.getsource(original.root));needle="bb=arr(o['boxes']);xy=";assert source.count(needle)==1
    new=source.replace(needle,"bb=arr(o['boxes']).astype(float);xy=")
    oldtree=ast.parse(source);newtree=ast.parse(new)
    class Strip(ast.NodeTransformer):
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Attribute)and n.func.attr=='astype'and len(n.args)==1 and isinstance(n.args[0],ast.Name)and n.args[0].id=='float':return n.func.value
            return n
    assert ast.dump(oldtree,include_attributes=False)==ast.dump(Strip().visit(newtree),include_attributes=False)
    recovery=BASE/'recovery/reader_precision002';lock=read(recovery/'RUNTIME.json')
    for rel,h in lock['pins'].items():assert sha(ROOT/rel)==h,rel
    ns=dict(original.__dict__);exec(compile(new,str(Path(__file__)),'exec'),ns);ns['root'](tid)
if __name__=='__main__':main()
