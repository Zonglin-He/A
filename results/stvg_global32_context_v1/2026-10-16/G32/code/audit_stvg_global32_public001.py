"""Independent saved scalar and anonymity audit for original global32-A export."""
from pathlib import Path
import ast,inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_stvg_controls256_public005 as original
def main():
    s=inspect.getsource(original.main);s=s.replace('stvg_spatial_controls256_v1','stvg_global32_context_v1');s=s.replace("            assert p['worst_sources']==sorted(delta,key=lambda c:(delta[c],original_id[c]))[:8]\n",'')
    class Count(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value)is int and n.value in (128,256,257):return ast.copy_location(ast.Constant({128:16,256:32,257:33}[n.value]),n)
            return n
    tree=ast.fix_missing_locations(Count().visit(ast.parse(s)));ns=dict(original.__dict__);exec(compile(tree,str(Path(__file__)),'exec'),ns);ns['main']('G32')
if __name__=='__main__':main()
