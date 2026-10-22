"""Dimension-only adapter for frozen BoxFixed geometry; numerical body unchanged."""
import ast,inspect,textwrap
import tools.boxfixed_tube.spatial as original


def generic_initializer():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.SpatialGeometry.__init__)));fn=tree.body[0];assert isinstance(fn,ast.FunctionDef);changes=0
    class Dimension(ast.NodeTransformer):
        def visit_Constant(self,node):
            nonlocal changes
            if node.value==256:changes+=1;return ast.copy_location(ast.Name(id='D',ctx=ast.Load()),node)
            return node
    fn=Dimension().visit(fn);assert changes==4,'Upstream constructor changed; re-audit dimension adapter'
    indices=[i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='raw'for t in n.targets)];assert len(indices)==1
    fn.body.insert(indices[0]+1,ast.parse('D = raw.shape[-1]').body[0]);tree.body[0]=fn;ast.fix_missing_locations(tree);scope=dict(vars(original));exec(compile(tree,__file__,'exec'),scope);return scope['__init__']

class SpatialGeometry(original.SpatialGeometry):
    __init__=generic_initializer()

def build_geometry(evidence,E,native_boxes):return SpatialGeometry(evidence,E,native_boxes)
