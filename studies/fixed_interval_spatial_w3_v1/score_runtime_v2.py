"""Engineering import compatibility only; original scoring main/formulas unchanged."""
from common import *
import ast,argparse
import score as numerical
source=(R/'studies/spatial_table_recheck_v1/common.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='corner');exact=ast.get_source_segment(source,node)
assert exact.startswith('def corner(')and 'corner'not in numerical.evaluator.__dict__
exec(compile(exact,str(R/'studies/spatial_table_recheck_v1/common.py'),'exec'),numerical.evaluator.__dict__)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);numerical.main(p.parse_args().dataset)
