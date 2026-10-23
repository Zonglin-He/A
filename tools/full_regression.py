"""Exact modular adoption; GT-free, no model or expert execution."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from full import RUN,PUBLIC,read,sha,atomic,check
from adoption_verify import main as verify

def main():
    plan=check(True)
    verify()
    out=read(ROOT/'outputs/adoption_simplepath_20261023/cpu_verification.json')
    out.update(queries=256,spatial_queries=67,plan_sha256=sha(RUN/'plan.json'))
    atomic(RUN/'regression.json',out)
    atomic(PUBLIC/'REGRESSION.json',out)
if __name__=='__main__':main()
