"""Unexecuted full5912 selected-factor wrapper; original ablation is immutable.

No algorithm change: only replace the original global factor/arm lists with the
single DEV-selected factor pinned in the new full-method execution plan.
"""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import noassociation51

ORIGINAL_SHA='052c52b2a3f285176606250e7f80fb1da48762a128052cc1012f5ee6c74751c0'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True);args=p.parse_args()
 plan=noassociation51.read(args.plan);auth=noassociation51.read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_NO_ASSOCIATION_RAW_SAFE_ABLATION_NO_GT' and auth['plan_sha256']==noassociation51.sha(args.plan)
 assert plan['n']==5912 and plan['method_selected_before_full_run'] is True and plan['selected_method']
 values=plan['appearance_factor_values'];assert len(values)==1 and values[0] in (.5,1.,2.)
 assert noassociation51.sha(noassociation51.__file__)==ORIGINAL_SHA
 own=Path(__file__).resolve().relative_to(ROOT).as_posix();assert plan['programs_sha256'][own]==noassociation51.sha(__file__)
 noassociation51.FACTORS=list(values)
 names={.5:'no_instance_association_c05',1.:'no_instance_association_c1',2.:'no_instance_association_c2'}
 noassociation51.ARMS=[names[values[0]]]
 noassociation51.main()

if __name__=='__main__':main()
