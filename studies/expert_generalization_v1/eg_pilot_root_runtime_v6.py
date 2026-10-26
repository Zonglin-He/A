"""Unchanged actual expert output/root numeric auditor with latest bindings."""
from eg_execution_common_v6 import *
import eg_pilot_root_v1 as original
import argparse
if __name__=='__main__':
 validate_execution_v6();p=argparse.ArgumentParser();p.add_argument('command',choices=['bridge','audit']);p.add_argument('--dataset',choices=rc.COUNTS,required=True);a=p.parse_args();getattr(original,a.command)(a.dataset)
