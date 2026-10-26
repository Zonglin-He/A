"""Same frozen actual-output auditor, additionally bind current execution v2."""
from eg_execution_common_v2 import *
import eg_pilot_root_v1 as original
import argparse
if __name__=='__main__':
 validate_execution();a=argparse.ArgumentParser();a.add_argument('command',choices=['bridge','audit']);a.add_argument('--dataset',choices=rc.COUNTS,required=True);x=a.parse_args();getattr(original,x.command)(x.dataset)
