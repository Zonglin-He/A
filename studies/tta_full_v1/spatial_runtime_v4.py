"""Original combined Vid collector/TTA hook with verified interface-only engine."""
from runtime_v4 import *

def run():
    validate_execution();install_engine()
    old=ns.load_old('_combined_Vid_source_full_TTA_preserved_v1',S/'spatial_tta.py')
    old.validate=validate_execution
    old.run()

if __name__=='__main__':run()
