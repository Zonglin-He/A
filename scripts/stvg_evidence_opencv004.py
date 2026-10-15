"""Use the pre-existing isolated OpenCV-contrib before any candidate module import."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'artifacts/stvg_weightfree_spatial_v1/deps'));sys.path.insert(0,str(ROOT))
import cv2
assert cv2.__version__=='4.11.0'and hasattr(cv2,'ximgproc'),(cv2.__version__,cv2.__file__)
if __name__=='__main__':
 sys.argv=[str(ROOT/'scripts/stvg_evidence_stage003.py'),*sys.argv[1:]];runpy.run_path(str(ROOT/'scripts/stvg_evidence_stage003.py'),run_name='__main__')
