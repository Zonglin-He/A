"""Reuse the existing text-only dependency parser and role inventory, no task labels."""
from pathlib import Path
import sys,json,time,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round97';B=R/'crossplatform/round97'
sys.path[:0]=[str(R/'crossplatform/round48'),str(R/'research/round85'),str(R/'research/round86'),str(R/'research/round87'),str(R/'crossplatform/round20/runtime')]
from parser48 import parse_subject,dependency_document
from evidence85 import parse85
from reference_roles87 import repair_inventory87
import stanza
