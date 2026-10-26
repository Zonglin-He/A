"""Preserved original CPU doctor, isolated missing tabulate dependency supplied."""
import importlib.util
from pathlib import Path
p=Path(__file__).with_name('runtime_doctor.py');s=importlib.util.spec_from_file_location('_alternative_expert_CPU_doctor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
if __name__=='__main__':m.run()
