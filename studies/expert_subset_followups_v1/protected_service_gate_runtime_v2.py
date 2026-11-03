"""Verify the already allowed root ToDesk service without forbidden /proc/exe access."""
from pathlib import Path
import subprocess,json
ALLOWED='/opt/todesk/bin/ToDesk_Session'
def identity_from_proc(pid,image):
 assert image==ALLOWED,(pid,image)
 base=Path('/proc')/str(pid);before=base.joinpath('stat').read_text().split(') ',1)[1].split()[19];status=base.joinpath('status').read_text();uid=next(x for x in status.splitlines()if x.startswith('Uid:')).split()[1:];assert all(int(x)==0 for x in uid),uid
 args=base.joinpath('cmdline').read_bytes().split(b'\0');comm=base.joinpath('comm').read_text().strip();assert args[0].decode()==ALLOWED and comm=='ToDesk_Session'
 assert base.joinpath('stat').read_text().split(') ',1)[1].split()[19]==before
 try:exe=str(base.joinpath('exe').resolve());assert exe==ALLOWED;access='exe_and_NV_driver_and_root_uid_cmdline_comm'
 except PermissionError:exe=None;access='exe_permission_denied_preserved_NV_driver_full_imagepath_root_uid_cmdline_comm_start_identity_verified'
 return dict(pid=pid,startticks=int(before),NVIDIA_driver_image=image,root_uid_verified=True,cmdline0=ALLOWED,comm=comm,proc_exe=exe,identity_evidence=access)
def foreign_check():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);entries=[]
 for line in out.splitlines():
  pid,name=line.split(',',1);entries.append(identity_from_proc(int(pid),name.strip()))
 return entries
if __name__=='__main__':
 # Unknown compute images fail before any proc/file lookup; not a wildcard allow.
 try:identity_from_proc(-1,'/bin/python')
 except AssertionError:pass
 else:raise AssertionError('foreign image unexpectedly allowed')
 print(json.dumps(dict(status='passed_strict_protected_service_identity_no_model',current=foreign_check(),new_GPU=0,new_model=0,new_Source=0,new_expert=0)))
