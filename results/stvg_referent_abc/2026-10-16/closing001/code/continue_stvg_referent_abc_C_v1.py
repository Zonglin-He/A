"""One finite pidfd handoff B -> C; no polling, scoring or scientific changes."""
from pathlib import Path
import hashlib,json,os,select,subprocess
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_referent_abc_v1/revision002'
def props():
    raw=subprocess.check_output(['systemctl','--user','show','stvg-referent-abc-b-002.service','-p','MainPID','-p','ActiveState','-p','Result','-p','ExecMainStatus'],text=True);return dict(x.split('=',1)for x in raw.strip().splitlines())
def save(name,d):
    p=B/'handoff'/name;p.parent.mkdir(exist_ok=True)
    with p.open('x')as f:json.dump(d,f,indent=2);f.write('\n')
p=props();pid=int(p['MainPID']);assert p['ActiveState']=='active'and pid>0;fd=os.pidfd_open(pid);cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes();assert b'run_stvg_referent_abc002.py' in cmd and cmd.endswith(b'supervise\x00B\x00');start=(Path('/proc')/str(pid)/'stat').read_text().split()[21]
save('C_WAIT_LAUNCH.json',dict(status='waiting',stage='B_normal_exit_then_C',pid=pid,start_tick=start,cmdline_sha256=hashlib.sha256(cmd).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GT_read=False))
poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd);p=props();assert p['Result']=='success'and p['ExecMainStatus']=='0'
for name,status in [('STUDENT_GLOBAL_BARRIER.json','complete'),('GLOBAL_PREDICTION_BARRIER.json','complete'),('ROOT_READBACK.json','pass')]:
    d=json.loads((B/'trials/B'/name).read_text());assert d['status']==status and d['count']==16 and d['GT_read']is False
save('C_HANDOFF_ROOT.json',dict(status='pass',stage='C_actual_supervisor_dispatch',count=16,GT_read=False,B_exit_status=0,no_GPU_overlap=True))
exe=ROOT/'.conda/tubedetr/bin/python';os.execv(exe,[str(exe),'-B',str(ROOT/'scripts/run_stvg_referent_abc002.py'),'supervise','C'])
