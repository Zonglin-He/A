"""Single owned experiment sequence; never resumes full or chooses from labels."""
import sys,subprocess,fcntl,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from research.native_event_dense_projection_v1.run import D,read,put,who

def main():
 D.mkdir(parents=True,exist_ok=True);lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(D/'state.json')['status']=='prepared';put(D/'controller.json',dict(**who(),role='fixed HC64 branch controller',started=time.time()))
 py=R/'.conda/tubedetr/bin/python'
 for stage,args in [('predict',['run.py','predict']),('score',['score.py'])]:
  with (D/(stage+'.log')).open('a')as log:
   proc=subprocess.Popen([str(py),'-B',str(Path(__file__).parent/args[0]),*args[1:]],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   child=dict(pid=proc.pid,startticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));put(D/'child.json',dict(stage=stage,**child));rc=proc.wait();put(D/(stage+'-exit.json'),dict(returncode=rc,child=child,finished=time.time()))
   if rc:raise SystemExit(rc)
 put(D/'controller_complete.json',dict(complete=True,full_resumed=False,finished=time.time()))
if __name__=='__main__':main()
