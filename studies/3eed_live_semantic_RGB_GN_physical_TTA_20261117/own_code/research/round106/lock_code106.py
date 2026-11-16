from coverage106 import *
def main():
 assert rd(H/'ENGINEERING.json')['n']==12;p=H/'experiment/CODE_LOCK.json';assert not p.exists();files={f:sha(R/f) for f in rd(R/'research/round105/experiment/CODE_LOCK.json')['files']};files.update({str(p.relative_to(R)):sha(p) for p in (R/'research/round105').glob('*.py')});files.update({str(p.relative_to(R)):sha(p) for p in H.glob('*.py')});files['tools/exploration_guard_round106.py']=sha(R/'tools/exploration_guard_round106.py');dump(p,{'files':files,'engineering_sha256':sha(H/'ENGINEERING.json'),'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'amendment_sha256':sha(H/'experiment/ENGINEERING_AMENDMENT.json')});print('CODE106_LOCKED',len(files))
if __name__=='__main__':main()
