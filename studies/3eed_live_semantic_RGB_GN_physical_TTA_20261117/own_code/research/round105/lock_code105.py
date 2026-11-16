from inventory105 import *
def main():
 assert rd(H/'ENGINEERING.json')['n']==12;p=H/'experiment/CODE_LOCK.json';assert not p.exists();old=rd(R/'research/round104/experiment/CODE_LOCK.json')['code_sha256'];files={k:sha(R/k) for k in old};files.update({str(p.relative_to(R)):sha(p) for p in H.glob('*.py')});files['tools/exploration_guard_round105.py']=sha(R/'tools/exploration_guard_round105.py');dump(p,{'files':files,'engineering_sha256':sha(H/'ENGINEERING.json'),'design_sha256':sha(H/'experiment/DESIGN_LOCK.json'),'full_registration_seal_sha256':sha(R/'crossplatform/round105/registration_full_PRIVATE/SEALED.json'),'registration_sensor_only_before_task':True});print('CODE105_LOCKED',len(files))
if __name__=='__main__':main()
