"""Run final validation, record environment/freeze, then execute the authorized campaign."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=str(ROOT),check=True)

def main():
    # Previous full development run is an explicit dependency, not a fixed time delay.
    dependency=ROOT/'reports/dev-gpt6-v6/summary.json'
    while not dependency.exists():time.sleep(10)
    summary=json.loads(dependency.read_text(encoding='utf-8'))
    if not summary['complete']:raise RuntimeError('Development coverage incomplete')
    run('eval.py','smoke')
    run('eval.py','run','--method','fixed','--run-id','dev-fixed-v7','--workers','3')
    subprocess.run([sys.executable,'-m','pytest','-q'],cwd=str(ROOT),check=True)
    run('verify_sources.py')
    run('calibrate.py','--run-id','dev-gpt6-v6','--write-config')
    config_path=ROOT/'configs/protocol.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    config.update(mujoco_stack_doubles=2000000,blas_threads=1,max_workers=3,success_termination='first_official_success')
    config_path.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze']).decode('utf-8')
    (ROOT/'configs/requirements-sim.lock').write_text(freeze,encoding='utf-8')
    run('eval.py','freeze')
    run('update_task_status.py')
    run('run_campaign.py','--workers','3','--prefix','frozen-v2')
    run('update_task_status.py')

if __name__=='__main__':main()
