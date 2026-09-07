"""Offline-only, single-task diagnostic runner. Network transport is disabled."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ.pop('OPENAI_API_KEY',None)
import sys, argparse, socket
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
def no_network(*args,**kwargs):
    raise RuntimeError('Network disabled: fixed semantic diagnostics only')
socket.socket.connect=no_network
socket.create_connection=no_network
from libero_eval.bootstrap import ROOT
from libero_eval.io import read,dump,code_hash,append
from libero_eval.runner import episode

if __name__=='__main__':
    p=argparse.ArgumentParser()
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--task',type=int)
    selection.add_argument('--tasks')
    p.add_argument('--states',default='0')
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if not a.run_id or Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):
        raise ValueError('run-id must be a simple directory name')
    config=read(ROOT/'configs/protocol.json')
    config['fixed_stop_on_failure']=True
    manifest=read(ROOT/'configs/task_manifest.json')
    tasks=[a.task] if a.task is not None else list(map(int,a.tasks.split(',')))
    states=list(map(int,a.states.split(',')))
    if len(tasks)!=len(set(tasks)) or any(t not in range(10) for t in tasks):raise ValueError('Invalid task IDs')
    if len(states)!=len(set(states)):raise ValueError('Duplicate state IDs')
    for task in tasks:
        if not set(states)<=set(manifest['tasks'][task]['dev_state_ids']):raise ValueError('Dev states only')
    run=ROOT/'runs'/a.run_id
    if run.exists(): raise RuntimeError('Use a new diagnostic run ID; preserve old results')
    run.mkdir(parents=True)
    import zipfile
    with zipfile.ZipFile(str(run/'source.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['src','scripts','configs','tests']:
            for file in (ROOT/folder).rglob('*'):
                if file.is_file() and '__pycache__' not in file.parts:
                    z.write(str(file),file.relative_to(ROOT).as_posix())
    fingerprint=code_hash(ROOT)
    dump(run/'diagnostic.json',dict(method='fixed',network_disabled=True,task_ids=tasks,state_ids=states,code_hash=fingerprint,config=config))
    from libero_eval.io import digest
    dump(run/'run.json',dict(run_id=a.run_id,method='fixed',split='dev',code_hash=fingerprint,config_hash=digest(ROOT/'configs/protocol.json'),manifest_hash=digest(ROOT/'configs/task_manifest.json'),schedule=[dict(task_id=t,init_state_id=s) for t in tasks for s in states],config=config))
    for task in tasks:
        for state in states:
            if code_hash(ROOT)!=fingerprint:raise RuntimeError('Source changed during validation')
            result=episode(manifest['tasks'][task],state,config,run/('task%02d_state%03d'%(task,state)),a.run_id,'fixed','dev',fingerprint)
            print({k:result[k] for k in ['task_id','init_state_id','success','termination_reason','env_steps','llm_calls','skill_failures']},flush=True)
            append(ROOT/'docs/fixed_repair_attempts.jsonl',result)
    from libero_eval.report import summarize
    summarize(run)
