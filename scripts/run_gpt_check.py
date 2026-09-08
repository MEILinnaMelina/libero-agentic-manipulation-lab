"""One immutable GPT episode per selected task, with an isolated budget and fail-fast API stop.

This is a ten-episode check, not completion or resumption of the old formal campaign.
No completed or interrupted attempt is rerun by this entry point.
"""
import os
for var in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[var]='1'
import argparse
import json
import sqlite3
import sys
import time
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.io import read,dump,code_hash,digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--state',type=int,default=3)
    parser.add_argument('--tasks',default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--budget-usd',type=float,default=10.)
    args=parser.parse_args()
    ids=[int(v) for v in args.tasks.split(',')]
    if not ids or len(ids)!=len(set(ids)) or any(t not in range(10) for t in ids):
        raise ValueError('Invalid or duplicate task IDs')
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        raise ValueError('Invalid run ID')
    if not 0<args.budget_usd<=10:
        raise ValueError('This diagnostic permits at most USD 10')
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is missing; no request was made')
    run=ROOT/'runs'/args.run_id
    run.mkdir(exist_ok=False)
    config=read(ROOT/'configs/protocol.json')
    config.update(formal_budget_usd=args.budget_usd,budget_ledger=(run/'budget.sqlite').relative_to(ROOT).as_posix(),max_workers=1)
    manifest=read(ROOT/'configs/task_manifest.json')
    tasks=[t for t in manifest['tasks'] if t['task_id'] in ids]
    for task in tasks:
        if args.state not in task['formal_state_ids']:
            raise ValueError('Choose one official non-development state')
    fingerprint=code_hash(ROOT)
    schedule=[dict(task_id=t['task_id'],init_state_id=args.state) for t in tasks]
    identity=dict(run_id=args.run_id,method='gpt6',split='formal',formal_benchmark=False,purpose='selected_task_single_state_check',code_hash=fingerprint,config_hash=digest(ROOT/'configs/protocol.json'),manifest_hash=digest(ROOT/'configs/task_manifest.json'),schedule=schedule,config=config,created_at=time.time(),note='One state per selected task; new versions retain all earlier successes and failures.')
    dump(run/'run.json',identity)
    with zipfile.ZipFile(str(run/'source.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['src','scripts','configs','tests']:
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(str(path),path.relative_to(ROOT).as_posix())
    from libero_eval.runner import episode
    from libero_eval.report import summarize
    results=[]
    stop_reason=None
    try:
        for task in tasks:
            if code_hash(ROOT)!=fingerprint:
                raise RuntimeError('Source changed during diagnostic')
            directory=run/('task%02d_state%03d'%(task['task_id'],args.state))
            print(json.dumps(dict(event='episode_start',task_id=task['task_id'],state=args.state)),flush=True)
            result=episode(task,args.state,config,directory,args.run_id,'gpt6','formal',fingerprint)
            results.append(result)
            print(json.dumps({k:result[k] for k in ['task_id','success','termination_reason','env_steps','llm_calls','llm_input_tokens','llm_output_tokens','wall_seconds']}),flush=True)
            if result['termination_reason']=='api_error':
                stop_reason='api_error: '+result.get('error','unknown')
                # No later task may repeat an exhausted-account/invalid-key request.
                # Also stop on unknown API failures; unexecuted tasks stay unexecuted.
                break
    except BaseException as exc:
        stop_reason='execution_interrupted: '+type(exc).__name__
        raise
    finally:
        summary=summarize(run)
        ledger=run/'budget.sqlite'
        balance=[]
        if ledger.exists():
            with sqlite3.connect(str(ledger)) as db:
                balance=[dict(state=s,requests=n,amount_usd=a) for s,n,a in db.execute('SELECT state,COUNT(*),SUM(amount) FROM requests GROUP BY state')]
        status=dict(run_id=args.run_id,scheduled=len(schedule),completed=len(results),successes=sum(r['success'] for r in results),unexecuted=len(schedule)-len(results),stop_reason=stop_reason,llm_calls=sum(r['llm_calls'] for r in results),budget_usd=args.budget_usd,ledger=balance,formal_benchmark=False,code_hash=fingerprint)
        dump(run/'check_status.json',status)
        print(json.dumps(status),flush=True)

if __name__=='__main__':main()
