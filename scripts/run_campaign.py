"""Resumable serial stages. Each stage may evaluate independent episodes in workers."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prefix',default='frozen-v1')
    args=parser.parse_args()
    # Four-worker diagnostics fit after memory fixes, but leave little system commit
    # headroom. Use at most three for the long formal campaign on this host.
    args.workers=min(args.workers,3)
    commands=[
        ['run','--split','formal','--run-id',args.prefix+'-gpt6'],
        ['run','--split','formal','--states','3,4,5,6,7','--method','fixed','--run-id',args.prefix+'-fixed'],
        ['run','--split','formal','--states','3,4,5,6,7','--method','gpt6_no_replan','--run-id',args.prefix+'-no-replan']
    ]
    for cmd in commands:
        subprocess.run([sys.executable,str(ROOT/'scripts/eval.py'),*cmd,'--workers',str(args.workers)],cwd=str(ROOT),check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/compare.py'),'--prefix',args.prefix],cwd=str(ROOT),check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_report.py'),'--prefix',args.prefix],cwd=str(ROOT),check=True)
    import json
    records=json.loads((ROOT/'reports'/(args.prefix+'-gpt6')/'raw_results.json').read_text(encoding='utf-8'))
    for success,label in [(True,'success'),(False,'failure')]:
        source=next((r for r in records if r['success']==success and r['env_steps']>0),None)
        if source is None:continue
        episode=ROOT/'runs'/(args.prefix+'-gpt6')/('task%02d_state%03d'%(source['task_id'],source['init_state_id']))
        output=ROOT/'reports'/args.prefix/'replay'/label
        if (output/'replay_result.json').exists():continue
        subprocess.run([sys.executable,str(ROOT/'scripts/replay.py'),'--episode',str(episode),'--output',str(output)],cwd=str(ROOT),check=True)

if __name__=='__main__':main()
