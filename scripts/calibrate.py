"""Freeze pilot-derived costs and limits; standard public pricing is an estimate."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.io import read,dump

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='pilot-gpt6-v1')
    p.add_argument('--write-config',action='store_true')
    args=p.parse_args()
    run=ROOT/'runs'/args.run_id
    rows=[read(p) for p in run.glob('task*/result.json')]
    if not rows or any(r['llm_calls']==0 for r in rows):
        raise RuntimeError('Need actual GPT pilot calls for all selected pilot episodes')
    n=len(rows)
    usage={'episodes':n,'calls':sum(r['llm_calls'] for r in rows),'input_tokens':sum(r['llm_input_tokens'] for r in rows),'output_tokens':sum(r['llm_output_tokens'] for r in rows),'episode_wall_seconds':sum(r['wall_seconds'] for r in rows)}
    cost=(usage['input_tokens']*10+usage['output_tokens']*50)/1e6
    manifest=read(ROOT/'configs/task_manifest.json')
    formal=sum(len(t['formal_state_ids']) for t in manifest['tasks'])
    estimate={'pilot_run_id':args.run_id,'actual_usage':usage,'pricing_source':'https://developers.openai.com/api/docs/models/gpt-6-astra','pricing_date':'2026-09-07','usd_per_million':{'uncached_input':10,'cached_input':1,'cache_write':12.5,'output':50},'cost_note':'Conservative no-cache Standard-price estimate; actual invoice may differ. Public rates, not a billing receipt.', 'pilot_estimated_usd':cost,'formal_episodes':formal,'formal_projected_usd':cost/n*formal,'formal_projected_serial_hours':usage['episode_wall_seconds']/n*formal/3600,'ablation_states_per_task':5,'ablation_projected_usd':cost/n*50, 'execution_budget':{'campaign_usd_cap':600,'max_llm_calls_per_episode':24,'max_output_tokens':4096,'max_replans':3,'workers':1}}
    dump(ROOT/'reports/cost_estimate.json',estimate)
    if args.write_config:
        config=read(ROOT/'configs/protocol.json')
        config.update(calibrated=True,formal_budget_usd=600,pricing_usd_per_million=estimate['usd_per_million'],ablation_states_per_task=5)
        dump(ROOT/'configs/protocol.json',config)
    print(json.dumps(estimate,indent=2))

if __name__=='__main__': main()
