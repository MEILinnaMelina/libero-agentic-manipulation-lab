import csv
import math
from collections import Counter
import numpy as np
from .io import read, dump
from .bootstrap import ROOT

def wilson(successes,n):
    if not n: return None
    z=1.959963984540054
    p=successes/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def summarize(run):
    identity=read(run/'run.json')
    rows=[]
    expected={(e['task_id'],e['init_state_id']) for e in identity['schedule']}
    seen=set()
    for p in sorted(run.glob('task*_state*/result.json')):
        r=read(p); key=(r['task_id'],r['init_state_id'])
        if key in seen or key not in expected: raise ValueError('Duplicate/unexpected result: '+str(key))
        if any(r[k]!=identity[k] for k in ['method','split','code_hash','config_hash','manifest_hash']): raise ValueError('Mixed protocol/code results')
        if r['env_steps']>identity['config']['max_env_steps']: raise ValueError('Step budget exceeded')
        seen.add(key);rows.append(r)
    per_task=[]
    for i in sorted({x[0] for x in expected}):
        values=[r for r in rows if r['task_id']==i]
        n=sum(1 for k in expected if k[0]==i)
        s=sum(r['success'] for r in values)
        per_task.append({'task_id':i,'scheduled':n,'completed':len(values),'missing':n-len(values),'successes':s,'success_rate':s/n,'wilson95':wilson(s,n),'termination_counts':dict(Counter(r['termination_reason'] for r in values))})
    complete=seen==expected
    macro=float(np.mean([t['success_rate'] for t in per_task])) if per_task else None
    # Stratified episode bootstrap preserves equal task weight. Missing count as unsuccessful;
    # incomplete experiments are labelled incomplete and never called final GPT performance.
    rng=np.random.RandomState(20260907)
    estimates=np.zeros(10000)
    for t in per_task:
        estimates+=rng.binomial(t['scheduled'],t['success_rate'],10000)/t['scheduled']/len(per_task)
    summary={'run_id':identity['run_id'],'method':identity['method'],'split':identity['split'],'complete':complete,'scheduled':len(expected),'completed':len(rows),'missing':sorted(expected-seen),'per_task':per_task,'macro_success_rate':macro,'macro_bootstrap95':np.quantile(estimates,[.025,.975]).tolist(),'bootstrap_note':'Stratified empirical bootstrap; degenerates at all-zero/all-one samples. See Wilson per-task intervals.', 'micro_wilson95':wilson(sum(r['success'] for r in rows),len(expected)), 'failure_counts':dict(Counter(f for r in rows for f in r['skill_failures'])),'mean_llm_calls':float(np.mean([r['llm_calls'] for r in rows])) if rows else 0.,'mean_replans':float(np.mean([r['replans'] for r in rows])) if rows else 0.,'total_input_tokens':sum(r['llm_input_tokens'] for r in rows),'total_output_tokens':sum(r['llm_output_tokens'] for r in rows),'total_llm_latency_seconds':sum(r['llm_latency_seconds'] for r in rows),'total_episode_wall_seconds':sum(r['wall_seconds'] for r in rows)}
    report=ROOT/'reports'/identity['run_id']
    if identity['config'].get('pricing_usd_per_million'):
        from .budget import actual_cost
        metadatas=[read(p) for p in run.glob('task*/llm_*.metadata.json')]
        summary['estimated_cost_usd']=sum(actual_cost(m.get('usage',{}),identity['config']['pricing_usd_per_million']) for m in metadatas)
        summary['cost_note']='Usage-priced estimate, not an invoice; unavailable-response usage remains unknown.'
    dump(report/'summary.json',summary)
    dump(report/'raw_results.json',rows)
    with (report/'per_task.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['task_id','scheduled','completed','missing','successes','success_rate','wilson95','termination_counts'])
        writer.writeheader();writer.writerows(per_task)
    lines=['# '+identity['run_id'], '', 'Method: '+identity['method']+'; split: '+identity['split']+'.', '', 'Completed: %d/%d. Macro success: %.2f%%.'%(len(rows),len(expected),(macro or 0)*100),'', '| Task | Success / scheduled | Rate | 95% Wilson CI |','|---|---|---|---|']
    for t in per_task:
        lines.append('| %d | %d / %d | %.2f%% | %.2f%%–%.2f%% |'%(t['task_id'],t['successes'],t['scheduled'],100*t['success_rate'],100*t['wilson95'][0],100*t['wilson95'][1]))
    lines+=['','Failures: '+str(summary['failure_counts']), '', 'These are state-and-BDDL-assisted manipulation results, not RGB-only VLA or lifelong learning results.']
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Summary: '+str(report/'summary.md'),flush=True)
    return summary
