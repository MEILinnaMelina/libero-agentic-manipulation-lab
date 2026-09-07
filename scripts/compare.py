import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.io import read,dump
from libero_eval.report import summarize

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix',default='frozen-v1')
    args=parser.parse_args()
    collections={}
    for suffix in ['gpt6','fixed','no-replan']:
        run=ROOT/'runs'/(args.prefix+'-'+suffix)
        summary=summarize(run)
        if not summary['complete']:raise RuntimeError('Incomplete condition: '+suffix)
        rows=read(ROOT/'reports'/(args.prefix+'-'+suffix)/'raw_results.json')
        collections[suffix]={(r['task_id'],r['init_state_id']):r for r in rows}
    matched=set.intersection(*(set(r) for r in collections.values()))
    if len(matched)!=50: raise RuntimeError('Expected 50 matched diagnostic states')
    rng=np.random.RandomState(20260907)
    results=[]
    for alternative in ['fixed','no-replan']:
        samples=[]
        for task in range(10):
            keys=sorted(k for k in matched if k[0]==task)
            d=np.array([float(collections['gpt6'][k]['success'])-float(collections[alternative][k]['success']) for k in keys])
            samples.append(d)
        boot=np.zeros(10000)
        for d in samples:boot+=rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1)/10
        result={'reference':'gpt6','alternative':alternative,'matched_n':50,'gpt6_successes':sum(collections['gpt6'][k]['success'] for k in matched),'alternative_successes':sum(collections[alternative][k]['success'] for k in matched),'paired_macro_difference':float(np.mean([d.mean() for d in samples])),'paired_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
        results.append(result)
    dump(ROOT/'reports'/args.prefix/'paired_comparison.json',results)
    print(results)

if __name__=='__main__':main()
