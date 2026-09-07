"""Replay recorded evaluated actions with the official init state, without LLM calls."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.env import LiberoEnvAdapter,OfficialSuccess
from libero_eval.io import read,dump
from libero_eval.runner import Recorder

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Replay output must be an empty/new directory')
    original=read(args.episode/'result.json')
    run=read(args.episode.parent/'run.json')
    task=read(ROOT/'configs/task_manifest.json')['tasks'][original['task_id']]
    trajectory=np.load(str(args.episode/'trajectory.npz'))
    env=LiberoEnvAdapter(task,run['config'])
    recorder=Recorder(args.output)
    env.on_step=recorder.step
    try:
        env.reset(original['init_state_id'],original['seed'])
        try:
            for action in trajectory['actions'][run['config']['settle_steps']:]:env.step(action)
        except OfficialSuccess:pass
        actual=np.asarray(recorder.states)
        expected=trajectory['sim_states'][:len(actual)]
        result={'source_episode':str(args.episode),'official_success':env.success(),'env_steps':env.steps,'frames':recorder.frames,'source_frames':len(trajectory['actions']),'max_state_error':float(np.max(np.abs(actual-expected))) if actual.shape==expected.shape else None,'state_match_atol_1e-8':bool(actual.shape==expected.shape and np.allclose(actual,expected,atol=1e-8,rtol=0))}
        dump(args.output/'replay_result.json',result)
        print(result)
    finally:
        recorder.close();env.close()

if __name__=='__main__':main()
