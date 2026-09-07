import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from libero_eval.planner import GPTPlanner
from libero_eval.io import read, dump
out = ROOT/'runs/api_probe'
planner = GPTPlanner(read(ROOT/'configs/protocol.json'), out)
try:
    decision = planner.decide({'language':'Put the can into the basket.'}, {'objects':['can','basket'],'holding':None,'official_success':False}, [])
    result = {'available':True,'decision':decision,'metadata':planner.metadata}
except Exception as exc:
    result = {'available':False,'error':str(exc),'metadata':planner.metadata}
dump(out/'result.json',result)
print(result)
