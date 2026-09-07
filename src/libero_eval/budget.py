"""Persistent reservation ledger prevents paid requests from exceeding campaign cap."""
import sqlite3
import time
import uuid
from .bootstrap import ROOT

def actual_cost(usage, prices):
    details=usage.get('input_tokens_details') or {}
    cached=details.get('cached_tokens',0)
    writes=details.get('cache_write_tokens',0)
    ordinary=max(0,usage.get('input_tokens',0)-cached-writes)
    return (ordinary*prices['uncached_input']+cached*prices['cached_input']+writes*prices['cache_write']+usage.get('output_tokens',0)*prices['output'])/1e6

def reserve(payload,config):
    if not config.get('calibrated'):
        return None
    prices=config['pricing_usd_per_million']
    # UTF-8 bytes bound text token count conservatively; includes schema and overhead.
    import json
    max_input=len(json.dumps(payload,ensure_ascii=False).encode('utf-8'))+2048
    if max_input>272000: raise RuntimeError('Long-context price tier not permitted by this protocol')
    bound=(max_input*max(prices['uncached_input'],prices['cache_write'])+config['max_output_tokens']*prices['output'])/1e6
    token=str(uuid.uuid4())
    with sqlite3.connect(str(ROOT/'runs/budget.sqlite'),timeout=30,isolation_level=None) as db:
        db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, amount REAL, state TEXT, created REAL)')
        db.execute('BEGIN IMMEDIATE')
        total=db.execute('SELECT COALESCE(SUM(amount),0) FROM requests').fetchone()[0]
        if total+bound>config['formal_budget_usd']:
            db.rollback(); raise RuntimeError('campaign_cost_budget')
        db.execute('INSERT INTO requests VALUES (?,?,?,?)',(token,bound,'reserved',time.time()))
        db.commit()
    return token

def settle(token,usage,config):
    if token is None:return
    with sqlite3.connect(str(ROOT/'runs/budget.sqlite'),timeout=30) as db:
        db.execute('UPDATE requests SET amount=?,state=? WHERE id=?',(actual_cost(usage,config['pricing_usd_per_million']),'observed',token))
