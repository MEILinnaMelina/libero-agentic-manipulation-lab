"""Responses REST transport: stdlib only, compatible with simulator Python 3.8.

Adapted from RoboEval agentic_v2/prompts.py and llm_planner.py.
No hidden SDK retries, model substitutions, or cross-episode conversation state.
"""
import json
import os
import time
import urllib.request
import urllib.error
from .io import dump

VERSION = 'libero.agentic_v2.skill_request.v1'
SKILLS = {
    'clear_obstruction': 'Hand is empty; physically move a movable obstacle away from a blocked grasp approach to free workspace. Object names the obstacle; deterministic code selects the destination.',
    'grasp': 'Object is movable and not held; one Panda arm.',
    'lift': 'Object must be held.',
    'transport': 'Object must be held; goal is a named object/region.',
    'place': 'Object must be held; goal is a named object/region; release and settle.',
    'open_drawer': 'Named articulated region is a sliding drawer; hand is empty.',
    'close_drawer': 'Named sliding drawer is unobstructed; hand is empty.',
    'open_door': 'Named hinged appliance; hand is empty.',
    'close_door': 'Named hinged appliance; placed objects clear the door arc.',
    'turn_on': 'Named stove has a rotary switch; hand is empty.',
    'finish': 'Official success is true, or no recovery remains.'
}
SCHEMA = {'type':'object', 'additionalProperties':False, 'required':['schema_version','thought','request'], 'properties':{
    'schema_version':{'type':'string','enum':[VERSION]}, 'thought':{'type':'string'},
    'request':{'type':'object','additionalProperties':False,'required':['skill','object','goal','strategy'], 'properties':{
        'skill':{'type':'string','enum':list(SKILLS)}, 'object':{'type':['string','null']},
        'goal':{'type':'string'}, 'strategy':{'type':['string','null'],'enum':[None,'top','side','handle']}}}}}
SYSTEM = ('You are the semantic task planner for a single-arm LIBERO adaptation of RoboEval Agentic v2. '
          'Choose exactly one skill. Deterministic robotics code alone chooses poses, offsets, joints, trajectories, gains, tolerances and step counts. '
          'Use only current named objects and regions. Fixtures cannot be lifted. Re-observe after each skill and use failure feedback. '
          'Do not claim success without official check_success. BDDL goals are explicitly provided as privileged task information. '
          'Return the supplied strict JSON schema. thought is a brief decision rationale.')

class PlannerError(RuntimeError):
    pass

def parse_response(response):
    if response.get('status') != 'completed':
        raise PlannerError('incomplete_response: ' + str(response.get('incomplete_details')))
    returned = response.get('model','')
    if returned != 'gpt-6-astra' and not returned.startswith('gpt-6-astra-'):
        raise PlannerError('model_mismatch: '+returned)
    chunks = [c['text'] for item in response.get('output',[]) if item.get('type') == 'message' for c in item.get('content',[]) if c.get('type') == 'output_text']
    try:
        value = json.loads(''.join(chunks))
        import jsonschema
        jsonschema.validate(value, SCHEMA)
    except Exception as exc:
        raise PlannerError('invalid_structured_output: '+type(exc).__name__) from exc
    return value

class GPTPlanner:
    def __init__(self, config, directory):
        self.config, self.directory = config, directory
        self.calls = 0
        self.metadata = []
        if config['model'] != 'gpt-6-astra':
            raise PlannerError('Only requested model gpt-6-astra is permitted')

    def decide(self, task, scene, history):
        if self.calls >= self.config['max_llm_calls']:
            raise PlannerError('llm_call_budget')
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise PlannerError('missing_OPENAI_API_KEY')
        def compact(x):
            if isinstance(x,float): return round(x,4)
            if isinstance(x,dict): return {k:compact(v) for k,v in x.items()}
            if isinstance(x,list): return [compact(v) for v in x]
            return x
        feedback=[{k:h[k] for k in ['request','success','failure_code','message','official_success'] if k in h} for h in history[-6:]]
        payload = {'model':self.config['model'], 'store':False,
                   'reasoning':{'effort':self.config['reasoning_effort']},
                   'max_output_tokens':self.config['max_output_tokens'],
                   'input':[{'role':'system','content':SYSTEM}, {'role':'user','content':json.dumps(compact({'task':task, 'skills':SKILLS, 'scene':scene, 'history':feedback}))}],
                   'text':{'format':{'type':'json_schema','name':'libero_semantic_skill','strict':True,'schema':SCHEMA}}}
        self.calls += 1
        stem = self.directory / ('llm_%03d' % self.calls)
        dump(stem.with_suffix('.request.json'), payload)
        start = time.monotonic()
        meta = {'requested_model':self.config['model'],'returned_model':None,'latency_seconds':None,'usage':{},'error':None}
        reservation=None
        try:
            from .budget import reserve,settle
            reservation=reserve(payload,self.config)
            url = os.getenv('OPENAI_BASE_URL','https://api.openai.com/v1').rstrip('/') + '/responses'
            request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=self.config['api_timeout_seconds']) as handle:
                response = json.load(handle)
            dump(stem.with_suffix('.response.json'), response)
            meta.update(returned_model=response.get('model'),usage=response.get('usage',{}),response_id=response.get('id'))
            settle(reservation,meta['usage'],self.config)
            return parse_response(response)
        except urllib.error.HTTPError as exc:
            # Save the API error body, but never the Authorization header/key.
            body = exc.read().decode('utf-8',errors='replace').replace(key,'[REDACTED]')
            dump(stem.with_suffix('.error.json'), {'http_status':exc.code,'body':body})
            meta['error'] = 'http_%d' % exc.code
            raise PlannerError(meta['error']) from None
        except Exception as exc:
            meta['error'] = str(exc).replace(key,'[REDACTED]')
            raise PlannerError(meta['error']) from None
        finally:
            meta['latency_seconds'] = time.monotonic()-start
            self.metadata.append(meta)
            dump(stem.with_suffix('.metadata.json'), meta)
